// Chapter 8 interactive page (long memory and ARFIMA): one block per chart slide, plus the result tables and the worked
// examples of the lecture, computed in the browser from ch8_data.js.
(function () {
'use strict';
const D = window.CH8_DATA;
const { sum, mean, nf, $, AX, layout, plot, Player, range, predict, exitTicket } = window.TSAI;
const out = (id, items) => window.TSAI.out(id, items);
const T = window.CH8_TEXT;
const L = T.L;
const NB = D.numbers;

// ---------------------------------------------------------------- colours of generate_all_charts.py / tsa_style.py
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', purple: '#8E44AD', teal: '#17A2B8', orange: '#E67E22', navy: '#1F2A44' };
const RGB = { [K.blue]: '26,58,110', [K.red]: '205,0,0', [K.green]: '46,125,50', [K.purple]: '142,68,173', [K.amber]: '181,133,63',
  [K.teal]: '23,162,184', [K.orange]: '230,126,34', [K.navy]: '31,42,68' };
const COL = { nile: K.blue, infl: K.red, infl12: K.orange, usinfl: K.green, unrate: K.purple, sp500: K.blue, bet: K.red, eurron: K.green };

// ---------------------------------------------------------------- small helpers
const set = (id, v) => { const e = $(id + '-v'); if (e) e.textContent = v; };
const has = id => !!$(id);
const cache = {};
const memo = (k, f) => (k in cache ? cache[k] : (cache[k] = f()));
const f2 = x => nf(x, 2), f3 = x => nf(x, 3);
const ci = (d, se) => [d - 1.96 * se, d + 1.96 * se];
const sci = x => { if (x === 0) return nf(0, 1); const e = Math.floor(Math.log10(Math.abs(x))); return e >= -4 ? nf(x, Math.max(3, 1 - e)) : `${nf(x / 10 ** e, 1)} · 10<sup>${String(e).replace('-', '−')}</sup>`; };
const cumsum = x => { let s = 0; return x.map(v => (s += v)); };
const sdp = x => { const m = mean(x); return Math.sqrt(sum(x.map(v => (v - m) ** 2)) / x.length); };
const slope = (x, y) => { const mx = mean(x), my = mean(y); let a = 0, b = 0; for (let i = 0; i < x.length; i++) { a += (x[i] - mx) * (y[i] - my); b += (x[i] - mx) ** 2; } return a / b; };
const isoM = (y0, m0, k) => { const mm = m0 - 1 + k, y = y0 + Math.floor(mm / 12), m = mm % 12 + 1; return `${y}-${String(m).padStart(2, '0')}-01`; };
const months = (iso, n) => Array.from({ length: n }, (_, k) => isoM(+iso.slice(0, 4), +iso.slice(5, 7), k));
const days = (d0, gap) => { const d = [d0]; let t = Date.parse(d0 + 'T00:00:00Z'); gap.forEach(g0 => { t += g0 * 864e5; d.push(new Date(t).toISOString().slice(0, 10)); }); return d; };
const idxFrom = (d, iso) => { const i = d.findIndex(x => x >= iso); return i < 0 ? d.length : i; };
const idxTo = (d, iso) => { let i = d.length - 1; while (i > 0 && d[i] > iso) i--; return i; };

// ---------------------------------------------------------------- special functions, FFT
const LG = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
function lgamma(x) {                         // log |Gamma(x)| (Lanczos)
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1; let a = LG[0]; const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LG[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
const gammaf = x => (x < 0.5 ? Math.PI / (Math.sin(Math.PI * x) * gammaf(1 - x)) : Math.exp(lgamma(x)));
function fft2(re, im) {                      // in place, length a power of 2
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let len = 2; len <= n; len <<= 1) {
    const h = len >> 1, cs = new Float64Array(h), sn = new Float64Array(h);
    for (let j = 0; j < h; j++) { cs[j] = Math.cos(-2 * Math.PI * j / len); sn[j] = Math.sin(-2 * Math.PI * j / len); }
    for (let i = 0; i < n; i += len) for (let j = 0; j < h; j++) {
      const a = i + j, b = a + h, tr = re[b] * cs[j] - im[b] * sn[j], ti = re[b] * sn[j] + im[b] * cs[j];
      re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
    }
  }
}
function fft(xr, xi) {                       // DFT of any length (Bluestein); returns [re, im]
  const n = xr.length;
  if (!(n & (n - 1))) { const re = Float64Array.from(xr), im = xi ? Float64Array.from(xi) : new Float64Array(n); fft2(re, im); return [re, im]; }
  let m = 1; while (m < 2 * n - 1) m <<= 1;
  const wr = new Float64Array(n), wi = new Float64Array(n);
  for (let k = 0; k < n; k++) { const a = Math.PI * ((k * k) % (2 * n)) / n; wr[k] = Math.cos(a); wi[k] = -Math.sin(a); }
  const ar = new Float64Array(m), ai = new Float64Array(m), br = new Float64Array(m), bi = new Float64Array(m);
  for (let k = 0; k < n; k++) {
    const r = xr[k], i = xi ? xi[k] : 0; ar[k] = r * wr[k] - i * wi[k]; ai[k] = r * wi[k] + i * wr[k];
    br[k] = wr[k]; bi[k] = -wi[k]; if (k) { br[m - k] = wr[k]; bi[m - k] = -wi[k]; }
  }
  fft2(ar, ai); fft2(br, bi);
  for (let k = 0; k < m; k++) { const r = ar[k] * br[k] - ai[k] * bi[k], i = ar[k] * bi[k] + ai[k] * br[k]; ar[k] = r; ai[k] = -i; }
  fft2(ar, ai);
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let k = 0; k < n; k++) { const cr = ar[k] / m, cj = -ai[k] / m; re[k] = cr * wr[k] - cj * wi[k]; im[k] = cr * wi[k] + cj * wr[k]; }
  return [re, im];
}

// ---------------------------------------------------------------- the chapter's estimators (ports of generate_all_charts.py)
function pgram(x) {                          // I(lambda_j) = |sum (x_t - mean) e^{-i lambda_j t}|^2 / (2 pi T), j = 1..[T/2]
  const n = x.length, m = mean(x), [re, im] = fft(x.map(v => v - m)), J = Math.floor(n / 2), lam = new Float64Array(J), I = new Float64Array(J);
  for (let j = 1; j <= J; j++) { lam[j - 1] = 2 * Math.PI * j / n; I[j - 1] = (re[j] ** 2 + im[j] ** 2) / (2 * Math.PI * n); }
  return { lam, I, n };
}
const bwm = (n, a) => Math.floor(Math.pow(n, a));
const gphX = l => -Math.log(4 * Math.sin(l / 2) ** 2);
function gph(P, m) {
  const X = [], Y = []; for (let j = 0; j < m; j++) { X.push(gphX(P.lam[j])); Y.push(Math.log(P.I[j])); }
  const d = slope(X, Y), c = mean(Y) - d * mean(X);
  return { d, se: Math.PI / Math.sqrt(24 * m), m, X, Y, c };
}
function lwObj(P, m) {
  let ml = 0; for (let j = 0; j < m; j++) ml += Math.log(P.lam[j]); ml /= m;
  return d => { let s = 0; for (let j = 0; j < m; j++) s += Math.exp(2 * d * Math.log(P.lam[j])) * P.I[j]; return Math.log(s / m) - 2 * d * ml; };
}
function argmin(f, a, b, tol) {
  const g = (Math.sqrt(5) - 1) / 2; let c = b - g * (b - a), d = a + g * (b - a), fc = f(c), fd = f(d);
  while (b - a > (tol || 1e-9)) { if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); } else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); } }
  return (a + b) / 2;
}
function lw(P, m) { const R = lwObj(P, m), d = argmin(R, -0.49, 1.49); return { d, se: 0.5 / Math.sqrt(m), m, R }; }
function acfv(x, Lm) {
  const m = mean(x), d = x.map(v => v - m); let g0 = 0; for (const v of d) g0 += v * v;
  const r = []; for (let k = 1; k <= Lm; k++) { let s = 0; for (let t = k; t < d.length; t++) s += d[t] * d[t - k]; r.push(s / g0); } return r;
}
function bsizes(n, nmin, nmax, num) {
  nmin = nmin || 10; nmax = nmax || Math.floor(n / 4); num = num || 20;
  const a = Math.log10(nmin), b = Math.log10(nmax), st = (b - a) / (num - 1), s = new Set();
  for (let i = 0; i < num; i++) s.add(Math.floor(Math.pow(10, i === num - 1 ? b : a + i * st)));
  return [...s].sort((p, q) => p - q);
}
function rsCurve(x, sizes) {
  return sizes.map(n => {
    const m = Math.floor(x.length / n); let acc = 0, cnt = 0;
    for (let b = 0; b < m; b++) {
      let mu = 0; for (let i = 0; i < n; i++) mu += x[b * n + i]; mu /= n;
      let y = 0, mx = -Infinity, mn = Infinity, ss = 0;
      for (let i = 0; i < n; i++) { const dv = x[b * n + i] - mu; y += dv; if (y > mx) mx = y; if (y < mn) mn = y; ss += dv * dv; }
      const S = Math.sqrt(ss / n); if (S > 0) { acc += (mx - mn) / S; cnt++; }
    }
    return acc / cnt;
  });
}
function dfaCurve(x, sizes) {
  const mu = mean(x), Y = new Float64Array(x.length); let c = 0;
  for (let i = 0; i < x.length; i++) { c += x[i] - mu; Y[i] = c; }
  return sizes.map(n => {
    const m = Math.floor(Y.length / n), tc = (n - 1) / 2; let tt = 0, ss = 0; for (let i = 0; i < n; i++) tt += (i - tc) ** 2;
    for (let b = 0; b < m; b++) {
      let bm = 0; for (let i = 0; i < n; i++) bm += Y[b * n + i]; bm /= n;
      let bt = 0; for (let i = 0; i < n; i++) bt += (Y[b * n + i] - bm) * (i - tc);
      const sl = bt / tt; for (let i = 0; i < n; i++) { const e = Y[b * n + i] - bm - sl * (i - tc); ss += e * e; }
    }
    return Math.sqrt(ss / (m * n));
  });
}
const logSlope = (n, v) => slope(n.map(Math.log10), v.map(Math.log10));
const hRS = (x, s) => logSlope(s || bsizes(x.length), rsCurve(x, s || bsizes(x.length)));
const hDFA = (x, s) => logSlope(s || bsizes(x.length), dfaCurve(x, s || bsizes(x.length)));
function fw(d, n) { const w = new Float64Array(n); w[0] = 1; for (let k = 1; k < n; k++) w[k] = w[k - 1] * (k - 1 - d) / k; return w; }
function acvf0(d, n) { const g = new Float64Array(n); g[0] = Math.exp(lgamma(1 - 2 * d) - 2 * lgamma(1 - d)); for (let k = 1; k < n; k++) g[k] = g[k - 1] * (k - 1 + d) / (k - d); return g; }
const acf0 = (d, n) => { const g = acvf0(d, n); return Array.from(g, v => v / g[0]); };
function armaPsi(phi, theta, n) {
  const psi = new Float64Array(n); psi[0] = 1;
  for (let j = 1; j < n; j++) { let s = j <= theta.length ? theta[j - 1] : 0; for (let i = 0; i < Math.min(phi.length, j); i++) s += phi[i] * psi[j - 1 - i]; psi[j] = s; }
  return psi;
}
function acvf(d, phi, theta, n) {             // ARFIMA(p,d,q) autocovariances (unit variance of the shocks), p <= 1
  if (!phi.length && !theta.length) return acvf0(d, n);
  const r = phi.length ? Math.abs(phi[0]) : 0;
  let Kt = r === 0 ? 1 + theta.length : Math.floor(Math.min(3000, Math.max(50, Math.ceil(Math.log(1e-10) / Math.log(r)))));
  Kt = Math.max(Kt, theta.length + 1);
  const psi = armaPsi(phi, theta, Kt), c = new Float64Array(Kt);
  for (let m = 0; m < Kt; m++) { let s = 0; for (let i = 0; i < Kt - m; i++) s += psi[i] * psi[i + m]; c[m] = s; }
  const gU = acvf0(d, n + Kt), g = new Float64Array(n);
  for (let h = 0; h < n; h++) { let s = c[0] * gU[h]; for (let m = 1; m < Kt; m++) s += c[m] * (gU[h + m] + gU[Math.abs(h - m)]); g[h] = s; }
  return g;
}
function dlLik(x0, g) {                       // exact Gaussian log-likelihood (Durbin-Levinson), mean = sample mean, sigma^2 concentrated out
  const m = mean(x0), x = x0.map(v => v - m), n = x.length;
  let phi = new Float64Array(0), v = g[0], Q = x[0] * x[0] / v, Ld = Math.log(v);
  for (let t = 1; t < n; t++) {
    let s = g[t]; for (let j = 0; j < phi.length; j++) s -= phi[j] * g[t - 1 - j];
    const k = s / v, np_ = new Float64Array(t);
    for (let j = 0; j < t - 1; j++) np_[j] = phi[j] - k * phi[t - 2 - j];
    np_[t - 1] = k; phi = np_; v *= 1 - k * k;
    let e = x[t]; for (let j = 0; j < t; j++) e -= phi[j] * x[t - 1 - j];
    Q += e * e / v; Ld += Math.log(v);
  }
  return -0.5 * (n * Math.log(2 * Math.PI * Q / n) + Ld + n);
}
const mlD = x => { const f = d => -dlLik(x, acvf0(d, x.length)), d = argmin(f, -0.49, 0.49, 1e-7); return { d, ll: -f(d) }; };
function circ(ac, Z) {                       // circulant embedding (Davies and Harte) with the draws of the chapter
  const n = ac.length, M = 2 * (n - 1), c = new Float64Array(M);
  for (let i = 0; i < n; i++) c[i] = ac[i];
  for (let i = 1; i < n - 1; i++) c[M - i] = ac[i];
  const [lr] = fft(c), re = new Float64Array(M), im = new Float64Array(M);
  for (let k = 0; k < M; k++) { const s = Math.sqrt(Math.max(lr[k], 0) / M); re[k] = s * Z.re[k]; im[k] = s * Z.im[k]; }
  return Array.from(fft(re, im)[0].slice(0, n));
}
const fgnAcvf = (n, H) => Array.from({ length: n }, (_, k) => 0.5 * (Math.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + Math.abs(k - 1) ** (2 * H)));
function cholSolve(A, b) {
  const n = A.length, Lc = A.map(r => r.slice());
  for (let j = 0; j < n; j++) { let s = Lc[j][j]; for (let k = 0; k < j; k++) s -= Lc[j][k] * Lc[j][k]; Lc[j][j] = Math.sqrt(s);
    for (let i = j + 1; i < n; i++) { let t = Lc[i][j]; for (let k = 0; k < j; k++) t -= Lc[i][k] * Lc[j][k]; Lc[i][j] = t / Lc[j][j]; } }
  const fw_ = v => { const y = v.slice(); for (let i = 0; i < n; i++) { for (let k = 0; k < i; k++) y[i] -= Lc[i][k] * y[k]; y[i] /= Lc[i][i]; } return y; };
  const bw_ = y => { const x = y.slice(); for (let i = n - 1; i >= 0; i--) { for (let k = i + 1; k < n; k++) x[i] -= Lc[k][i] * x[k]; x[i] /= Lc[i][i]; } return x; };
  return { x: b ? bw_(fw_(b)) : null, inv: () => { const I = []; for (let j = 0; j < n; j++) { const e = new Array(n).fill(0); e[j] = 1; I.push(bw_(fw_(e))); } return I; } };
}
function ols(y, row, k) {
  const n = y.length, G = Array.from({ length: k }, () => new Array(k).fill(0)), h = new Array(k).fill(0);
  for (let t = 0; t < n; t++) { const x = row(t); for (let a = 0; a < k; a++) { h[a] += x[a] * y[t]; for (let c = a; c < k; c++) G[a][c] += x[a] * x[c]; } }
  for (let a = 0; a < k; a++) for (let c = 0; c < a; c++) G[a][c] = G[c][a];
  const S = cholSolve(G, h), b = S.x; let ssr = 0;
  for (let t = 0; t < n; t++) { const x = row(t); let f = 0; for (let a = 0; a < k; a++) f += x[a] * b[a]; ssr += (y[t] - f) ** 2; }
  return { b, ssr, n, k, inv: S.inv };
}
function adf1(x) {                           // statsmodels adfuller(x, maxlag=1, regression='c', autolag=None): statistic and 5% value
  const dx = []; for (let t = 1; t < x.length; t++) dx.push(x[t] - x[t - 1]);
  const nobs = dx.length - 1, y = dx.slice(1), f = ols(y, t => [x[t + 1], dx[t], 1], 3), iv = f.inv();
  const s2 = f.ssr / (f.n - 3), crit = D.mk.c[1].reduce((s, c, i) => s + c / nobs ** i, 0);
  return { stat: f.b[0] / Math.sqrt(s2 * iv[0][0]), crit, nobs };
}
function ffd(x, d, thresh) {                 // (1 - L)^d with the fixed window of weights above thresh
  if (d === 0) return { y: x.slice(), K: 1 };
  const w = fw(d, x.length); let K = w.findIndex(v => Math.abs(v) < thresh); K = K < 0 ? w.length : Math.max(K, 2);
  const y = new Float64Array(x.length - K + 1);
  for (let t = K - 1; t < x.length; t++) { let s = 0; for (let j = 0; j < K; j++) s += w[j] * x[t - j]; y[t - K + 1] = s; }
  return { y: Array.from(y), K };
}
const corr = (a, b) => { const ma = mean(a), mb = mean(b); let s = 0, sa = 0, sb = 0; for (let i = 0; i < a.length; i++) { s += (a[i] - ma) * (b[i] - mb); sa += (a[i] - ma) ** 2; sb += (b[i] - mb) ** 2; } return s / Math.sqrt(sa * sb); };
function figarchW(d, phi, beta, n) {          // ARCH(infinity) weights of FIGARCH(1,d,1)
  const w = fw(d, n), a = new Float64Array(n);
  for (let k = 0; k < n; k++) a[k] = w[k] - (k ? phi * w[k - 1] : 0);
  const b = new Float64Array(n);
  for (let k = 0; k < n; k++) { let s = 0, p = 1; for (let j = k; j >= 0; j--) { s += a[j] * p; p *= beta; } b[k] = s; }
  return Array.from(b.slice(1), v => -v);
}
function arfimaFc(x, d, phi, H) {             // forecasts from the AR(infinity) form, all observations used
  const n = x.length, mu = mean(x), z = new Float64Array(n + H), w = fw(d, n + H), pi = new Float64Array(n + H);
  for (let k = 0; k < n + H; k++) pi[k] = w[k] - (k ? phi * w[k - 1] : 0);
  for (let t = 0; t < n; t++) z[t] = x[t] - mu;
  for (let t = n; t < n + H; t++) { let s = 0; for (let k = 1; k <= t; k++) s += pi[k] * z[t - k]; z[t] = -s; }
  return Array.from(z.slice(n), v => v + mu);
}
const sa = x => {                            // monthly means removed, sample mean added back (x starts in January)
  const mm = new Array(12).fill(0), cnt = new Array(12).fill(0); x.forEach((v, t) => { mm[t % 12] += v; cnt[t % 12]++; });
  const mu = mean(x); return x.map((v, t) => v - mm[t % 12] / cnt[t % 12] + mu);
};

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
  const gx = o.gx || 0.08, gy = Math.min(0.5, (narrow ? 62 + extra : (o.gyPx || 62)) / plotH), w = (1 - gx * (C - 1)) / C, h = (1 - gy * (R - 1)) / R;
  const lay = layout({ height: H, margin: { l: o.ml || 52, r: o.mr || 10, t: 26 + extra, b: 34 }, annotations: [], shapes: [], showlegend: o.legend !== false,
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
const hband = (a, y0, y1, col, al) => ({ type: 'rect', xref: a.xd, yref: a.ya, x0: 0, x1: 1, y0, y1, fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, layer: 'below' });
const leg = (name, line, marker) => Object.assign({ x: [null], y: [null], name, hoverinfo: 'skip' }, line ? { mode: 'lines', line } : { mode: 'markers', marker });
const ln = (x, y, a, color, name, o) => { const r = Object.assign({ x, y, mode: 'lines', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }, o || {}); r.line = Object.assign({ color, width: 1.4 }, (o && o.line) || {}); return r; };
const mk = (x, y, a, color, name, o) => Object.assign({ x, y, mode: 'markers', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, marker: { color, size: 6 }, hovertemplate: '%{x}: %{y:.4f}<extra></extra>' }, o || {});
const bar = (x, y, a, color, name, o) => Object.assign({ x, y, type: 'bar', marker: { color }, name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }, o || {});
const band = (x, lo, hi, a, col, al, name) => ({ x: [...x, ...[...x].reverse()], y: [...hi, ...[...lo].reverse()], fill: 'toself', fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, mode: 'lines', hoverinfo: 'skip', name: name || '', showlegend: !!name, xaxis: a.xa, yaxis: a.ya });
const DATE = { type: 'date' }, LOG = { type: 'log', exponentformat: 'power' };
const legBand = name => leg(name, null, { color: `rgba(${RGB[K.teal]},0.35)`, symbol: 'square', size: 12 });

// ---------------------------------------------------------------- tables and worked steps (HTML)
function table(id, head, rows, o) {
  const el = $(id); if (!el) return;
  const cell = (c, tag) => { const v = typeof c === 'object' && c !== null ? c : { v: c }; return `<${tag}${v.cls ? ` class="${v.cls}"` : ''}>${v.v}</${tag}>`; };
  el.innerHTML = (o && o.cap ? `<p class="c8cap">${o.cap}</p>` : '') + '<table><thead><tr>' + head.map(c => cell(c, 'th')).join('') + '</tr></thead><tbody>'
    + rows.map(r => '<tr' + (r.hi ? ' class="hi"' : '') + '>' + r.map(c => cell(c, 'td')).join('') + '</tr>').join('') + '</tbody></table>' + (o && o.note ? `<p class="c8cap">${o.note}</p>` : '');
}
const num = (x, d, bold) => ({ v: bold ? `<b>${nf(x, d)}</b>` : nf(x, d), cls: 'r' });
const txt = (v, bold) => ({ v: bold ? `<b>${v}</b>` : v, cls: 'r' });
const rh = { v: '', cls: 'r' };
const H_ = v => ({ v, cls: 'r' });
const steps = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
const civ = (d, se) => { const c = ci(d, se); return `[${f2(c[0])}; ${f2(c[1])}]`.replace('; ', T.sep); };

// ---------------------------------------------------------------- data of the chapter
const NILE = { y: D.data.nile.v.map((_, k) => D.data.nile.y0 + k), v: D.data.nile.v };
const ROd = months(D.data.ro.m0, D.data.ro.raw.length);
const RO = { d: ROd, raw: D.data.ro.raw, i12: D.data.ro.i12, im: sa(D.data.ro.raw) };
const US = { d: D.data.us.d.map(m => m + '-01'), v: D.data.us.v, ud: D.data.us.ud.map(m => m + '-01'), u: D.data.us.u };
const RV = { d: months(D.data.rv.m0, D.data.rv.sp500.length), sp500: D.data.rv.sp500, bet: D.data.rv.bet };
const SPd = days(D.data.sp.d0, D.data.sp.gap), LP = D.data.sp.lp;
const SP = { d: SPd.slice(1), r: LP.slice(1).map((v, t) => 100 * (v - LP[t])) };
const BET = { d: days(D.data.bet.d0, D.data.bet.gap), r: D.data.bet.r };
const EUR = { r: D.data.eur.r };
const PK = { d: days(D.data.park.d0, D.data.park.gap), v: D.data.park.v };
const absv = x => x.map(Math.abs);
const PG = (key, x) => memo('pg|' + key, () => pgram(x));
const VOL = NB.volm;

// ================================================================= 1. four persistent series
function maTrail(x, k) { const o = new Array(x.length).fill(null); let s = 0; for (let t = 0; t < x.length; t++) { s += x[t]; if (t >= k) s -= x[t - k]; if (t >= k - 1) o[t] = s / k; } return o; }
function longestRun(x, mu) { let best = 0, cur = 0, side = 0; x.forEach(v => { if (v === null) return; const s = v > mu ? 1 : -1; cur = s === side ? cur + 1 : 1; side = s; best = Math.max(best, cur); }); return best; }
function blkMd() {
  const k = +$('md-k').value; set('md-k', k);
  const G = grid('md-chart', [{ r: 0, c: 0, t: 'Nile, annual flow', x: {} }, { r: 0, c: 1, t: 'Romania, HICP inflation', x: DATE }, { r: 1, c: 0, t: 'United States, unemployment rate', x: DATE },
    { r: 1, c: 1, t: 'Monthly realised volatility, % (log scale)', x: DATE, y: { type: 'log', tickvals: [2, 5, 10, 20, 50], ticktext: ['2', '5', '10', '20', '50'] } }], { cols: 2, h: 520, legendRows: 3 });
  const A = G.ax, mo = { line: { color: K.navy, width: 1.6, dash: 'dot' } };
  const rvs = RV.sp500.map(Math.exp), rvb = RV.bet.map(Math.exp);
  const tr = [ln(NILE.y, NILE.v, A[0], COL.nile, 'Nile flow at Aswan (10^8 m^3), 1871-1970', { line: { width: 1.2 } }),
    bar(RO.d, RO.im.map(v => 12 * v), A[1], `rgba(${RGB[K.red]},0.6)`, 'RO inflation, monthly SA, annualised (x12, %)'),
    ln(RO.d, RO.i12, A[1], COL.infl12, 'RO inflation, 12-month (%)'), ln(US.ud, US.u, A[2], COL.unrate, 'US unemployment rate (%)', { line: { width: 1.1 } }),
    ln(RV.d, rvs, A[3], COL.sp500, 'S&P 500: monthly realised volatility (%)', { line: { width: 1 } }), ln(RV.d, rvb, A[3], COL.bet, 'BET: monthly realised volatility (%)', { line: { width: 1 } })];
  G.lay.shapes.push(hl(A[1], 0, K.navy, 'solid', 0.5));
  const S = [NILE.v, RO.i12, US.u, RV.sp500];
  if (k > 1) {
    tr.push(ln(NILE.y, maTrail(NILE.v, k), A[0], K.navy, `moving average of the last ${k} observations`, mo), ln(RO.d, maTrail(RO.i12, k), A[1], K.navy, '', mo),
      ln(US.ud, maTrail(US.u, k), A[2], K.navy, '', mo), ln(RV.d, maTrail(RV.sp500, k).map(v => (v === null ? null : Math.exp(v))), A[3], K.navy, '', mo),
      ln(RV.d, maTrail(RV.bet, k).map(v => (v === null ? null : Math.exp(v))), A[3], K.navy, '', { line: { color: K.navy, width: 1.6, dash: 'dash' } }));
  }
  plot('md-chart', tr, G.lay);
  const runs = S.map(x => longestRun(maTrail(x, k), mean(x)));
  const D8 = NB.data;
  out('md-out', [T.md1(k, runs), T.md2(nf(D8.nile_mean, 0), nf(D8.i12_max, 1), T.my(D8.i12_max_d), nf(D8.i12_last, 1), T.my(D8.i12_last_d), nf(D8.u_max, 1), T.my(D8.u_max_d),
    nf(D8.u_last, 1), T.my(D8.u_last_d), nf(D8.rv_max.sp500, 0), T.my(D8.rv_max_d.sp500), nf(D8.rv_max.bet, 0), T.my(D8.rv_max_d.bet))]);
}

// ================================================================= 2. sample ACF against an AR(1)
const ACFS = () => [['nile', NILE.v, 30, 'Nile flow'], ['infl', RO.im, 48, 'RO inflation, monthly SA'], ['usinfl', US.v, 48, 'US inflation, monthly, annualised'],
  ['sp500', RV.sp500, 48, 'S&P 500, log monthly realised volatility']];
function acfState(k) {
  return ACFS().map(([key, x0, Lm, t]) => { const x = x0.slice(Math.floor(x0.length * k / 100)), r = acfv(x, Lm), band = 1.96 / Math.sqrt(x.length);
    return { key, t, n: x.length, Lm, r, band, npos: r.filter(v => v > band).length, ar10: r[0] ** 10 }; });
}
function blkMa() {
  const k = +$('ma-k').value; set('ma-k', k + '%');
  const S = acfState(k);
  const G = grid('ma-chart', S.map((s, i) => ({ r: Math.floor(i / 2), c: i % 2, t: s.t, xt: 'lag k' })), { cols: 2, h: 540, legendRows: 1 });
  const tr = [];
  S.forEach((s, i) => { const a = G.ax[i], lags = range(1, s.Lm, 1);
    tr.push(bar(lags, s.r, a, `rgba(${RGB[COL[s.key]]},0.7)`, ''), ln(lags, lags.map(l => s.r[0] ** l), a, K.navy, i ? '' : 'AR(1) with the same lag-1 autocorrelation', { line: { color: K.navy, dash: 'dash', width: 1.2 } }));
    G.lay.shapes.push(hband(a, -s.band, s.band, K.teal, 0.15), hl(a, 0, K.navy, 'solid', 0.5)); });
  tr.push(legBand('95% band for white noise'));
  plot('ma-chart', tr, G.lay);
  out('ma-out', [T.ma1(k, S.map(s => [s.key, s.n, f2(s.r[0]), f2(s.r[9]), sci(s.ar10), s.npos, s.Lm]))]);
}

// ================================================================= 3. hyperbolic against exponential decay
function decayState(d) {
  const r = acf0(d, 1001).slice(1), phi = r[0], C = gammaf(1 - d) / gammaf(d);
  return { r, phi, C, s100: sum(r.slice(0, 100)), s1000: sum(r), asum: phi / (1 - phi) };
}
function blkDc() {
  const d = +$('dc-d').value; set('dc-d', f2(d));
  const S = decayState(d), k = range(1, 1000, 1), k50 = k.slice(0, 50);
  const G = grid('dc-chart', [{ r: 0, c: 0, xt: 'lag k', yt: 'autocorrelation' }, { r: 0, c: 1, xt: 'lag k (log scale)', yt: 'autocorrelation (log scale)', x: LOG, y: { type: 'log', exponentformat: 'power', range: [-4, 0] } }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax;
  plot('dc-chart', [ln(k50, S.r.slice(0, 50), a, K.blue, `ARFIMA(0,d,0), d = ${d.toFixed(2)}: hyperbolic decay`, { mode: 'lines+markers', marker: { size: 4, color: K.blue } }),
    ln(k50, k50.map(j => S.phi ** j), a, K.orange, `AR(1), phi = ${S.phi.toFixed(3)}: exponential decay`, { mode: 'lines+markers', marker: { size: 4, symbol: 'square', color: K.orange } }),
    ln(k, S.r, b, K.blue, ''), ln(k, k.map(j => S.phi ** j).map(v => (v < 1e-6 ? null : v)), b, K.orange, ''),
    ln(k, k.map(j => S.C * j ** (2 * d - 1)), b, K.navy, 'C k^(2d-1): the power law', { line: { color: K.navy, dash: 'dash', width: 1 } })], G.lay);
  out('dc-out', [T.dc1(f2(d), f3(S.phi), f3(S.r[9]), sci(S.phi ** 10), f3(S.r[49]), sci(S.phi ** 50), f3(S.r[999])), T.dc2(f2(2 * d - 1), nf(S.s100, 1), nf(S.s1000, 1), nf(S.asum, 1))]);
}

// ================================================================= 4. fractional differencing and integration
function blkWt() {
  const d = +$('wt-d').value, ph = +$('wt-p').value; set('wt-d', f2(d)); set('wt-p', f2(ph));
  const ds = [[0.2, K.green], [d, K.blue], [0.7, K.purple], [1.0, K.red]], off = [-0.3, -0.1, 0.1, 0.3], k10 = range(1, 10, 1), kk = range(1, 200, 1);
  const G = grid('wt-chart', [{ r: 0, c: 0, xt: 'lag k', yt: 'weight pi_k of (1-L)^d' }, { r: 0, c: 1, xt: 'horizon k (log scale)', yt: 'impulse response psi_k of (1-L)^-d', x: LOG }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax, tr = [];
  ds.forEach(([dd, c], i) => { const w = fw(dd, 11), psi = fw(-dd, 201);
    tr.push(bar(k10.map(v => v + off[i]), Array.from(w.slice(1)), a, c, `d = ${+dd.toFixed(2)}`, { width: 0.2 }), ln(kk, Array.from(psi.slice(1)), b, c, '', { line: { width: 1.5 } })); });
  tr.push(ln(kk, kk.map(j => ph ** j), b, K.orange, `AR(1), phi = ${ph.toFixed(2)}`, { line: { color: K.orange, dash: 'dash', width: 1.5 } }));
  G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.5)); G.lay.barmode = 'overlay';
  plot('wt-chart', tr, G.lay);
  const psi = fw(-d, 101), w = fw(d, 3);
  out('wt-out', [T.wt1(f2(d), f3(w[1]), f3(w[2]), f3(psi[10]), f3(psi[100]), f2(ph), f3(ph ** 10), sci(ph ** 100))]);
}

// ================================================================= 5. the degree of differencing a price needs
const GRID20 = range(0, 1, 0.05);
function ffdState(y0, th) {
  return memo(`ffd|${y0}|${th}`, () => {
    const i0 = idxFrom(SPd, `${y0}-01-01`), p = LP.slice(i0), adf = [], cor = [], wid = [];
    GRID20.forEach(d => { const f = ffd(p, d, th); adf.push(adf1(f.y).stat); cor.push(corr(f.y, p.slice(f.K - 1))); wid.push(f.K); });
    const crit = adf1(p).crit, j = adf.findIndex(v => v < crit);
    return { n: p.length, d0: SPd[i0], adf, cor, wid, crit, j };
  });
}
function blkFf() {
  const y0 = +$('ff-y').value, th = +$('ff-t').value; set('ff-y', y0);
  const S = ffdState(y0, th);
  const G = grid('ff-chart', [{ r: 0, c: 0, xt: 'd', yt: 'ADF statistic' }, { r: 0, c: 1, xt: 'd', yt: 'correlation', y: { range: [-0.05, 1.05] } }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax;
  plot('ff-chart', [ln(GRID20, S.adf, a, K.blue, 'ADF statistic of (1-L)^d log S&P 500', { mode: 'lines+markers', marker: { size: 5, color: K.blue } }),
    ln([0, 1], [S.crit, S.crit], a, K.red, `5% critical value (${S.crit.toFixed(2)})`, { line: { color: K.red, dash: 'dash', width: 1.1 } }),
    ln(GRID20, S.cor, b, K.green, 'correlation with the log level', { mode: 'lines+markers', marker: { size: 5, symbol: 'square', color: K.green } })], G.lay);
  const j = S.j;
  out('ff-out', [T.ff1(T.dmy(S.d0), S.n, j >= 0 ? f2(GRID20[j]) : '–', j >= 0 ? f3(S.cor[j]) : '–', j >= 0 ? S.wid[j] : '–', f2(S.crit)),
    T.ff2(f2(S.adf[0]), nf(S.adf[20], 1), f2(S.cor[20]))]);
}

// ================================================================= 6. ARFIMA(0,d,0) with d and -d
const SIMZ = D.sim;
function pathState(d) {
  return memo('path|' + d, () => [d, -d].map((dd, row) => { const x = circ(Array.from(acvf0(dd, 1000)), SIMZ.path[row]), th = acf0(dd, 51).slice(1);
    return { d: dd, x, th, acf: SIMZ.acf[d.toFixed(2)][row], sd: sdp(x), v: acvf0(dd, 1)[0] }; }));
}
function blkAp() {
  const d = +$('ap-d').value; set('ap-d', f2(d));
  const S = pathState(d), t = range(0, 999, 1), k = range(1, 50, 1);
  const G = grid('ap-chart', [{ r: 0, c: 0, cs: 2 }, { r: 0, c: 2 }, { r: 1, c: 0, cs: 2, xt: 'time t' }, { r: 1, c: 2, xt: 'lag k' }], { cols: 3, h: 440, gx: 0.06, legendRows: 2 });
  const tr = [];
  S.forEach((s, row) => { const c = row ? K.red : K.blue, a = G.ax[2 * row], b = G.ax[2 * row + 1];
    tr.push(ln(t, s.x, a, c, `ARFIMA(0,d,0), d = ${s.d.toFixed(2)}: path and sample ACF`, { line: { width: 0.7 } }),
      bar(k, s.acf, b, `rgba(${RGB[c]},0.55)`, ''), ln(k, s.th, b, K.navy, row ? '' : 'theoretical ACF', { line: { color: K.navy, width: 1.2 } }));
    G.lay.shapes.push(hl(b, 0, K.navy, 'solid', 0.5)); });
  plot('ap-chart', tr, G.lay);
  out('ap-out', S.map(s => T.ap1(f2(s.d), f3(s.th[0]), f3(s.th[1]), f2(s.v), f2(s.sd ** 2))));
}

// ================================================================= 7. fractional Brownian motion
function blkFb() {
  const dh = +$('fb-h').value; set('fb-h', f2(dh));
  const Hs = [0.5 - dh, 0.5, 0.5 + dh].map(v => +v.toFixed(2)), cols = [K.red, K.blue, K.green];
  const G = grid('fb-chart', [{ r: 0, c: 0, t: 'fractional Brownian motion B_H(t)', xt: 'time t' }, { r: 0, c: 1, t: 'its increments (fractional Gaussian noise), first 150', xt: 'time t' }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax, tr = [], st = [];
  Hs.forEach((H, i) => { const x = memo('fgn|' + H, () => circ(fgnAcvf(1000, H), SIMZ.fbm)), B = [0, ...cumsum(x)];
    const lab = H === 0.5 ? 'H = 0.5: Brownian motion' : `H = ${H}: ` + (H < 0.5 ? 'anti-persistent' : 'persistent');
    tr.push(ln(range(0, 1000, 1), B, a, cols[i], lab, { line: { width: 1 } }), ln(range(0, 149, 1), x.slice(0, 150), b, cols[i], '', { line: { width: 0.9 } }));
    st.push({ H, a1: acfv(x, 1)[0], th: 2 ** (2 * H - 1) - 1, R: Math.max(...B) - Math.min(...B) }); });
  plot('fb-chart', tr, G.lay);
  out('fb-out', [T.fb1(st.map(s => [nf(s.H, 2), f2(s.a1), f2(s.th)])), T.fb2(st.map(s => [nf(s.H, 2), nf(s.R, 0)]))]);
}

// ================================================================= 8. R/S and DFA
const RSS = () => [['US inflation, monthly', US.v, K.green, 'circle'], ['S&P 500 returns r', SP.r, K.blue, 'square'], ['S&P 500 |r|', absv(SP.r), K.purple, 'diamond']];
function rsState(nmin) {
  return memo('rs|' + nmin, () => RSS().map(([nm, x, c, sym]) => { const s = bsizes(x.length, nmin), rs = rsCurve(x, s), df = dfaCurve(x, s);
    return { nm, c, sym, s, rs, df, hr: logSlope(s, rs), hd: logSlope(s, df) }; }));
}
function blkRd() {
  const nmin = +$('rd-n').value; set('rd-n', nmin);
  const S = rsState(nmin);
  const G = grid('rd-chart', [{ r: 0, c: 0, t: 'R/S analysis', xt: 'block size n (log scale)', yt: `(R/S)_n relative to n = ${nmin}`, x: LOG, y: LOG },
    { r: 0, c: 1, t: 'DFA', xt: 'block size n (log scale)', yt: `F(n) relative to n = ${nmin}`, x: LOG, y: LOG }], { cols: 2, h: 380, legendRows: 4 });
  const tr = [];
  S.forEach(s => { [[G.ax[0], s.rs, s.hr, 'R/S'], [G.ax[1], s.df, s.hd, 'DFA']].forEach(([a, v, h, nm]) =>
    tr.push(ln(s.s, v.map(z => z / v[0]), a, s.c, `${s.nm}, ${nm}: slope ${h.toFixed(2)}`, { mode: 'lines+markers', marker: { size: 5, symbol: s.sym, color: s.c }, line: { width: 1 } }))); });
  G.ax.forEach((a, i) => tr.push(ln([nmin, 1700], [1, (1700 / nmin) ** 0.5], a, K.navy, i ? '' : 'slope 0.5 (no memory)', { line: { color: K.navy, dash: 'dash', width: 1 } })));
  plot('rd-chart', tr, G.lay);
  out('rd-out', [T.rd1(nmin, S.map(s => [s.nm, f2(s.hr), f2(s.hd)]))]);
}

// ================================================================= 9. GPH regressions
const GPS = () => [['nile', 'Nile flow', NILE.v], ['infl', 'RO inflation (monthly, SA)', RO.im]];
function gphState(a, sers) {
  return (sers || GPS()).map(([key, nm, x]) => { const P = PG(key, x), m = bwm(x.length, a), g = gph(P, m), w = lw(P, m);
    return { key, nm, n: x.length, m, g, w, lam: P.lam[m - 1], per: 2 * Math.PI / P.lam[m - 1] }; });
}
function blkGp() {
  const a = +$('gp-a').value; set('gp-a', f2(a));
  const S = gphState(a);
  const G = grid('gp-chart', S.map((s, i) => ({ r: 0, c: i, t: `${s.nm}: d = ${s.g.d.toFixed(2)} (SE ${s.g.se.toFixed(2)}), m = ${s.m}`, xt: '-log(4 sin^2(lambda_j/2))', yt: i ? '' : 'log periodogram' })), { cols: 2, h: 380, legendRows: 2 });
  const tr = [];
  S.forEach((s, i) => { const ax = G.ax[i], x0 = Math.min(...s.g.X), x1 = Math.max(...s.g.X);
    tr.push(mk(s.g.X, s.g.Y, ax, COL[s.key], `${s.nm}: log I(lambda_j), j = 1, ..., m`), ln([x0, x1], [s.g.c + s.g.d * x0, s.g.c + s.g.d * x1], ax, K.navy, i ? '' : 'OLS line, slope = d', { line: { color: K.navy, width: 1.2 } })); });
  plot('gp-chart', tr, G.lay);
  out('gp-out', S.map(s => T.gp1(s.key, f2(s.g.d), f2(s.g.se), s.m, nf(s.per, 1), civ(s.g.d, s.g.se), f2(s.w.d), f2(s.w.se), civ(s.w.d, s.w.se))));
}

// ================================================================= 10. sensitivity to the bandwidth
const AGRID = range(0.4, 0.8, 0.05);
function bwState(Y) {
  return memo('bw|' + Y, () => {
    const end = `${Y}-08-01`;
    const ro = sa(RO.raw.slice(0, idxTo(RO.d, end) + 1)), us = US.v.slice(0, idxTo(US.d, end) + 1);
    return [['infl', 'RO inflation (monthly, SA)', ro], ['usinfl', 'US inflation (monthly, annualised)', us]].map(([key, nm, x]) => {
      const P = pgram(x), G_ = AGRID.map(a => gph(P, bwm(x.length, a)).d), W = AGRID.map(a => lw(P, bwm(x.length, a)));
      return { key, nm, n: x.length, gph: G_, lw: W.map(w => w.d), se: W.map(w => w.se), m: W.map(w => w.m) };
    });
  });
}
function blkBw() {
  const Y = +$('bw-e').value; set('bw-e', T.my(`${Y}-08-01`));
  const S = bwState(Y);
  const G = grid('bw-chart', S.map((s, i) => ({ r: 0, c: i, t: `${s.nm}, T = ${s.n}`, xt: 'bandwidth exponent a, m = [T^a]', yt: i ? '' : 'estimated d' })), { cols: 2, h: 380, legendRows: 2 });
  const tr = [];
  S.forEach((s, i) => { const a = G.ax[i];
    tr.push(band(AGRID, s.lw.map((v, j) => v - 1.96 * s.se[j]), s.lw.map((v, j) => v + 1.96 * s.se[j]), a, K.teal, 0.18, i ? '' : 'local Whittle: 95% interval'),
      ln(AGRID, s.lw, a, K.blue, i ? '' : 'local Whittle', { mode: 'lines+markers', marker: { size: 5, color: K.blue } }),
      ln(AGRID, s.gph, a, K.red, i ? '' : 'GPH', { mode: 'lines+markers', marker: { size: 5, symbol: 'square', color: K.red }, line: { color: K.red, dash: 'dash', width: 1.2 } }));
    G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.5), hl(a, 0.5, K.purple, 'dot', 0.9)); });
  tr.push(leg('d = 0.5 (non-stationarity)', { color: K.purple, dash: 'dot', width: 1 }));
  plot('bw-chart', tr, G.lay);
  out('bw-out', S.map(s => T.bw1(s.key, s.n, s.m[0], s.m[s.m.length - 1], f2(Math.min(...s.lw)), f2(Math.max(...s.lw)), f2(Math.min(...s.gph)), f2(Math.max(...s.gph)))));
}

// ================================================================= 11. Monte Carlo of five estimators
const MCN = D.mc.names, MCC = [K.red, K.blue, K.green, K.amber, K.purple];
const mcStat = (v, d) => ({ mean: mean(v), sd: sdp(v), rmse: Math.sqrt(mean(v.map(x => (x - d) ** 2))) });
function blkMc() {
  const R = +$('mc-r').value; set('mc-r', R);
  const G = grid('mc-chart', D.mc.designs.map((s, i) => ({ r: 0, c: i, t: s.lab.replace('phi', 'φ'), yt: i ? '' : 'estimated d' })), { cols: 3, h: 380, gx: 0.05, legendRows: 2 });
  const tr = [];
  D.mc.designs.forEach((s, i) => { const a = G.ax[i];
    MCN.forEach((n, j) => tr.push({ type: 'box', y: s.v[n].slice(0, R), name: n, x0: n, marker: { color: MCC[j] }, fillcolor: `rgba(${RGB[MCC[j]]},0.55)`, line: { color: K.navy, width: 1 },
      boxpoints: false, showlegend: false, xaxis: a.xa, yaxis: a.ya, hoverinfo: 'y' }));
    G.lay.shapes.push(hl(a, s.d, K.navy, 'dash', 1)); });
  ['GPH', 'local Whittle', 'Whittle ML, ARFIMA(0,d,0)', 'R/S: H - 0.5', 'DFA: H - 0.5'].forEach((n, j) => tr.push(leg(n, null, { color: `rgba(${RGB[MCC[j]]},0.55)`, symbol: 'square', size: 12 })));
  tr.push(leg('true d', { color: K.navy, dash: 'dash', width: 1 }));
  plot('mc-chart', tr, G.lay);
  const st = D.mc.designs.map(s => Object.fromEntries(MCN.map(n => [n, mcStat(s.v[n].slice(0, R), s.d)])));
  out('mc-out', [T.mc1(R, f2(st[1].Whittle.sd), f2(st[1].LW.sd), f2(st[1].GPH.sd), f2(st[0]['R/S'].mean)), T.mc2(f2(st[2].Whittle.mean), f2(st[2].GPH.mean), f2(st[2].LW.mean))]);
}

// ================================================================= 12. Romanian inflation: fitted autocorrelations
const FIT = NB.fit.table, FD = FIT.find(r => r.name === 'ARFIMA(0,d,0)'), FA = FIT.find(r => r.name === 'ARMA(1,1)');
function blkAf() {
  const dd = +$('af-d').value, Lm = +$('af-L').value, d = FD.d + dd; set('af-d', (dd >= 0 ? '+' : '') + f2(dd) + ` (d = ${f3(d)})`); set('af-L', Lm);
  const x = RO.im, r = acfv(x, Lm), k = range(1, Lm, 1), gd = acvf0(d, Lm + 1), ga = acvf(0, FA.phi, FA.theta, Lm + 1), band_ = 1.96 / Math.sqrt(x.length);
  const G = grid('af-chart', [{ r: 0, c: 0, xt: 'lag k (months)', yt: 'autocorrelation' }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0];
  G.lay.shapes.push(hband(a, -band_, band_, K.teal, 0.15), hl(a, 0, K.navy, 'solid', 0.5));
  plot('af-chart', [bar(k, r, a, `rgba(${RGB[K.red]},0.55)`, 'sample ACF, RO monthly inflation (SA)'),
    ln(k, Array.from(gd.slice(1), v => v / gd[0]), a, K.blue, `ARFIMA(0,d,0), d = ${d.toFixed(2)}`, { mode: 'lines+markers', marker: { size: 4, color: K.blue } }),
    ln(k, Array.from(ga.slice(1), v => v / ga[0]), a, K.green, `ARMA(1,1), phi = ${FA.phi[0].toFixed(2)}, theta = ${FA.theta[0].toFixed(2)}`, { mode: 'lines+markers', marker: { size: 4, symbol: 'square', color: K.green }, line: { color: K.green, dash: 'dash' } }),
    legBand('95% band for white noise')], G.lay);
  const ll = memo('afll|' + d.toFixed(4), () => dlLik(x, acvf0(d, x.length)));
  out('af-out', [T.af1(f2(r[11]), f2(gd[12] / gd[0]), f2(ga[12] / ga[0]), f2(r[23]), f2(gd[24] / gd[0]), f2(ga[24] / ga[0])),
    T.af2(f3(d), nf(ll, 1), nf(FD.loglik, 1), nf(2 * (FD.loglik - ll), 1))]);
}

// ================================================================= 13. US inflation: ARFIMA and AR forecasts
const FPO = D.fcpath;
function fpState(o) {
  const M = FPO.m[o], x = US.v.slice(0, M.T), fa = arfimaFc(x, M.d, M.phi, FPO.H), d0 = US.d[M.T - 1];
  const fut = months(isoM(+d0.slice(0, 4), +d0.slice(5, 7), 1), FPO.H);
  const act = US.v.slice(M.T, M.T + FPO.H), e = (f) => Math.sqrt(mean(act.map((v, h) => (v - f[h]) ** 2)));
  return { M, x, fa, fr: M.ar, fut, d0, mu: mean(x), m12: mean(x.slice(-12)), nact: act.length, ra: act.length ? e(fa) : null, rr: act.length ? e(M.ar) : null };
}
function blkFp() {
  const o = $('fp-o').value, S = fpState(o);
  const i0 = idxFrom(US.d, '2019-01-01');
  const G = grid('fp-chart', [{ r: 0, c: 0, yt: '% per year', x: DATE }], { cols: 1, h: 360, legendRows: 2 });
  const a = G.ax[0], xs = [US.d[i0], S.fut[S.fut.length - 1]];
  plot('fp-chart', [ln(US.d.slice(i0), US.v.slice(i0), a, K.green, 'US inflation, monthly, annualised (%)', { line: { width: 1 } }),
    ln(S.fut, S.fa, a, K.blue, `ARFIMA(1,d,0): d = ${S.M.d.toFixed(2)}, phi = ${S.M.phi.toFixed(2)}`, { mode: 'lines+markers', marker: { size: 4, color: K.blue } }),
    ln(S.fut, S.fr, a, K.orange, `AR(${S.M.p}), p by AIC`, { mode: 'lines+markers', marker: { size: 4, symbol: 'square', color: K.orange } }),
    ln(xs, [S.mu, S.mu], a, K.navy, `sample mean since 1947 (${S.mu.toFixed(2)}%)`, { line: { color: K.navy, dash: 'dash', width: 1 } })], G.lay);
  const items = [T.fp1(T.my(S.d0), f2(S.M.d), f2(S.M.phi), S.M.p, f2(S.mu), f2(S.m12)), T.fp2(f2(S.fa[0]), f2(S.fr[0]), f2(S.fa[11]), f2(S.fr[11]), f2(S.fa[35]), f2(S.fr[35]))];
  if (S.nact) items.push(T.fp3(S.nact, f2(S.ra), f2(S.rr)));
  out('fp-out', items);
}

// ================================================================= 14. does long memory improve forecasts?
const FCM = ['ARFIMA', 'AR', 'Mean', 'RW'], FCL = { ARFIMA: 'ARFIMA(1,d,0)', AR: 'AR(p) with p by AIC', Mean: 'sample mean', RW: 'random walk' }, FCC = { ARFIMA: K.blue, AR: K.navy, Mean: K.green, RW: K.red };
function fcState(k) {
  return memo('fc|' + k, () => ['ro', 'us'].map(c => { const F = D.fc[c], i0 = Math.floor(F.n * k / 100), o0 = isoM(+F.o0.slice(0, 4), +F.o0.slice(5, 7), i0);
    const rm = Object.fromEntries(FCM.map(m => [m, range(0, 23, 1).map(h => Math.sqrt(mean(F.e[m].slice(i0).map(e => e[h] ** 2))))]));
    return { c, n: F.n - i0, o0, rm }; }));
}
function blkFo() {
  const k = +$('fo-k').value, b = $('fo-b').value; set('fo-k', k + '%');
  const S = fcState(k), h = range(1, 24, 1), mk_ = { ARFIMA: 'circle', AR: 'diamond', Mean: 'square', RW: 'triangle-up' };
  const G = grid('fo-chart', S.map((s, i) => ({ r: 0, c: i, t: `${s.c === 'ro' ? 'RO inflation (monthly, SA)' : 'US inflation (monthly, annualised)'}: ${s.n} forecast origins`, xt: 'horizon h (months)', yt: i ? '' : `RMSE relative to ${FCL[b]}` })), { cols: 2, h: 380, legendRows: 2 });
  const tr = [];
  S.forEach((s, i) => { FCM.filter(m => m !== b).forEach(m => tr.push(ln(h, s.rm[m].map((v, j) => v / s.rm[b][j]), G.ax[i], FCC[m], i ? '' : FCL[m], { mode: 'lines+markers', marker: { size: 5, symbol: mk_[m], color: FCC[m] } })));
    tr.push(ln([1, 24], [1, 1], G.ax[i], K.navy, i ? '' : `${FCL[b]}: the benchmark (= 1)`, { line: { color: K.navy, dash: 'dash', width: 1 } })); });
  plot('fo-chart', tr, G.lay);
  const rat = s => s.rm.ARFIMA.map((v, j) => v / s.rm.AR[j]), ro = rat(S[0]), us = rat(S[1]);
  out('fo-out', [T.fo1(S[0].n, T.my(S[0].o0), f3(ro[0]), f3(ro[11]), f3(ro[23]), nf(100 * (1 - Math.max(...ro)), 1), nf(100 * (1 - Math.min(...ro)), 1)),
    T.fo2(S[1].n, T.my(S[1].o0), f3(Math.min(...us)), f3(Math.max(...us)))]);
}

// ================================================================= 15. returns, absolute and squared returns
function volState(key, p, shuf) {
  return memo(`vol|${key}|${p}|${shuf}`, () => {
    const x0 = key === 'sp500' ? SP.r : BET.r, x = shuf ? D.data.perm[key].map(i => x0[i]) : x0, ap = x.map(v => Math.abs(v) ** p), band_ = 1.96 / Math.sqrt(x.length);
    const r = acfv(x, 250), a = acfv(ap, 250), q = acfv(x.map(v => v * v), 250);
    return { n: x.length, band: band_, r, a, q, npos: a.filter(v => v > band_).length, lwd: lw(pgram(ap), bwm(ap.length, 0.65)).d };
  });
}
function blkVa() {
  const p = +$('va-p').value, sh = $('va-s').value === '1'; set('va-p', f2(p));
  const S = ['sp500', 'bet'].map(k => [k, volState(k, p, sh)]), lags = range(1, 250, 1);
  const G = grid('va-chart', S.map(([k, s], i) => ({ r: 0, c: i, t: `${k === 'sp500' ? 'S&P 500' : 'BET'}, daily, 2000-2026` + (sh ? ', days shuffled' : ''), xt: 'lag k (days)', yt: i ? '' : 'autocorrelation' })), { cols: 2, h: 360, legendRows: 2 });
  const tr = [], pl = p === 1 ? '|r_t|' : `|r_t|^${p}`;
  S.forEach(([k, s], i) => { const a = G.ax[i];
    tr.push(ln(lags, s.r, a, COL[k], `${k === 'sp500' ? 'S&P 500' : 'BET'} returns r_t`, { line: { width: 1.1 } }), ln(lags, s.a, a, K.purple, i ? '' : pl, { line: { width: 1.1 } }),
      ln(lags, s.q, a, K.orange, i ? '' : 'r_t^2', { line: { width: 1.1 } }));
    G.lay.shapes.push(hband(a, -s.band, s.band, K.teal, 0.15), hl(a, 0, K.navy, 'solid', 0.5)); });
  tr.push(legBand('95% band, i.i.d.'));
  plot('va-chart', tr, G.lay);
  const ps = p === 1 ? '' : f2(p);
  out('va-out', [T.va1(ps, sh, S.map(([k, s]) => [k, f2(s.a[0]), f2(s.a[99]), f2(s.a[249]), s.npos])), T.va2(ps, sh, f2(S[0][1].lwd), f2(S[1][1].lwd))]);
}

// ================================================================= 16. monthly realised volatility
const RVN = NB.rv;
function rvState(key, dd) {
  return memo(`rv|${key}|${dd}`, () => { const x = RV[key], R = RVN[key], d = R.ml + dd, g = acvf(d, [R.ml_phi], [], x.length);
    const g0 = memo('rv0|' + key, () => dlLik(x, acvf(R.ml, [R.ml_phi], [], x.length)));
    return { d, g, ll: dlLik(x, g), ll0: g0, r: acfv(x, 36), ga: acvf(0, [R.a_phi], [R.a_theta], 37) }; });
}
function blkRv() {
  const dd = +$('rv-d').value; set('rv-d', (dd >= 0 ? '+' : '') + f2(dd));
  const S = ['sp500', 'bet'].map(k => [k, rvState(k, dd)]), k_ = range(1, 36, 1);
  const G = grid('rv-chart', S.map(([k, s], i) => ({ r: 0, c: i, t: `${k === 'sp500' ? 'S&P 500' : 'BET'}: d = ${s.d.toFixed(2)} (ML ${RVN[k].ml.toFixed(2)}${dd ? ' ' + (dd > 0 ? '+' : '-') + ' ' + Math.abs(dd).toFixed(2) : ''}), ${RVN[k].lw.toFixed(2)} (LW), ${RVN[k].gph.toFixed(2)} (GPH)`, xt: 'lag k (months)', yt: i ? '' : 'autocorrelation' })), { cols: 2, h: 360, legendRows: 2 });
  const tr = [];
  S.forEach(([k, s], i) => { const a = G.ax[i];
    tr.push(bar(k_, s.r, a, `rgba(${RGB[COL[k]]},0.55)`, `${k === 'sp500' ? 'S&P 500' : 'BET'}: sample ACF of log RV`),
      ln(k_, Array.from(s.g.slice(1, 37), v => v / s.g[0]), a, K.navy, i ? '' : 'ARFIMA(1,d,0), exact ML', { mode: 'lines+markers', marker: { size: 4, color: K.navy } }),
      ln(k_, Array.from(s.ga.slice(1), v => v / s.ga[0]), a, K.orange, i ? '' : 'ARMA(1,1), exact ML', { line: { color: K.orange, dash: 'dash' } }));
    G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.5)); });
  plot('rv-chart', tr, G.lay);
  out('rv-out', S.map(([k, s]) => T.rv1(k, f2(s.d), f2(RVN[k].ml_phi), f2(s.r[23]), f2(s.g[24] / s.g[0]), nf(s.ll - s.ll0, 1))));
}

// ================================================================= 17. FIGARCH against GARCH; HAR against (1 - L)^d
const KV = [1, 2, 5, 6, 10, 20, 22, 50, 100, 250, 500];
const harFit = end => memo('har|' + end, () => {
  const i1 = end >= 2026 ? PK.v.length - 1 : idxTo(PK.d, `${end}-12-31`), y = PK.v.slice(0, i1 + 1), n = y.length - 22, cs = [0, ...cumsum(y)];
  const row = t => { const s = t + 22; return [1, y[s - 1], (cs[s] - cs[s - 5]) / 5, (cs[s] - cs[s - 22]) / 22]; };
  const f = ols(y.slice(22), row, 4), [c, bd, bw, bm] = f.b, fit = range(0, n - 1, 1).map(t => row(t).reduce((s, v, j) => s + v * f.b[j], 0));
  const w = [bd + bw / 5 + bm / 22, ...new Array(4).fill(bw / 5 + bm / 22), ...new Array(17).fill(bm / 22)];
  return { c, bd, bw, bm, n, w, y: y.slice(22), fit, d: PK.d.slice(22, i1 + 1) };
});
const parkD = () => memo('parkd', () => lw(pgram(PK.v), bwm(PK.v.length, 0.65)).d);
const figW = k => memo('figw|' + k, () => figarchW(VOL[k].figarch.d, VOL[k].figarch.phi, VOL[k].figarch.beta, 1000));
function blkVm() {
  const kk = KV[+$('vm-k').value], dd = +$('vm-d').value; set('vm-k', kk); set('vm-d', (dd >= 0 ? '+' : '') + f2(dd));
  const H = harFit(2026), dl = parkD() + dd, pw = Array.from(fw(dl, 23).slice(1), v => -v), lag = range(1, 999, 1), k22 = range(1, 22, 1);
  const G = grid('vm-chart', [{ r: 0, c: 0, t: 'ARCH(infinity) weights', xt: 'lag k (days, log scale)', yt: 'weight of r^2_(t-k) in sigma^2_t', x: LOG, y: { type: 'log', exponentformat: 'power', range: [-8, Math.log10(0.5)] } },
    { r: 0, c: 1, t: 'HAR: three steps that mimic a power law', xt: 'lag k (days)', yt: 'weight of lag k' }], { cols: 2, h: 380, legendRows: 4 });
  const [a, b] = G.ax, tr = [];
  ['sp500', 'bet'].forEach(k => { const g = VOL[k].garch, f = figW(k), nm = k === 'sp500' ? 'S&P 500' : 'BET';
    tr.push(ln(lag, lag.map(j => g.alpha * g.beta ** (j - 1)), a, COL[k], `${nm}: GARCH, alpha + beta = ${(g.alpha + g.beta).toFixed(3)}`, { line: { color: COL[k], dash: 'dash', width: 1.3 } }),
      ln(lag, f.map(v => Math.max(v, 1e-12)), a, COL[k], `${nm}: FIGARCH, d = ${VOL[k].figarch.d.toFixed(2)}`, { line: { width: 1.6 } }),
      mk([kk, kk], [g.alpha * g.beta ** (kk - 1), f[kk - 1]], a, COL[k], '', { marker: { color: COL[k], size: 8, symbol: 'circle-open', line: { width: 2 } } })); });
  tr.push(bar(k22, H.w, b, `rgba(${RGB[K.teal]},0.6)`, `HAR (S&P 500): beta_d, beta_w, beta_m = ${H.bd.toFixed(2)}, ${H.bw.toFixed(2)}, ${H.bm.toFixed(2)}`),
    ln(k22, pw, b, K.purple, `-pi_k of (1-L)^d, d = ${dl.toFixed(2)}`, { mode: 'lines+markers', marker: { size: 5, color: K.purple } }));
  if (kk <= 22) tr.push(mk([kk, kk], [H.w[kk - 1], pw[kk - 1]], b, K.navy, '', { marker: { color: K.navy, size: 9, symbol: 'circle-open', line: { width: 2 } } }));
  plot('vm-chart', tr, G.lay);
  const ws = ['sp500', 'bet'].map(k => { const g = VOL[k].garch; return [sci(g.alpha * g.beta ** (kk - 1)), sci(figW(k)[kk - 1])]; });
  const fwk = -fw(dl, kk + 1)[kk];
  out('vm-out', [T.vm1(kk, ws), T.vm2(kk, kk <= 22 ? sci(H.w[kk - 1]) : '0', sci(fwk), f2(dl))]);
}

// ================================================================= 18. short memory that looks long
const SPU = D.spurious;
const at = (M, v) => M[Object.keys(M).find(k => +k === v)];
const qv = (v, p) => { const b = [...v].sort((x, y) => x - y), h = (b.length - 1) * p, k = Math.floor(h); return b[k] + (h - k) * ((b[k + 1] ?? b[k]) - b[k]); };
function blkSp() {
  const R = +$('sp-r').value, bnd = $('sp-b').value === '1'; set('sp-r', R);
  const G = grid('sp-chart', [{ r: 0, c: 0, t: 'One break', xt: 'size of one mean shift at T/2 (in standard deviations)', yt: 'estimated d' },
    { r: 0, c: 1, t: 'Markov switching between means 0 and 1', xt: 'switching probability p (log scale)', x: LOG }], { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax, tr = [];
  [[a, SPU.shifts, SPU.br], [b, SPU.switch, SPU.ms]].forEach(([ax, xs, M], i) => {
    const mg = xs.map(v => mean(at(M, v).gph.slice(0, R))), ml = xs.map(v => mean(at(M, v).lw.slice(0, R)));
    if (bnd) tr.push(band(xs, xs.map(v => qv(at(M, v).lw.slice(0, R), 0.05)), xs.map(v => qv(at(M, v).lw.slice(0, R), 0.95)), ax, K.blue, 0.15, i ? '' : 'local Whittle: 5%-95% of the samples'));
    tr.push(ln(xs, mg, ax, K.red, i ? '' : 'GPH: mean estimate', { mode: 'lines+markers', marker: { size: 5, symbol: 'square', color: K.red }, line: { color: K.red, dash: 'dash', width: 1.3 } }),
      ln(xs, ml, ax, K.blue, i ? '' : 'local Whittle: mean estimate', { mode: 'lines+markers', marker: { size: 5, color: K.blue } }));
    G.lay.shapes.push(hl(ax, 0, K.navy, 'dash', 1)); });
  tr.push(leg('true d = 0', { color: K.navy, dash: 'dash', width: 1 }));
  plot('sp-chart', tr, G.lay);
  const m = (M, v, e) => mean(at(M, v)[e].slice(0, R));
  const items = [T.sp1(R, f2(m(SPU.br, 0.5, 'lw')), f2(m(SPU.br, 1, 'lw')), f2(m(SPU.br, 0.5, 'gph'))),
    T.sp2(f2(m(SPU.ms, 0.01, 'lw')), nf(mean(at(SPU.ms, 0.01).sw.slice(0, R)), 0), f2(m(SPU.ms, 0.2, 'lw')), nf(mean(at(SPU.ms, 0.2).sw.slice(0, R)), 0))];
  if (bnd) items.push(T.sp3(f2(qv(at(SPU.br, 1).lw.slice(0, R), 0.05)), f2(qv(at(SPU.br, 0).lw.slice(0, R), 0.95))));
  out('sp-out', items);
}

// ================================================================= 19. the Nile in 1898
const NSZ = bsizes(100, 6, 50, 10);
function nileState(B) {
  return memo('nile|' + B, () => {
    const i = NILE.y.indexOf(B) + 1, pre = NILE.v.slice(0, i), post = NILE.v.slice(i), mp = mean(pre), mq = mean(post);
    const adj = [...pre.map(v => v - mp), ...post.map(v => v - mq)];
    const est = z => { const rs = rsCurve(z, NSZ), P = pgram(z), m = bwm(100, 0.65); return { rs, H: logSlope(NSZ, rs), lw: lw(P, m).d, ml: mlD(z).d, r1: acfv(z, 1)[0] }; };
    return { pre: mp, post: mq, np: pre.length, nq: post.length, raw: est(NILE.v), adj: est(adj) };
  });
}
function blkNb() {
  const B = +$('nb-y').value; set('nb-y', B);
  const S = nileState(B);
  const G = grid('nb-chart', [{ r: 0, c: 0, cs: 3, xt: 'year' }, { r: 0, c: 3, cs: 2, xt: 'block size n', yt: '(R/S)_n', x: LOG, y: LOG }], { cols: 5, h: 360, gx: 0.06, legendRows: 3 });
  const [a, b] = G.ax;
  plot('nb-chart', [ln(NILE.y, NILE.v, a, K.blue, 'annual flow at Aswan (10^8 m^3)', { line: { width: 1.2 } }),
    ln([1871, B], [S.pre, S.pre], a, K.red, `mean 1871-${B}: ${S.pre.toFixed(0)}`, { line: { color: K.red, width: 2 } }),
    ln([B + 1, 1970], [S.post, S.post], a, K.green, `mean ${B + 1}-1970: ${S.post.toFixed(0)}`, { line: { color: K.green, width: 2 } }),
    ln(NSZ, S.raw.rs, b, K.blue, `raw series: H = ${S.raw.H.toFixed(2)}, d (ML) = ${S.raw.ml.toFixed(2)}`, { mode: 'lines+markers', marker: { size: 5, color: K.blue } }),
    ln(NSZ, S.adj.rs, b, K.orange, `regime means removed: H = ${S.adj.H.toFixed(2)}, d (ML) = ${S.adj.ml.toFixed(2)}`, { mode: 'lines+markers', marker: { size: 5, symbol: 'square', color: K.orange } }),
    ln(NSZ, NSZ.map(n => (n / NSZ[0]) ** 0.5 * S.adj.rs[0]), b, K.navy, 'slope 0.5', { line: { color: K.navy, dash: 'dash', width: 0.9 } })], G.lay);
  out('nb-out', [T.nb1(B, nf(S.pre, 0), nf(S.post, 0), nf(100 * (1 - S.post / S.pre), 0), S.np, S.nq),
    T.nb2(f2(S.raw.ml), f2(S.raw.lw), f2(S.raw.H), f2(S.raw.r1), f2(S.adj.ml), f2(S.adj.lw), f2(S.adj.H), f2(S.adj.r1))]);
}

// ================================================================= 20. the stability of memory over time
function rollLW(x, d, W, step) { const o = { d: [], v: [] }; for (let e = W; e <= x.length; e += step) { const z = x.slice(e - W, e); o.d.push(d[e - 1]); o.v.push(lw(pgram(z), bwm(W, 0.65)).d); } return o; }
function rollDFA(x, d, W, step) { const s = bsizes(W), o = { d: [], v: [] }; for (let e = W; e <= x.length; e += step) { o.d.push(d[e - 1]); o.v.push(hDFA(x.slice(e - W, e), s)); } return o; }
function rlState(m) {
  return memo('rl|' + m, () => { const B = D.bands[Object.keys(D.bands).find(k => +k === m)];
    return { B, us: rollLW(US.v, US.d, B.wu, 6), sp: rollDFA(SP.r, SP.d, B.wd, 21), bet: rollDFA(BET.r, BET.d, B.wd, 21) }; });
}
function blkRl() {
  const m = +$('rl-w').value; set('rl-w', `${nf(m, 2)} (${nf(20 * m, 0)} ${T.yrs}, ${nf(1000 * m, 0)} ${T.dys})`);
  const S = rlState(m);
  const G = grid('rl-chart', [{ r: 0, c: 0, yt: 'd', x: DATE }, { r: 1, c: 0, yt: 'H', x: DATE }], { cols: 1, h: 520, legendRows: 3 });
  const [a, b] = G.ax;
  G.lay.shapes.push(hband(a, S.B.bu[0], S.B.bu[1], K.teal, 0.18), hl(a, 0, K.navy, 'solid', 0.5), hband(b, S.B.bh[0], S.B.bh[1], K.teal, 0.18), hl(b, 0.5, K.navy, 'dash', 0.6));
  plot('rl-chart', [ln(S.us.d, S.us.v, a, K.green, `US monthly inflation: d (local Whittle), ${nf(20 * m, 0)}-year windows`),
    ln(S.sp.d, S.sp.v, b, K.blue, `S&P 500 returns: DFA exponent H, ${S.B.wd}-day windows`, { line: { width: 1.2 } }),
    ln(S.bet.d, S.bet.v, b, K.red, `BET returns: DFA exponent H, ${S.B.wd}-day windows`, { line: { width: 1.2 } }), legBand('95% band of i.i.d. series (Monte Carlo)')], G.lay);
  const iM = S.us.v.indexOf(Math.max(...S.us.v)), im = S.us.v.indexOf(Math.min(...S.us.v)), jm = S.sp.v.indexOf(Math.min(...S.sp.v));
  const above = mean(S.bet.v.map(v => (v > S.B.bh[1] ? 1 : 0)));
  out('rl-out', [T.rl1(nf(20 * m, 0), f2(S.us.v[iM]), T.my(S.us.d[iM]), f2(S.us.v[im]), T.my(S.us.d[im]), f3(S.us.v[S.us.v.length - 1]), f3(S.B.bu[0]), f3(S.B.bu[1])),
    T.rl2(S.B.wd, f2(S.sp.v[jm]), T.my(S.sp.d[jm]), nf(100 * above, 0), f3(S.B.bh[0]), f3(S.B.bh[1]))]);
}

// ================================================================= EXTRA. the two decay laws, step by step
function blkWd() {
  const d = +$('wd-d').value, k = +$('wd-k').value; set('wd-d', f2(d));
  const S = decayState(d), kk = range(1, 1000, 1), cs = cumsum(S.r), ca = cumsum(kk.map(j => S.phi ** j));
  const G = grid('wd-chart', [{ r: 0, c: 0, t: 'Partial sums of the autocorrelations', xt: 'K (log scale)', yt: 'sum of rho(k), k = 1..K', x: LOG },
    { r: 0, c: 1, t: 'rho(k) / (C k^(2d-1))', xt: 'lag k (log scale)', x: LOG }], { cols: 2, h: 340, legendRows: 2 });
  const [a, b] = G.ax;
  plot('wd-chart', [ln(kk, cs, a, K.blue, `ARFIMA(0,d,0), d = ${d.toFixed(2)}`), ln(kk, ca, a, K.orange, `AR(1), phi = ${S.phi.toFixed(3)}`),
    ln(kk, S.r.map((v, j) => v / (S.C * (j + 1) ** (2 * d - 1))), b, K.blue, ''), ln([1, 1000], [1, 1], b, K.navy, 'ratio 1', { line: { color: K.navy, dash: 'dash', width: 1 } })], G.lay);
  const rk = Math.exp(lgamma(k + d) + lgamma(1 - d) - lgamma(k - d + 1) - lgamma(d)) * Math.sign(gammaf(d));
  steps('wd-ws', [T.wds1(f2(d), f3(S.phi)), T.wds2(k, f2(d), f3(rk), sci(S.phi ** k)), T.wds3(f3(S.C), k, f2(2 * d - 1), f3(S.C * k ** (2 * d - 1)), f3(rk / (S.C * k ** (2 * d - 1)))),
    T.wds4(nf(S.s100, 1), nf(S.s1000, 1), f3(S.phi), nf(S.asum, 1))]);
  out('wd-out', [T.wd1(f2(d), nf(S.s1000 / S.asum, 0))]);
}

// ================================================================= EXTRA. the weights of (1 - L)^0.4
function blkWw() {
  const d = +$('ww-d').value; set('ww-d', nf(d, 1));
  const w = fw(d, 1001), k10 = range(1, 10, 1), kk = range(1, 1000, 1), ps = cumsum(Array.from(w)).slice(1);
  const G = grid('ww-chart', [{ r: 0, c: 0, t: `weights pi_k of (1-L)^${d.toFixed(1)}`, xt: 'lag k' }, { r: 0, c: 1, t: '1 + pi_1 + ... + pi_K', xt: 'K (log scale)', x: LOG }], { cols: 2, h: 320, legendRows: 1 });
  const [a, b] = G.ax;
  G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.5), hl(b, 0, K.navy, 'dash', 0.8));
  plot('ww-chart', [bar(k10, Array.from(w.slice(1, 11)), a, K.blue, `pi_k, d = ${d.toFixed(1)}`), ln(kk, ps, b, K.purple, 'partial sum of the weights')], G.lay);
  const r = k => `π<sub>${k}</sub> = ${nf(w[k - 1], 4)} · (${k - 1} − ${nf(d, 1)})/${k} = <b>${nf(w[k], 4)}</b>`;
  steps('ww-ws', [r(1), r(2), r(3), r(4) + '; ' + r(5), T.wws1(f2(-w[1]), f3(-w[2]), f3(-w[3]), nf(-w[4], 4)), T.wws2(f3(ps[9]), f3(ps[99]), f3(ps[999]))]);
  out('ww-out', [T.ww1(nf(d, 1), nf(w[1], 2), nf(w[2], 3), nf(-d / gammaf(1 - d) * 5 ** (-1 - d), 4), nf(w[5], 4))]);
}

// ================================================================= EXTRA. the parameter d: what each range means
const DRNG = d => (d <= -0.5 + 1e-9 ? 0 : d < -1e-9 ? 1 : Math.abs(d) < 1e-9 ? 2 : d < 0.5 - 1e-9 ? 3 : d < 1 - 1e-9 ? 4 : Math.abs(d - 1) < 1e-9 ? 5 : 6);
function blkDr() {
  const d = +(+$('dr-d').value).toFixed(2); set('dr-d', nf(d, 1));
  const ds = d >= 0.5 - 1e-9 ? +(d - 1).toFixed(2) : d, kk = range(1, 500, 1), psi = fw(-d, 501), j = DRNG(d), k50 = range(1, 50, 1);
  const G = grid('dr-chart', [{ r: 0, c: 0, t: `impulse responses psi_k of (1-L)^-d, d = ${d.toFixed(1)}`, xt: 'horizon k (log scale)', x: LOG },
    { r: 0, c: 1, t: d >= 0.5 - 1e-9 ? `ACF of (1-L)x_t, ARFIMA(0, ${ds.toFixed(1)}, 0)` : `ACF of ARFIMA(0, ${ds.toFixed(1)}, 0)`, xt: 'lag k' }], { cols: 2, h: 320, legendRows: 1 });
  const [a, b] = G.ax, r = ds > -1 && ds < 0.5 ? acf0(ds, 51).slice(1) : k50.map(() => 0);
  G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.5), hl(b, 0, K.navy, 'solid', 0.5));
  plot('dr-chart', [ln(kk, Array.from(psi.slice(1)), a, K.blue, 'psi_k'), bar(k50, r, b, `rgba(${RGB[K.purple]},0.6)`, 'rho(k)')], G.lay);
  table('dr-tb', [L.drR, L.drP, L.drA], T.drRows.map((row, i) => { const rr = row.map(c => (i === j ? `<b>${c}</b>` : c)); rr.hi = i === j; return rr; }), { cap: T.drCap(nf(d, 1)) });
  out('dr-out', [T.dr1(nf(d, 1), j, nf(d + 0.5, 1), f3(psi[100]), ds > -1 && ds < 0.5 ? f3(r[0]) : null, nf(ds, 1), d >= 0.5 - 1e-9)]);
}

// ================================================================= EXTRA. ARFIMA(0, 0.3, 0)
function blkWa() {
  const d = +$('wa-d').value; set('wa-d', f2(d));
  const r = acf0(d, 51), k = range(1, 50, 1), ph = r[1];
  const G = grid('wa-chart', [{ r: 0, c: 0, xt: 'lag k', yt: 'autocorrelation (log scale)', y: { type: 'log', exponentformat: 'power', range: [-4, 0] } }], { cols: 1, h: 320, legendRows: 1 });
  const a = G.ax[0], ks = [1, 2, 3, 5, 10];
  plot('wa-chart', [ln(k, r.slice(1), a, K.blue, `ARFIMA(0,d,0), d = ${d.toFixed(2)}`), ln(k, k.map(j => ph ** j).map(v => (v < 1e-5 ? null : v)), a, K.orange, `AR(1) with phi = rho(1) = ${ph.toFixed(3)}`),
    mk(ks, ks.map(j => r[j]), a, K.blue, '', { marker: { color: K.blue, size: 8 } })], G.lay);
  const g10 = Math.exp(lgamma(10 + d) + lgamma(1 - d) - lgamma(11 - d) - lgamma(d)), v = acvf0(d, 1)[0];
  steps('wa-ws', [T.was1(f2(d), f3(r[1]), f3(r[2]), f3(r[3]), f3(r[5]), f3(r[10])), T.was2(f2(d), f3(g10)), T.was3(nf(1 - 2 * d, 1), nf(1 - d, 1), f3(v)),
    T.was4(f3(ph), sci(ph ** 10), f3(r[10]), nf(r[10] / ph ** 10, 0)), T.was5(f2(d + 0.5))]);
  out('wa-out', [T.wa1(f2(d), f3(r[10]), sci(ph ** 10), f3(v))]);
}

// ================================================================= EXTRA. the R/S statistic, step by step
const RSX = [0.5, -0.3, 0.8, -0.2, 0.6, -0.1, 0.4, -0.7];
function blkWr() {
  const x8 = +$('wr-x').value; set('wr-x', nf(x8, 1));
  const x = [...RSX.slice(0, 7), x8], mu = mean(x), Y = cumsum(x.map(v => v - mu)), mx = Math.max(...Y), mn = Math.min(...Y), S = sdp(x), k = range(1, 8, 1);
  const G = grid('wr-chart', [{ r: 0, c: 0, xt: 'k' }], { cols: 1, h: 300, legendRows: 1 });
  const a = G.ax[0];
  G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.5));
  plot('wr-chart', [bar(k, x, a, `rgba(${RGB[K.blue]},0.55)`, 'x_k'), ln(k, Y, a, K.red, 'cumulative deviations Y_k', { mode: 'lines+markers', marker: { size: 6, color: K.red } }),
    mk([Y.indexOf(mx) + 1, Y.indexOf(mn) + 1], [mx, mn], a, K.navy, 'max and min of Y_k', { marker: { color: K.navy, size: 11, symbol: 'circle-open', line: { width: 2 } } })], G.lay);
  steps('wr-ws', [T.wrs1(nf(sum(x), 1), nf(mu, 3)), T.wrs2(Y.map(v => nf(v, 3)).join('; ')), T.wrs3(nf(mx, 3), nf(mn, 3), nf(mx - mn, 3)), T.wrs4(nf(S, 3), nf((mx - mn) / S, 2))]);
  out('wr-out', [T.wr1(nf(x8, 1), nf(mx - mn, 3), nf(S, 3), nf((mx - mn) / S, 2))]);
}

// ================================================================= EXTRA. the periodogram and the GPH estimator
function blkWg() {
  const key = $('wg-s').value, a_ = +$('wg-a').value; set('wg-a', f2(a_));
  const [, nm, x] = GPS().find(s => s[0] === key), P = PG(key, x), m = bwm(x.length, a_), g = gph(P, m), J = P.lam.length;
  const G = grid('wg-chart', [{ r: 0, c: 0, t: `${nm}: periodogram, T = ${x.length}, m = ${m}`, xt: 'lambda_j (log scale)', yt: 'log I(lambda_j)', x: LOG }], { cols: 1, h: 340, legendRows: 1 });
  const ax = G.ax[0], lam = Array.from(P.lam), lI = Array.from(P.I, Math.log);
  plot('wg-chart', [mk(lam.slice(m), lI.slice(m), ax, `rgba(${RGB[COL[key]]},0.3)`, 'j > m', { marker: { color: `rgba(${RGB[COL[key]]},0.3)`, size: 5 } }),
    mk(lam.slice(0, m), lI.slice(0, m), ax, COL[key], 'j = 1, ..., m'), ln(lam.slice(0, m), g.X.map(v => g.c + g.d * v), ax, K.navy, 'GPH fit: c + d [-log(4 sin^2(lambda_j/2))]', { line: { color: K.navy, width: 1.4 } })], G.lay);
  const mX = mean(g.X), mY = mean(g.Y); let sxy = 0, sxx = 0; g.X.forEach((v, j) => { sxy += (v - mX) * (g.Y[j] - mY); sxx += (v - mX) ** 2; });
  steps('wg-ws', [T.wgs1(x.length, f2(a_), m, nf(P.lam[0], 4), nf(P.lam[m - 1], 3), nf(2 * Math.PI / P.lam[m - 1], 1), J), T.wgs2(nf(g.X[0], 3), nf(g.Y[0], 3), nf(mX, 3), nf(mY, 3)),
    T.wgs3(nf(sxy, 3), nf(sxx, 3), f3(g.d)), T.wgs4(m, f3(g.se), civ(g.d, g.se))]);
  out('wg-out', [T.wg1(key, m, f2(g.d), f2(g.se))]);
}

// ================================================================= EXTRA. GPH and local Whittle with intervals
function blkGt() {
  const a = +$('gt-a').value; set('gt-a', f2(a));
  const S = gphState(a), labs = [], vals = [], err = [], cols = [];
  S.forEach(s => { [['GPH', s.g, K.red], ['LW', s.w, K.blue]].forEach(([e, r, c]) => { labs.push(`${s.key === 'nile' ? 'Nile' : 'RO inflation'}, ${e}`); vals.push(r.d); err.push(1.96 * r.se); cols.push(c); }); });
  const G = grid('gt-chart', [{ r: 0, c: 0, xt: 'd: estimate and 95% interval' }], { cols: 1, h: 300, legendRows: 1, ml: 130 });
  const ax = G.ax[0];
  G.lay.shapes.push(vl(ax, 0, K.navy, 'solid', 0.8), vl(ax, 0.5, K.purple, 'dot', 1.2));
  plot('gt-chart', [{ x: vals, y: labs, type: 'scatter', mode: 'markers', marker: { size: 9, color: cols }, error_x: { type: 'data', array: err, color: K.navy, thickness: 1.3, width: 5 }, xaxis: ax.xa, yaxis: ax.ya, showlegend: false, hovertemplate: '%{y}: %{x:.3f}<extra></extra>' },
    leg('d = 0.5', { color: K.purple, dash: 'dot', width: 1.2 })], G.lay);
  table('gt-tb', [L.series, H_('T'), H_('m'), H_(L.cyc), H_('d<sub>GPH</sub> (SE)'), H_(L.ci95), H_('d<sub>LW</sub> (SE)'), H_(L.ci95), H_(L.has05)],
    S.map(s => { const c1 = ci(s.g.d, s.g.se), c2 = ci(s.w.d, s.w.se), in5 = c => c[0] <= 0.5 && c[1] >= 0.5;
      return [T.gpName[s.key], txt(String(s.n)), txt(String(s.m)), txt(nf(s.per, 1) + ' ' + (s.key === 'nile' ? T.yrs : T.mos)), txt(`${f2(s.g.d)} (${f2(s.g.se)})`), txt(civ(s.g.d, s.g.se)),
        txt(`${f2(s.w.d)} (${f2(s.w.se)})`), txt(civ(s.w.d, s.w.se)), txt(`${T.yes(in5(c1))} / ${T.yes(in5(c2))}`, in5(c1) || in5(c2))]; }), { cap: T.gtCap(f2(a)) });
  out('gt-out', [T.gt1(S.map(s => [s.key, civ(s.g.d, s.g.se), civ(s.w.d, s.w.se), ci(s.w.d, s.w.se)[0] > 0]))]);
}

// ================================================================= EXTRA. the local Whittle estimator
const LWS = () => [['nile', 'Nile flow', NILE.v], ['infl', 'RO inflation (monthly, SA)', RO.im], ['usinfl', 'US inflation (monthly)', US.v], ['unrate', 'US unemployment', US.u]];
function blkWl() {
  const key = $('wl-s').value, a = +$('wl-a').value; set('wl-a', f2(a));
  const [, nm, x] = LWS().find(s => s[0] === key), P = PG(key, x), m = bwm(x.length, a), w = lw(P, m), ds = range(-0.4, 1.4, 0.02), Rv = ds.map(w.R);
  const G = grid('wl-chart', [{ r: 0, c: 0, t: `${nm}: R(d), m = ${m}`, xt: 'd', yt: 'R(d)' }], { cols: 1, h: 320, legendRows: 1 });
  const ax = G.ax[0];
  G.lay.shapes.push(vl(ax, 0.75, K.purple, 'dot', 1.2));
  plot('wl-chart', [ln(ds, Rv, ax, K.blue, 'R(d) = log G(d) - 2d mean(log lambda_j)'), mk([w.d], [w.R(w.d)], ax, K.red, `minimum: d = ${w.d.toFixed(3)}`, { marker: { color: K.red, size: 10 } }),
    leg('d = 3/4: the limit of the normal approximation', { color: K.purple, dash: 'dot', width: 1.2 })], G.lay);
  let G0 = 0; for (let j = 0; j < m; j++) G0 += Math.exp(2 * w.d * Math.log(P.lam[j])) * P.I[j]; G0 /= m;
  const valid = w.d > -0.5 && w.d < 0.75;
  steps('wl-ws', [T.wls1(x.length, f2(a), m, nf(P.lam[m - 1], 3)), T.wls2(f3(w.d), sci(G0), nf(w.R(w.d), 4)), T.wls3(m, f3(w.se), f3(Math.PI / Math.sqrt(24 * m)), nf(Math.PI / Math.sqrt(6), 3)),
    T.wls4(civ(w.d, w.se), valid)]);
  out('wl-out', [T.wl1(key, f2(w.d), f2(w.se), civ(w.d, w.se), valid)]);
}

// ================================================================= EXTRA. Romanian inflation: ARMA or ARFIMA?
const FTN = D.fit;
function blkFt() {
  const e = $('ft-e').value, M = FTN.m[e], rows = M.rows;
  const G = grid('ft-chart', [{ r: 0, c: 0, yt: 'information criterion (smaller is better)', y: { range: [Math.min(...rows.map(r => r.aic)) - 8, Math.max(...rows.map(r => r.bic)) + 4] } }], { cols: 1, h: 320, legendRows: 1 });
  const ax = G.ax[0], nm = rows.map(r => r.name.replace('d', 'd'));
  G.lay.barmode = 'group';
  plot('ft-chart', [bar(nm, rows.map(r => r.aic), ax, K.blue, 'AIC'), bar(nm, rows.map(r => r.bic), ax, K.red, 'BIC')], G.lay);
  const ma = Math.min(...rows.map(r => r.aic)), mb = Math.min(...rows.map(r => r.bic));
  const am = r => (r.phi.length ? r.phi.map(f2).join(', ') : '–') + ' / ' + (r.theta.length ? r.theta.map(f2).join(', ') : '–');
  table('ft-tb', [L.model, H_('d̂'), H_(L.arma), H_('ℓ'), H_('AIC'), H_('BIC')],
    rows.map(r => [r.name, num(r.d, 2), txt(am(r)), num(r.ll, 1), num(r.aic, 1, r.aic === ma), num(r.bic, 1, r.bic === mb)]), { cap: T.ftCap(T.my(e + '-01'), M.T) });
  const ba = rows.find(r => r.aic === ma), bb = rows.find(r => r.bic === mb), fd = rows.find(r => r.name === 'ARFIMA(0,d,0)'), a11 = rows[0];
  out('ft-out', [T.ft1(T.my(e + '-01'), ba.name, bb.name, f2(fd.d), nf(fd.ll, 1), nf(a11.ll, 1), f2(a11.phi[0]))]);
}

// ================================================================= EXTRA. long memory in twelve series
const MTS = () => [['Nile', NILE.v], ['RO inflation, monthly SA', RO.im], ['RO inflation, 12-month', RO.i12], ['US inflation, monthly', US.v],
  ['US inflation, 1985-2019', US.v.slice(idxFrom(US.d, '1985-01-01'), idxTo(US.d, '2019-12-01') + 1)], ['US unemployment', US.u], ['S&P 500, r_t', SP.r], ['S&P 500, |r_t|', absv(SP.r)],
  ['BET, r_t', BET.r], ['BET, |r_t|', absv(BET.r)], ['EUR/RON, r_t', EUR.r], ['EUR/RON, |r_t|', absv(EUR.r)]];
function mtState(a) {
  return memo('mt|' + a, () => MTS().map(([nm, x], i) => { const P = PG('mt' + i, x), m = bwm(x.length, a), g = gph(P, m), w = lw(P, m);
    const h = memo('mth' + i, () => [hRS(x), hDFA(x)]);
    return { nm, n: x.length, m, g, w, rs: h[0], dfa: h[1] }; }));
}
function blkMt() {
  const a = +$('mt-a').value; set('mt-a', f2(a));
  const S = mtState(a), labs = S.map(s => T.mtName[s.nm] || s.nm).reverse(), rev = [...S].reverse();
  const G = grid('mt-chart', [{ r: 0, c: 0, xt: 'd: local Whittle with 95% interval, GPH', y: { automargin: true } }], { cols: 1, h: 460, legendRows: 2, ml: 150 });
  const ax = G.ax[0];
  [0, 0.5, 0.75, 1].forEach((v, i) => G.lay.shapes.push(vl(ax, v, [K.navy, K.purple, K.orange, K.red][i], i ? 'dot' : 'solid', 1)));
  plot('mt-chart', [{ x: rev.map(s => s.w.d), y: labs, type: 'scatter', mode: 'markers', name: 'local Whittle, ±1.96 SE', marker: { size: 9, color: K.blue }, error_x: { type: 'data', array: rev.map(s => 1.96 * s.w.se), color: K.blue, thickness: 1.3, width: 4 }, xaxis: ax.xa, yaxis: ax.ya, hovertemplate: '%{y}: %{x:.3f}<extra></extra>' },
    { x: rev.map(s => s.g.d), y: labs, type: 'scatter', mode: 'markers', name: 'GPH', marker: { size: 8, color: K.red, symbol: 'square-open', line: { width: 2 } }, xaxis: ax.xa, yaxis: ax.ya, hovertemplate: '%{y}: %{x:.3f}<extra></extra>' },
    leg('d = 0.5, 0.75, 1', { color: K.purple, dash: 'dot', width: 1 })], G.lay);
  table('mt-tb', [L.series, H_('T'), H_('H<sub>R/S</sub>'), H_('H<sub>DFA</sub>'), H_('d<sub>GPH</sub>'), H_('d<sub>LW</sub> (SE)'), H_(L.ci95)],
    S.map(s => [T.mtName[s.nm] || s.nm, txt(s.n.toLocaleString('en-US').replace(/,/g, ' ')), num(s.rs, 2), num(s.dfa, 2), num(s.g.d, 2), txt(`${f2(s.w.d)} (${f2(s.w.se)})` + (s.w.d >= 0.75 ? '*' : '')), txt(civ(s.w.d, s.w.se))]),
    { cap: T.mtCap(f2(a)), note: T.mtNote });
  const un = S[5], c = ci(un.w.d, un.w.se), r12 = S[2], rm = S[1];
  out('mt-out', [T.mt1(f2(un.w.d), civ(un.w.d, un.w.se), c[0] <= 1 && c[1] >= 1, un.w.d >= 0.75), T.mt2(f2(r12.w.d), f2(rm.w.d), f2(S[6].w.d), f2(S[7].w.d))]);
}

// ================================================================= EXTRA. the forecast comparison in numbers
function blkRt() {
  const k = +$('rt-k').value, h = +$('rt-h').value; set('rt-k', k + '%');
  const S = fcState(k), hh = range(1, 24, 1), sym = { ARFIMA: 'circle', AR: 'diamond', Mean: 'square', RW: 'triangle-up' };
  const G = grid('rt-chart', S.map((s, i) => ({ r: 0, c: i, t: `${s.c === 'ro' ? 'Romania' : 'United States'}: RMSE (pp per month${s.c === 'us' ? ', annualised' : ''})`, xt: 'horizon h (months)' })), { cols: 2, h: 360, legendRows: 2 });
  plot('rt-chart', S.flatMap((s, i) => FCM.map(m => ln(hh, s.rm[m], G.ax[i], FCC[m], i ? '' : FCL[m], { mode: 'lines+markers', marker: { size: 5, symbol: sym[m], color: FCC[m] } }))), G.lay);
  table('rt-tb', [L.model, H_(`RMSE RO, h = ${h}`), H_(L.ratioAR), H_(`RMSE ${T.usA}, h = ${h}`), H_(L.ratioAR)],
    FCM.map(m => { const a = S[0].rm[m][h - 1], b = S[1].rm[m][h - 1], ra = a / S[0].rm.AR[h - 1], rb = b / S[1].rm.AR[h - 1];
      return [T.fcName[m], num(a, 2), num(ra, 3, ra < 1), num(b, 2), num(rb, 3, rb < 1)]; }), { cap: T.rtCap(S[0].n, T.my(S[0].o0), S[1].n, T.my(S[1].o0)) });
  const r = S[0].rm;
  out('rt-out', [T.rt1(f2(r.ARFIMA[11]), f2(r.AR[11]), f2(r.Mean[11]), f2(r.RW[11]))]);
}

// ================================================================= EXTRA. realised volatility: d by estimator
function blkRq() {
  const a = +$('rq-a').value; set('rq-a', f2(a));
  const S = ['sp500', 'bet'].map(k => { const P = PG('rv' + k, RV[k]), m = bwm(RV[k].length, a), g = gph(P, m), w = lw(P, m), R = RVN[k];
    return { k, m, est: [['LW', w.d, w.se], ['GPH', g.d, g.se], ['ML', R.ml, R.ml_se]] }; });
  const labs = [], vals = [], err = [], cols = [];
  S.forEach(s => s.est.forEach(([e, d, se], j) => { labs.push(`${s.k === 'sp500' ? 'S&P 500' : 'BET'}, ${e}`); vals.push(d); err.push(1.96 * se); cols.push([K.blue, K.red, K.navy][j]); }));
  const G = grid('rq-chart', [{ r: 0, c: 0, xt: 'd of log RV: estimate and 95% interval' }], { cols: 1, h: 320, legendRows: 1, ml: 110 });
  const ax = G.ax[0];
  G.lay.shapes.push(vl(ax, 0, K.navy, 'solid', 0.8), vl(ax, 0.5, K.purple, 'dot', 1.2));
  plot('rq-chart', [{ x: vals, y: labs, type: 'scatter', mode: 'markers', marker: { size: 9, color: cols }, error_x: { type: 'data', array: err, color: K.navy, thickness: 1.3, width: 5 }, xaxis: ax.xa, yaxis: ax.ya, showlegend: false, hovertemplate: '%{y}: %{x:.3f}<extra></extra>' },
    leg('d = 0.5', { color: K.purple, dash: 'dot', width: 1.2 })], G.lay);
  table('rq-tb', [L.est, H_('S&P 500: d (SE)'), H_(L.ci95), H_('BET: d (SE)'), H_(L.ci95)],
    [0, 1, 2].map(j => [T.rqName[j] + (j < 2 ? ` (m = ${S[0].m})` : ''), txt(`${f2(S[0].est[j][1])} (${f2(S[0].est[j][2])})`), txt(civ(S[0].est[j][1], S[0].est[j][2])), txt(`${f2(S[1].est[j][1])} (${f2(S[1].est[j][2])})`), txt(civ(S[1].est[j][1], S[1].est[j][2]))])
      .concat([[T.rqBic, txt(`${nf(RVN.sp500.bic_f, 1)} / ${nf(RVN.sp500.bic_a, 1)}`, RVN.sp500.bic_f < RVN.sp500.bic_a), '', txt(`${nf(RVN.bet.bic_f, 1)} / ${nf(RVN.bet.bic_a, 1)}`, RVN.bet.bic_f < RVN.bet.bic_a), '']]),
    { cap: T.rqCap(RVN.sp500.T) });
  const all = S.flatMap(s => s.est.map(e => e[1]));
  out('rq-out', [T.rq1(f2(Math.min(...all)), f2(Math.max(...all)))]);
}

// ================================================================= EXTRA. the HAR regression, step by step
function blkWh() {
  const e = +$('wh-e').value; set('wh-e', e === 2026 ? T.sep26 : e);
  const H = harFit(e), k22 = range(1, 22, 1), n0 = Math.max(0, H.y.length - 250);
  const G = grid('wh-chart', [{ r: 0, c: 0, t: 'Implied weight of lag k', xt: 'lag k (days)' }, { r: 0, c: 1, cs: 2, t: 'Last 250 days of the sample: volatility and HAR fit (%)', x: DATE }], { cols: 3, h: 340, legendRows: 2 });
  const [a, b] = G.ax;
  plot('wh-chart', [bar(k22, H.w, a, `rgba(${RGB[K.teal]},0.7)`, 'HAR weights'), ln(H.d.slice(n0), H.y.slice(n0), b, K.blue, 'S&P 500 range volatility (Parkinson)', { line: { width: 0.9 } }),
    ln(H.d.slice(n0), H.fit.slice(n0), b, K.red, 'HAR fitted value', { line: { width: 1.3 } })], G.lay);
  steps('wh-ws', [T.whs1(T.dmy(H.d[0]), T.dmy(H.d[H.d.length - 1]), H.n), T.whs2(f3(H.c), f3(H.bd), f3(H.bw), f3(H.bm), f3(H.bd + H.bw + H.bm)),
    T.whs3(f3(H.bd), f3(H.bw), f3(H.bm), f3(H.w[0])), T.whs4(f3(H.bw), f3(H.bm), f3(H.w[1]), nf(H.w[5], 4))]);
  out('wh-out', [T.wh1(e === 2026 ? T.sep26 : String(e), f2(H.bd), f2(H.bw), f2(H.bm), f2(H.bd + H.bw + H.bm))]);
}

// ================================================================= EXTRA. HAR against (1 - L)^d in numbers
function blkHb() {
  const dd = +$('hb-d').value; set('hb-d', (dd >= 0 ? '+' : '') + f2(dd));
  const H = harFit(2026), dl = parkD() + dd, pw = Array.from(fw(dl, 23).slice(1), v => -v), k22 = range(1, 22, 1), ks = [1, 2, 6, 22];
  const G = grid('hb-chart', [{ r: 0, c: 0, xt: 'lag k (days)', yt: 'weight (log scale)', y: LOG }], { cols: 1, h: 320, legendRows: 1 });
  const ax = G.ax[0];
  plot('hb-chart', [bar(k22, H.w, ax, `rgba(${RGB[K.teal]},0.6)`, 'HAR'), ln(k22, pw, ax, K.purple, `-pi_k of (1-L)^d, d = ${dl.toFixed(2)}`, { mode: 'lines+markers', marker: { size: 5, color: K.purple } })], G.lay);
  table('hb-tb', [L.lag, ...ks.map(k => H_('k = ' + k))], [[T.hbHar, ...ks.map(k => num(H.w[k - 1], 3))], [T.hbFd(f2(dl)), ...ks.map(k => num(pw[k - 1], 3))]], { cap: T.hbCap });
  const g = ['sp500', 'bet'].map(k => { const v = VOL[k]; return [sci(v.garch.alpha * v.garch.beta ** 99), sci(figW(k)[99]), nf(v.garch.bic - v.figarch.bic, 0), f3(v.garch.alpha + v.garch.beta)]; });
  out('hb-out', [T.hb1(f2(dl), ks.map(k => [nf(H.w[k - 1], 3), nf(pw[k - 1], 3)])), T.hb2(g)]);
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
// [anchor, prefix, function, input moved by ▶, its values (null: the options of a select), ms]; the lags of vol-models start after the
// lecture's 100 days and wrap around to lag 1, so that ▶ also passes the HAR lags 1-22
const BLK = [['memory-data', 'md', blkMd, 'md-k', range(1, 36, 5), 700], ['memory-acf', 'ma', blkMa, 'ma-k', range(0, 60, 5), 700],
  ['decay', 'dc', blkDc, 'dc-d', range(0.05, 0.45, 0.05), 600], ['weights', 'wt', blkWt, 'wt-d', range(0.05, 0.95, 0.05), 500],
  ['ffd', 'ff', blkFf, 'ff-y', range(2000, 2018, 2), 900], ['arfima-paths', 'ap', blkAp, 'ap-d', range(0.05, 0.45, 0.05), 700],
  ['fbm', 'fb', blkFb, 'fb-h', range(0.05, 0.45, 0.05), 700], ['rs-dfa', 'rd', blkRd, 'rd-n', range(6, 60, 6), 900],
  ['gph', 'gp', blkGp, 'gp-a', range(0.4, 0.8, 0.05), 700], ['bandwidth', 'bw', blkBw, 'bw-e', range(2014, 2026, 2), 800],
  ['mc', 'mc', blkMc, 'mc-r', range(30, 300, 30), 600], ['arfima-fit', 'af', blkAf, 'af-d', range(-0.3, 0.18, 0.04), 600],
  ['forecast-path', 'fp', blkFp, 'fp-o', null, 1000], ['forecast', 'fo', blkFo, 'fo-k', range(0, 80, 10), 700],
  ['vol-acf', 'va', blkVa, 'va-p', range(0.25, 3, 0.25), 900], ['rv', 'rv', blkRv, 'rv-d', range(-0.3, 0.08, 0.02), 500],
  ['vol-models', 'vm', blkVm, 'vm-k', [8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7], 800], ['spurious', 'sp', blkSp, 'sp-r', range(20, 200, 20), 600],
  ['nile-break', 'nb', blkNb, 'nb-y', range(1880, 1960, 4), 600], ['rolling', 'rl', blkRl, 'rl-w', [0.5, 0.75, 1, 1.25, 1.5], 1300],
  ['wx-decay', 'wd', blkWd, 'wd-d', range(0.05, 0.45, 0.05), 600], ['wx-weights', 'ww', blkWw, 'ww-d', range(0.1, 1.5, 0.1), 600],
  ['d-range', 'dr', blkDr, 'dr-d', range(-0.6, 1.2, 0.1), 600], ['wx-acf', 'wa', blkWa, 'wa-d', range(0.05, 0.45, 0.05), 600],
  ['wx-rs', 'wr', blkWr, 'wr-x', range(-1.5, 1.5, 0.3), 500], ['wx-gph', 'wg', blkWg, 'wg-a', range(0.3, 0.9, 0.05), 600],
  ['gph-table', 'gt', blkGt, 'gt-a', range(0.4, 0.8, 0.05), 700], ['wx-lw', 'wl', blkWl, 'wl-a', range(0.4, 0.8, 0.05), 700],
  ['fit-table', 'ft', blkFt, 'ft-e', null, 1000], ['memory-table', 'mt', blkMt, 'mt-a', range(0.5, 0.8, 0.05), 1100],
  ['ratio-table', 'rt', blkRt, 'rt-k', range(0, 80, 10), 700], ['rv-table', 'rq', blkRq, 'rq-a', range(0.4, 0.8, 0.05), 700],
  ['wx-har', 'wh', blkWh, 'wh-e', range(2006, 2026, 2), 700], ['har-table', 'hb', blkHb, 'hb-d', range(-0.3, 0.3, 0.05), 600]];
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
window.CH8 = { NILE, RO, US, RV, SP, BET, EUR, PK, pgram, gph, lw, acfv, bsizes, rsCurve, dfaCurve, hRS, hDFA, fw, acvf0, acf0, acvf, dlLik, mlD, circ, adf1, ffd, figarchW, arfimaFc,
  lgamma, gammaf, decayState, acfState, ffdState, pathState, rsState, gphState, bwState, fpState, fcState, volState, rvState, harFit, parkD, figW, nileState, rlState, mtState, BLK, init };
if (!window.CH8_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
