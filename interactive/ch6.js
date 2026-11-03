// Chapter 6 interactive page (VAR models and Granger causality): one block per chart slide, plus the result tables and the
// worked examples of the lecture, computed in the browser from ch6_data.js.
(function () {
'use strict';
const D = window.CH6_DATA;
const { sum, chol, quantile, mean, nf, $, AX, layout, plot, out, Player, range, predict, exitTicket } = window.TSAI;
const T = window.CH6_TEXT;
const L = T.L;
const NB = D.numbers;

// ---------------------------------------------------------------- colours of generate_all_charts.py / tsa_style.py
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', purple: '#8E44AD', teal: '#17A2B8', orange: '#E67E22', navy: '#1F2A44' };
const RGB = { [K.blue]: '26,58,110', [K.red]: '205,0,0', [K.green]: '46,125,50', [K.purple]: '142,68,173', [K.amber]: '181,133,63',
  [K.teal]: '23,162,184', [K.orange]: '230,126,34', [K.navy]: '31,42,68' };
const RV = ['g', 'pi', 'i'];
const LAB = { g: 'GDP growth', pi: 'inflation', i: 'ROBOR 3M', u: 'unemployment', R: 'fed funds rate', sp500: 'S&P 500', dax: 'DAX', bet: 'BET' };
const COLV = { g: K.green, pi: K.red, i: K.blue, u: K.purple, R: K.blue, sp500: K.blue, dax: K.teal, bet: K.red };
const MK = ['sp500', 'dax', 'bet'];

// ---------------------------------------------------------------- small helpers
const set = (id, v) => { const e = $(id + '-v'); if (e) e.textContent = v; };
const pe = p => (p < 0.001 ? 'p &lt; ' + nf(0.001, 3) : 'p = ' + nf(p, 3));
const pv = p => (p < 0.001 ? '&lt; ' + nf(0.001, 3) : nf(p, 3));
const pn = (x, d) => (x < 0 ? `(${nf(x, d)})` : nf(x, d));      // a factor in a product: negative numbers in parentheses
const has = id => !!$(id);
const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const eye = n => zeros(n, n).map((r, i) => { r[i] = 1; return r; });
const mm = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const tr = A => A[0].map((_, j) => A.map(r => r[j]));
const madd = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
const msub = (A, B) => A.map((r, i) => r.map((v, j) => v - B[i][j]));
const mv = (A, x) => A.map(r => r.reduce((s, v, k) => s + v * x[k], 0));
function inv(A) {
  const n = A.length, M = A.map((r, i) => [...r, ...eye(n)[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c]; for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c]; if (f) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j]; }
  }
  return M.map(r => r.slice(n));
}
function det(A) { const n = A.length, M = A.map(r => [...r]); let d = 1;
  for (let c = 0; c < n; c++) { let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (p !== c) { [M[c], M[p]] = [M[p], M[c]]; d = -d; } d *= M[c][c]; if (!M[c][c]) return 0;
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let j = c; j < n; j++) M[r][j] -= f * M[c][j]; } }
  return d; }
const corr = (a, b) => { const ma = mean(a), mb = mean(b); let s = 0, sa = 0, sb = 0; a.forEach((v, i) => { s += (v - ma) * (b[i] - mb); sa += (v - ma) ** 2; sb += (b[i] - mb) ** 2; }); return s / Math.sqrt(sa * sb); };
const sd1 = x => { const m = mean(x); return Math.sqrt(sum(x.map(v => (v - m) ** 2)) / (x.length - 1)); };
const pvar = x => { const m = mean(x); return sum(x.map(v => (v - m) ** 2)) / x.length; };
function skewK(x) { const n = x.length, m = mean(x); let m2 = 0, m3 = 0, m4 = 0; x.forEach(v => { const d = v - m; m2 += d * d; m3 += d ** 3; m4 += d ** 4; }); m2 /= n; m3 /= n; m4 /= n;
  const g1 = m3 / m2 ** 1.5, g2 = m4 / (m2 * m2) - 3;                        // pandas: bias-corrected skewness and excess kurtosis
  return { s: Math.sqrt(n * (n - 1)) / (n - 2) * g1, k: ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3)) + 3 }; }

// ---------------------------------------------------------------- distributions
function gammaln(z) {
  const g = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let x = z, y = z, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let s = 1.000000000190015;
  for (const c of g) s += c / ++y;
  return -t + Math.log(2.5066282746310005 * s / x);
}
function betacf(a, b, x) {
  let c = 1, d = 1 - (a + b) * x / (a + 1); if (Math.abs(d) < 1e-300) d = 1e-300; d = 1 / d; let h = d;
  for (let m = 1; m <= 400; m++) {
    const m2 = 2 * m; let aa = m * (b - m) * x / ((a - 1 + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; h *= d * c;
    aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2));
    d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return h;
}
function ibeta(x, a, b) {        // regularised incomplete beta I_x(a, b)
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
const fsf = (F, d1, d2) => (F <= 0 ? 1 : ibeta(d2 / (d2 + d1 * F), d2 / 2, d1 / 2));
function fppf(q, d1, d2) { let lo = 0, hi = 1000; for (let i = 0; i < 200; i++) { const m = (lo + hi) / 2; if (1 - fsf(m, d1, d2) < q) lo = m; else hi = m; } return (lo + hi) / 2; }
function chi2sf(x, k) {
  const a = k / 2, z = x / 2;
  if (z <= 0) return 1;
  if (z < a + 1) { let ap = a, s = 1 / a, d = s; for (let n = 0; n < 1000; n++) { d *= z / ++ap; s += d; if (Math.abs(d) < Math.abs(s) * 1e-15) break; }
    return 1 - s * Math.exp(-z + a * Math.log(z) - gammaln(a)); }
  let b = z + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (let i = 1; i < 1000; i++) { const an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300; c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-15) break; }
  return Math.exp(-z + a * Math.log(z) - gammaln(a)) * h;
}
const normCdf = window.TSAI.normCdf;
const Z975 = 1.959963984540054;

// ---------------------------------------------------------------- eigenvalues of a real matrix (Hessenberg reduction + shifted QR, Numerical Recipes hqr)
function eigvals(A0) {
  const n = A0.length, a = A0.map(r => [...r]);
  for (let m = 1; m < n - 1; m++) {
    let x = 0, i = m;
    for (let j = m; j < n; j++) if (Math.abs(a[j][m - 1]) > Math.abs(x)) { x = a[j][m - 1]; i = j; }
    if (i !== m) { [a[i], a[m]] = [a[m], a[i]]; for (let j = 0; j < n; j++) { const t = a[j][i]; a[j][i] = a[j][m]; a[j][m] = t; } }
    if (x) for (i = m + 1; i < n; i++) { let y = a[i][m - 1]; if (y) { y /= x; a[i][m - 1] = y; for (let j = m; j < n; j++) a[i][j] -= y * a[m][j]; for (let j = 0; j < n; j++) a[j][m] += y * a[j][i]; } }
  }
  for (let i = 2; i < n; i++) for (let j = 0; j < i - 1; j++) a[i][j] = 0;
  const EPS = 2.220446049250313e-16, wr = new Array(n).fill(0), wi = new Array(n).fill(0);
  let nn = n - 1, t = 0, anorm = 0, l, m, its, x, y, z, w, v, u, s, r = 0, q = 0, p = 0;
  for (let i = 0; i < n; i++) for (let j = Math.max(i - 1, 0); j < n; j++) anorm += Math.abs(a[i][j]);
  while (nn >= 0) {
    its = 0;
    do {
      for (l = nn; l > 0; l--) { s = Math.abs(a[l - 1][l - 1]) + Math.abs(a[l][l]); if (s === 0) s = anorm; if (Math.abs(a[l][l - 1]) <= EPS * s) { a[l][l - 1] = 0; break; } }
      x = a[nn][nn];
      if (l === nn) { wr[nn] = x + t; wi[nn--] = 0; }
      else {
        y = a[nn - 1][nn - 1]; w = a[nn][nn - 1] * a[nn - 1][nn];
        if (l === nn - 1) {
          p = 0.5 * (y - x); q = p * p + w; z = Math.sqrt(Math.abs(q)); x += t;
          if (q >= 0) { z = p + (p >= 0 ? Math.abs(z) : -Math.abs(z)); wr[nn - 1] = wr[nn] = x + z; if (z) wr[nn] = x - w / z; wi[nn - 1] = wi[nn] = 0; }
          else { wr[nn - 1] = wr[nn] = x + p; wi[nn - 1] = z; wi[nn] = -z; }
          nn -= 2;
        } else {
          if (its === 60) throw new Error('hqr');
          if (its === 10 || its === 20) { t += x; for (let i = 0; i <= nn; i++) a[i][i] -= x; s = Math.abs(a[nn][nn - 1]) + Math.abs(a[nn - 1][nn - 2]); y = x = 0.75 * s; w = -0.4375 * s * s; }
          ++its;
          for (m = nn - 2; m >= l; m--) {
            z = a[m][m]; r = x - z; s = y - z; p = (r * s - w) / a[m + 1][m] + a[m][m + 1]; q = a[m + 1][m + 1] - z - r - s; r = a[m + 2][m + 1];
            s = Math.abs(p) + Math.abs(q) + Math.abs(r); p /= s; q /= s; r /= s;
            if (m === l) break;
            u = Math.abs(a[m][m - 1]) * (Math.abs(q) + Math.abs(r)); v = Math.abs(p) * (Math.abs(a[m - 1][m - 1]) + Math.abs(z) + Math.abs(a[m + 1][m + 1]));
            if (u <= EPS * v) break;
          }
          for (let i = m; i < nn - 1; i++) { a[i + 2][i] = 0; if (i !== m) a[i + 2][i - 1] = 0; }
          for (let k = m; k < nn; k++) {
            if (k !== m) { p = a[k][k - 1]; q = a[k + 1][k - 1]; r = 0; if (k + 1 !== nn) r = a[k + 2][k - 1]; if ((x = Math.abs(p) + Math.abs(q) + Math.abs(r)) !== 0) { p /= x; q /= x; r /= x; } }
            const sq = Math.sqrt(p * p + q * q + r * r); s = p >= 0 ? sq : -sq;
            if (s !== 0) {
              if (k === m) { if (l !== m) a[k][k - 1] = -a[k][k - 1]; } else a[k][k - 1] = -s * x;
              p += s; x = p / s; y = q / s; z = r / s; q /= p; r /= p;
              for (let j = k; j <= nn; j++) { p = a[k][j] + q * a[k + 1][j]; if (k + 1 !== nn) { p += r * a[k + 2][j]; a[k + 2][j] -= p * z; } a[k + 1][j] -= p * y; a[k][j] -= p * x; }
              const mmin = nn < k + 3 ? nn : k + 3;
              for (let i = l; i <= mmin; i++) { p = x * a[i][k] + y * a[i][k + 1]; if (k + 1 !== nn) { p += z * a[i][k + 2]; a[i][k + 2] -= p * r; } a[i][k + 1] -= p * q; a[i][k] -= p; }
            }
          }
        }
      }
    } while (l + 1 < nn);
  }
  return wr.map((re, i) => ({ re, im: wi[i], mod: Math.hypot(re, wi[i]) })).sort((a1, b1) => b1.mod - a1.mod || b1.im - a1.im);
}

// ---------------------------------------------------------------- the VAR toolbox: OLS equation by equation, MA form, IRF, FEVD
function design(Y, p, ex) {      // rows t = p..N-1: [1, Y_{t-1}, ..., Y_{t-p}, exog_t]
  const N = Y.length, Kk = Y[0].length, X = [];
  for (let t = p; t < N; t++) { const r = [1]; for (let j = 1; j <= p; j++) for (let k = 0; k < Kk; k++) r.push(Y[t - j][k]); if (ex) r.push(...ex[t]); X.push(r); }
  return X;
}
function crossp(X, Y) {          // X'X and X'Y
  const m = X[0].length, Kk = Y[0].length, G = zeros(m, m), H = zeros(m, Kk);
  for (let t = 0; t < X.length; t++) { const x = X[t], y = Y[t];
    for (let a = 0; a < m; a++) { const v = x[a]; if (!v) continue; const Ga = G[a]; for (let b = a; b < m; b++) Ga[b] += v * x[b]; const Ha = H[a]; for (let k = 0; k < Kk; k++) Ha[k] += v * y[k]; } }
  for (let a = 0; a < m; a++) for (let b = 0; b < a; b++) G[a][b] = G[b][a];
  return { G, H };
}
function varFit(Y, p, ex) {
  const Kk = Y[0].length, X = design(Y, p, ex), Z = Y.slice(p), n = X.length, m = X[0].length;
  const { G, H } = crossp(X, Z), Gi = inv(G), B = mm(Gi, H);
  const U = Z.map((z, t) => z.map((v, k) => v - X[t].reduce((s, x, a) => s + x * B[a][k], 0)));
  const S = zeros(Kk, Kk); U.forEach(u => { for (let a = 0; a < Kk; a++) for (let b = 0; b < Kk; b++) S[a][b] += u[a] * u[b]; });
  const Sd = S.map(r => r.map(v => v / (n - m)));
  const A = Array.from({ length: p }, (_, j) => Array.from({ length: Kk }, (_, i) => Array.from({ length: Kk }, (_, k) => B[1 + j * Kk + k][i])));
  const c = B[0].slice();
  const se = B.map((r, a) => r.map((_, i) => Math.sqrt(Gi[a][a] * Sd[i][i])));
  return { B, U, S: Sd, A, c, n, m, p, K: Kk, Gi, se, X, Z };
}
function maRep(A, H) {
  const Kk = A[0].length, p = A.length, Ph = [eye(Kk)];
  for (let h = 1; h <= H; h++) { let M = zeros(Kk, Kk); for (let j = 1; j <= Math.min(h, p); j++) M = madd(M, mm(A[j - 1], Ph[h - j])); Ph.push(M); }
  return Ph;
}
const orthIrf = (f, H) => { const P = chol(f.S); return maRep(f.A, H).map(M => mm(M, P)); };
function fevdOf(Th) {            // [i][h][j]: share of shock j in the (h+1)-step forecast error variance of i
  const Kk = Th[0].length, out = zeros(Kk, Th.length).map(r => r.map(() => new Array(Kk).fill(0)));
  for (let i = 0; i < Kk; i++) { const acc = new Array(Kk).fill(0);
    Th.forEach((M, h) => { for (let j = 0; j < Kk; j++) acc[j] += M[i][j] ** 2; const tot = sum(acc); for (let j = 0; j < Kk; j++) out[i][h][j] = acc[j] / tot; }); }
  return out;
}
const girfOf = (f, H) => { const S = f.S, d = S.map((r, j) => Math.sqrt(r[j])); return maRep(f.A, H).map(M => mm(M, S).map(r => r.map((v, j) => v / d[j]))); };
function gfevd(f, H) {
  const S = f.S, Kk = f.K, Ph = maRep(f.A, H - 1), num = zeros(Kk, Kk), den = new Array(Kk).fill(0);
  Ph.forEach(M => { const MS = mm(M, S), MSM = mm(MS, tr(M)); for (let i = 0; i < Kk; i++) { den[i] += MSM[i][i]; for (let j = 0; j < Kk; j++) num[i][j] += MS[i][j] ** 2 / S[j][j]; } });
  const th = num.map((r, i) => r.map(v => v / den[i]));
  return th.map(r => { const s = sum(r); return r.map(v => v / s); });
}
const spillIdx = th => 100 * (sum(th.map(r => sum(r))) - sum(th.map((r, i) => r[i]))) / th.length;
function companion(f) {
  const Kk = f.K, p = f.p, F = zeros(Kk * p, Kk * p);
  for (let i = 0; i < Kk; i++) for (let j = 0; j < p; j++) for (let k = 0; k < Kk; k++) F[i][j * Kk + k] = f.A[j][i][k];
  for (let i = Kk; i < Kk * p; i++) F[i][i - Kk] = 1;
  return F;
}
function granger(f, Y, cause, effect) {   // F test of the p lags of `cause` in the equation of `effect` (same sample and regressors as f)
  const Kk = f.K, p = f.p, keep = [0];
  for (let j = 0; j < p; j++) for (let k = 0; k < Kk; k++) if (k !== cause) keep.push(1 + j * Kk + k);
  const yv = f.Z.map(z => z[effect]), yy = sum(yv.map(v => v * v));
  const rssOf = cols => { const X = f.X, m = cols.length, G = zeros(m, m), h = new Array(m).fill(0);
    for (let t = 0; t < X.length; t++) { const x = X[t]; for (let a = 0; a < m; a++) { const v = x[cols[a]]; h[a] += v * yv[t]; for (let b = a; b < m; b++) G[a][b] += v * x[cols[b]]; } }
    for (let a = 0; a < m; a++) for (let b = 0; b < a; b++) G[a][b] = G[b][a];
    const bb = mv(inv(G), h); return yy - sum(bb.map((v, a) => v * h[a])); };
  const rssU = sum(f.U.map(u => u[effect] ** 2)), rssR = rssOf(keep), df2 = f.n - Kk * p - 1;
  const F = ((rssR - rssU) / p) / (rssU / df2);
  return { F, p: fsf(F, p, df2), rssU, rssR, df1: p, df2, n: f.n, crit: fppf(0.95, p, df2) };
}
function forecast(f, Y, H) {
  const p = f.p, hist = Y.slice(-p).map(r => [...r]), pt = [];
  for (let h = 0; h < H; h++) { const y = f.c.slice(); for (let j = 1; j <= p; j++) { const z = hist[hist.length - j]; for (let i = 0; i < f.K; i++) for (let k = 0; k < f.K; k++) y[i] += f.A[j - 1][i][k] * z[k]; } pt.push(y); hist.push(y); }
  const Ph = maRep(f.A, H - 1); let M = zeros(f.K, f.K); const sd = [];
  Ph.forEach(P => { M = madd(M, mm(mm(P, f.S), tr(P))); sd.push(M.map((r, i) => Math.sqrt(r[i]))); });
  return { pt, sd };
}
const meanOf = f => mv(inv(f.A.reduce((M, A) => msub(M, A), eye(f.K))), f.c);

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
const hband = (a, y0, y1, col, al) => ({ type: 'rect', xref: a.xd, yref: a.ya, x0: 0, x1: 1, y0, y1, fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, layer: 'below' });
const leg = (name, line, marker) => Object.assign({ x: [null], y: [null], name, hoverinfo: 'skip' }, line ? { mode: 'lines', line } : marker.fill ? { mode: 'markers', marker: { color: marker.fill, symbol: 'square', size: 11 } } : { mode: 'markers', marker });
const ln = (x, y, a, color, name, o) => Object.assign({ x, y, mode: 'lines', line: { color, width: 1.5 }, name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya }, o || {});
const bar = (x, y, a, color, name, o) => Object.assign({ x, y, type: 'bar', marker: { color }, name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya }, o || {});
const band = (x, lo, hi, a, col, al, name) => ({ x: [...x, ...[...x].reverse()], y: [...hi, ...[...lo].reverse()], fill: 'toself', fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, mode: 'lines', hoverinfo: 'skip', name: name || '', showlegend: !!name, xaxis: a.xa, yaxis: a.ya });

// ---------------------------------------------------------------- tables and worked steps (HTML)
function table(id, head, rows, o) {
  const el = $(id); if (!el) return;
  const cell = (c, tag) => { const v = typeof c === 'object' && c !== null ? c : { v: c }; return `<${tag}${v.cls ? ` class="${v.cls}"` : ''}${v.span ? ` colspan="${v.span}"` : ''}>${v.v}</${tag}>`; };
  el.innerHTML = (o && o.cap ? `<p class="c6cap">${o.cap}</p>` : '') + '<table><thead>' + (o && o.head2 ? '<tr>' + o.head2.map(c => cell(c, 'th')).join('') + '</tr>' : '')
    + '<tr>' + head.map(c => cell(c, 'th')).join('') + '</tr></thead><tbody>'
    + rows.map(r => '<tr>' + r.map(c => cell(c, 'td')).join('') + '</tr>').join('') + '</tbody></table>' + (o && o.note ? `<p class="c6cap">${o.note}</p>` : '');
}
const num = (x, d, bold) => ({ v: bold ? `<b>${nf(x, d)}</b>` : nf(x, d), cls: 'r' });
const steps = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
const vec = (v, d) => `(${v.map(x => nf(x, d)).join('; ')})′`;
const mat = (M, d) => '[' + M.map(r => r.map(x => nf(x, d)).join('  ')).join(' | ') + ']';
const sgn = (x, d) => (x < 0 ? ' − ' + nf(-x, d) : ' + ' + nf(x, d));

// ---------------------------------------------------------------- data of the chapter
const qiso = (y0, k) => { const m = 3 * (k % 4) + 1, y = y0 + Math.floor(k / 4); return `${y}-${String(m).padStart(2, '0')}-01`; };
const miso = (y0, k) => { const y = y0 + Math.floor(k / 12), m = k % 12 + 1; return `${y}-${String(m).padStart(2, '0')}-01`; };
const RO = { d: D.ro.g.map((_, k) => qiso(2000, k)), g: D.ro.g, pi: D.ro.pi, i: D.ro.i };
RO.dm = D.ro.pim.map((_, k) => miso(2000, k)); RO.dfx = D.ro.fxm.map((_, k) => miso(2000, k));
const RO0 = RO.d.indexOf('2005-01-01'), ROE = RO.d.length - 1;
const ORD = { gpi: [0, 1, 2], gip: [0, 2, 1], pgi: [1, 0, 2], pig: [1, 2, 0], igp: [2, 0, 1], ipg: [2, 1, 0] };
const ORDN = o => ORD[o].map(k => T.v[RV[k]]).join(', ');
function roY(end, ord, i0) { const e = end === undefined ? ROE : end, o = ORD[ord || 'gpi'], out = [];
  for (let t = i0 === undefined ? RO0 : i0; t <= e; t++) out.push(o.map(k => RO[RV[k]][t])); return out; }
const roIdx = iso => RO.d.indexOf(iso);
const reord = (M, o) => { const inv_ = [0, 1, 2].map(k => o.indexOf(k)); return [0, 1, 2].map(i => [0, 1, 2].map(j => M[inv_[i]][inv_[j]])); };   // to the canonical order g, pi, i

const MKT = (() => { const d = [D.mkt.d0]; let t = Date.parse(D.mkt.d0 + 'T00:00:00Z'); D.mkt.gap.forEach(g0 => { t += g0 * 864e5; d.push(new Date(t).toISOString().slice(0, 10)); });
  return { d, r: MK.map(k => D.mkt[k]) }; })();
const mktY = (i0, i1, ord) => { const o = ord || [0, 1, 2], Y = []; for (let t = i0; t < i1; t++) Y.push(o.map(k => MKT.r[k][t])); return Y; };
const mIdx = day => { const d = MKT.d; let lo = 0, hi = d.length; while (lo < hi) { const m = (lo + hi) >> 1; if (d[m] < day) lo = m + 1; else hi = m; } return lo; };

const US = { d: D.us.pi.map((_, k) => qiso(1960, k)), pi: D.us.pi, u: D.us.u, R: D.us.R };
const usY = (end, ord) => { const o = ord || [0, 1, 2], V = ['pi', 'u', 'R'], Y = []; for (let t = 0; t <= end; t++) Y.push(o.map(k => US[V[k]][t])); return Y; };
const US_SW = US.d.indexOf('2000-10-01');

const BOOT = (() => { const s = D.boot, a = new Array(s.length / 2); for (let i = 0; i < a.length; i++) a[i] = parseInt(s.substr(2 * i, 2), 36); return a; })();
const cache = {};
const memo = (k, f) => (k in cache ? cache[k] : (cache[k] = f()));

// ================================================================= 1. Romania: four macro series
function blkRoMacro() {
  const y0 = +$('rm-y0').value, y1 = Math.max(+$('rm-y1').value, y0 + 3); $('rm-y1').value = y1; set('rm-y0', y0); set('rm-y1', y1);
  const a = `${y0}-01-01`, b = `${y1}-12-31`;
  const G = grid('rm-chart', [{ r: 0, c: 0, t: 'Real GDP growth (quarterly, SA)', x: { type: 'date' } }, { r: 0, c: 1, t: 'Inflation and the 3-month interest rate', x: { type: 'date' }, y: { range: [-3, 32] } },
    { r: 1, c: 0, t: 'Unemployment rate (monthly, SA)', x: { type: 'date' } }, { r: 1, c: 1, t: 'EUR/RON, BNR reference rate', x: { type: 'date' } }], { cols: 2, h: 470, gyPx: 56, legendRows: 2 });
  const A = G.ax;
  const tr_ = [bar(RO.d, RO.g, A[0], K.green, 'real GDP growth, q/q (%)', { hovertemplate: '%{x|%Y-%m}: %{y:.2f}%<extra></extra>' }),
    ln(RO.dm, D.ro.pim, A[1], K.red, '12-month HICP inflation (%)', { hovertemplate: '%{x|%Y-%m}: %{y:.2f}%<extra></extra>' }),
    ln(RO.dm, D.ro.im, A[1], K.blue, 'ROBOR 3M (% p.a.)', { hovertemplate: '%{x|%Y-%m}: %{y:.2f}%<extra></extra>' }),
    ln(RO.dm, D.ro.um, A[2], K.purple, 'unemployment rate (%, SA)', { hovertemplate: '%{x|%Y-%m}: %{y:.1f}%<extra></extra>' }),
    ln(RO.dfx, D.ro.fxm, A[3], K.amber, 'EUR/RON (monthly mean)', { hovertemplate: '%{x|%Y-%m}: %{y:.4f}<extra></extra>' }),
    leg(`selected window ${y0}-${Math.min(y1, 2026)}`, null, { fill: `rgba(${RGB[K.teal]},0.35)` })];
  A.forEach(ax => G.lay.shapes.push(rect(ax, a, b > '2026-09-30' ? '2026-09-30' : b, K.teal, 0.14)));
  G.lay.shapes.push(hl(A[0], 0, K.navy, 'solid', 0.6));
  plot('rm-chart', tr_, G.lay);
  const ix = RO.d.map((d, k) => k).filter(k => RO.d[k] >= a && RO.d[k] <= b), pick = v => ix.map(k => RO[v][k]);
  const X = RV.map(pick), C = RV.map((_, i) => RV.map((_, j) => corr(X[i], X[j])));
  const mi = RO.dm.map((d, k) => k).filter(k => RO.dm[k] >= a && RO.dm[k] <= b), mx = v => mi.reduce((m, k) => (v[k] > v[m] ? k : m), mi[0]);
  const kp = mx(D.ro.pim), ki = mx(D.ro.im);
  out('rm-out', [T.rm1(T.q(RO.d[ix[0]]), T.q(RO.d[ix[ix.length - 1]]), ix.length, nf(C[1][2], 2), nf(C[0][1], 2), nf(C[0][2], 2)),
    T.rm2(nf(D.ro.pim[kp], 1), T.my(RO.dm[kp]), nf(D.ro.im[ki], 1), T.my(RO.dm[ki]))]);
  table('rm-tb', [L.var, L.mean, L.sd, ...RV.map(v => T.v[v])], RV.map((v, i) => [T.v[v], num(mean(X[i]), 2), num(sd1(X[i]), 2), ...C[i].map((c, j) => num(c, 2, j !== i && Math.abs(c) >= 0.5))]),
    { cap: T.rmCap(T.q(RO.d[ix[0]]), T.q(RO.d[ix[ix.length - 1]]), ix.length) });
}

// ================================================================= 2. cross-correlations
function ccfOf(y, x, kmax) { const n = y.length, out_ = {};
  for (let k = -kmax; k <= kmax; k++) { const a = [], b = []; for (let t = 0; t < n; t++) { const s = t - k; if (s >= 0 && s < n) { a.push(y[t]); b.push(x[s]); } } out_[k] = corr(a, b); }
  return out_; }
function ccfStats(y0, tq) {
  const i0 = mIdx(`${y0}-01-01`), r = MKT.r.map(v => v.slice(i0));
  const dax = ccfOf(r[1], r[0], 5), bet = ccfOf(r[2], r[0], 5), band_ = 1.96 / Math.sqrt(r[0].length);
  let pi = RO.pi.slice(RO0), ii = RO.i.slice(RO0);
  if (tq === 'd1') { pi = pi.slice(1).map((v, k) => v - pi[k]); ii = ii.slice(1).map((v, k) => v - ii[k]); }
  const ip = ccfOf(ii, pi, 8), kmax = Object.keys(ip).map(Number).reduce((m, k) => (ip[k] > ip[m] ? k : m), -8);
  return { dax, bet, band: band_, n: r[0].length, d0: MKT.d[i0], ip, kmax, bq: 1.96 / Math.sqrt(ii.length) };
}
function blkCcf() {
  const y0 = +$('cc-y0').value, tq = $('cc-t').value; set('cc-y0', y0);
  const s = ccfStats(y0, tq), k5 = range(-5, 5, 1), k8 = range(-8, 8, 1);
  const G = grid('cc-chart', [{ r: 0, c: 0, t: `Daily returns, ${y0}-2026`, xt: 'lag k (days)' }, { r: 0, c: 1, t: `Romania, quarterly, 2005-2026${tq === 'd1' ? ', first differences' : ''}`, xt: 'lag k (quarters)' }], { cols: 2, h: 380, legendRows: 2 });
  const [a0, a1] = G.ax;
  plot('cc-chart', [bar(k5.map(k => k - 0.18), k5.map(k => s.dax[k]), a0, K.teal, 'DAX(t) vs S&P 500(t-k)', { width: 0.36, hovertemplate: 'k = %{text}: %{y:.3f}<extra>DAX</extra>', text: k5, textposition: 'none' }),
    bar(k5.map(k => k + 0.18), k5.map(k => s.bet[k]), a0, K.red, 'BET(t) vs S&P 500(t-k)', { width: 0.36, hovertemplate: 'k = %{text}: %{y:.3f}<extra>BET</extra>', text: k5, textposition: 'none' }),
    bar(k8, k8.map(k => s.ip[k]), a1, K.blue, tq === 'd1' ? 'change in ROBOR(t) vs change in inflation(t-k)' : 'ROBOR(t) vs inflation(t-k)', { width: 0.55, hovertemplate: 'k = %{x}: %{y:.3f}<extra></extra>' }),
    leg('+/- 1.96/sqrt(T)', { color: K.teal, dash: 'dash', width: 1.2 })],
  Object.assign(G.lay, { shapes: [hl(a0, s.band, K.teal, 'dash', 1), hl(a0, -s.band, K.teal, 'dash', 1), hl(a0, 0, K.navy, 'solid', 0.6), hl(a1, 0, K.navy, 'solid', 0.6),
    hl(a1, s.bq, K.teal, 'dash', 1), hl(a1, -s.bq, K.teal, 'dash', 1)] }));
  const outB = c => k5.filter(k => Math.abs(c[k]) > s.band).map(k => String(k).replace('-', '−'));
  out('cc-out', [T.cc1(s.d0, s.n, nf(s.band, 3)), T.cc2('DAX', nf(s.dax[0], 2), nf(s.dax[1], 3), nf(s.dax[-1], 3), outB(s.dax)), T.cc2('BET', nf(s.bet[0], 2), nf(s.bet[1], 3), nf(s.bet[-1], 3), outB(s.bet)),
    T.cc3(tq, s.kmax, nf(s.ip[s.kmax], 2), nf(s.ip[-4], 2), nf(s.ip[4], 2), nf(s.bq, 2))]);
  table('cc-tb', ['k', ...[-3, -2, -1, 0, 1, 2, 3].map(k => ({ v: String(k).replace('-', '−'), cls: 'r' }))],
    [['DAX', s.dax], ['BET', s.bet]].map(([n, c]) => [n, ...[-3, -2, -1, 0, 1, 2, 3].map(k => num(c[k], 3, Math.abs(c[k]) > s.band))]), { cap: T.ccCap(nf(s.band, 3)) });
}

// ================================================================= worked example: a VAR(1)
const WX = NB.wx, WXC = [1.2, 0.6], WXY = [5, 2], WXS = [[1, 0.5], [0.5, 1]];
const wxA = (a12, a21) => [[0.5, a12], [a21, 0.4]];
function eig2(A) { const t = A[0][0] + A[1][1], d = A[0][0] * A[1][1] - A[0][1] * A[1][0], q = t * t - 4 * d;
  return q >= 0 ? { t, d, q, re: true, l: [(t + Math.sqrt(q)) / 2, (t - Math.sqrt(q)) / 2], mod: Math.max(Math.abs((t + Math.sqrt(q)) / 2), Math.abs((t - Math.sqrt(q)) / 2)) }
    : { t, d, q, re: false, l: [t / 2, t / 2], im: Math.sqrt(-q) / 2, mod: Math.sqrt(d) }; }
function wxPath(A, S, H) { const f = [WXY]; for (let h = 1; h <= H; h++) f.push(WXC.map((c, i) => c + A[i][0] * f[h - 1][0] + A[i][1] * f[h - 1][1]));
  let M = zeros(2, 2), P = eye(2); const v = [[0, 0]]; for (let h = 1; h <= H; h++) { M = madd(M, mm(mm(P, S), tr(P))); P = mm(A, P); v.push([M[0][0], M[1][1]]); }
  return { f, v }; }
function blkWx1() {
  const a12 = +$('w1-a12').value, a21 = +$('w1-a21').value, h = +$('w1-h').value; set('w1-a12', nf(a12, 2)); set('w1-a21', nf(a21, 2)); set('w1-h', h);
  const A = wxA(a12, a21), e = eig2(A), H = 12, P = wxPath(A, WXS, H), stable = e.mod < 1;
  const dI = (1 - A[0][0]) * (1 - A[1][1]) - A[0][1] * A[1][0], mu = stable ? [((1 - A[1][1]) * WXC[0] + A[0][1] * WXC[1]) / dI, (A[1][0] * WXC[0] + (1 - A[0][0]) * WXC[1]) / dI] : null;
  const f1 = P.f[1], f2 = P.f[2], fh = P.f[h];
  steps('w1-ws', [T.w1s1(pn(A[0][1], 2), pn(A[1][0], 2), nf(f1[0], 2), nf(f1[1], 2)),
    T.w1s2(pn(f1[0], 2), pn(f1[1], 2), pn(A[0][1], 2), pn(A[1][0], 2), nf(f2[0], 2), nf(f2[1], 2)),
    T.w1s3(h, nf(fh[0], 2), nf(fh[1], 2), nf(Math.sqrt(P.v[h][0]) * Z975, 2), nf(Math.sqrt(P.v[h][1]) * Z975, 2), nf(P.v[h][0], 2), nf(P.v[h][1], 2)),
    T.w1s4(nf(e.t, 2), pn(A[0][1] * A[1][0], 2), nf(e.d, 3), e.re ? nf(e.q, 3) : null, e.re ? e.l.map(x => nf(x, 3)) : [nf(e.l[0], 3), nf(e.im, 3)], nf(e.mod, 3), stable),
    stable ? T.w1s5(pn(A[0][1], 2), pn(A[1][0], 2), nf(dI, 3), nf(mu[0], 2), nf(mu[1], 2)) : T.w1s5x(nf(dI, 3))]);
  const G = grid('w1-chart', [{ r: 0, c: 0, xt: 'forecast horizon h', yt: 'forecast' }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0], hh = range(0, H, 1), cap = v => (stable ? v : v);
  const lo = j => P.f.map((f, k) => f[j] - Z975 * Math.sqrt(P.v[k][j])), hi = j => P.f.map((f, k) => f[j] + Z975 * Math.sqrt(P.v[k][j]));
  const tr_ = [band(hh, lo(0), hi(0), a, K.blue, 0.12, '95% interval of y1'), band(hh, lo(1), hi(1), a, K.red, 0.12, '95% interval of y2'),
    ln(hh, P.f.map(f => cap(f[0])), a, K.blue, 'forecast of y1', { mode: 'lines+markers', marker: { size: 5 }, hovertemplate: 'h = %{x}: %{y:.3f}<extra>y1</extra>' }),
    ln(hh, P.f.map(f => cap(f[1])), a, K.red, 'forecast of y2', { mode: 'lines+markers', marker: { size: 5, symbol: 'square' }, hovertemplate: 'h = %{x}: %{y:.3f}<extra>y2</extra>' })];
  if (stable) tr_.push(leg('mean mu', { color: K.navy, dash: 'dash', width: 1 }));
  G.lay.shapes = stable ? [hl(a, mu[0], K.blue, 'dash', 1), hl(a, mu[1], K.red, 'dot', 1.2)] : [];
  G.lay.shapes.push(vl(a, h, K.teal, 'dot', 1.2));
  plot('w1-chart', tr_, G.lay);
  out('w1-out', [stable ? T.w1o1(nf(e.mod, 3), nf(mu[0], 2), nf(mu[1], 2)) : T.w1o2(nf(e.mod, 3)), T.w1o3(nf(P.v[2][0], 2), nf(P.v[2][1], 2))]);
  table('w1-tb', ['h', ...hh.slice(0, 9).map(k => ({ v: String(k), cls: 'r' }))],
    [['ŷ<sub>1</sub>', ...P.f.slice(0, 9).map(f => num(f[0], 2))], ['ŷ<sub>2</sub>', ...P.f.slice(0, 9).map(f => num(f[1], 2))],
      ['σ<sub>11</sub>(h)', ...P.v.slice(0, 9).map(v => num(v[0], 2))], ['σ<sub>22</sub>(h)', ...P.v.slice(0, 9).map(v => num(v[1], 2))]]);
}

// ================================================================= 3. the worked example, simulated
function simVar(a21, rho) {
  const A = wxA(0.2, a21), P = chol([[1, rho], [rho, 1]]), e = eig2(A), z = D.sim.z;
  if (e.mod >= 1) return null;
  const dI = (1 - A[0][0]) * (1 - A[1][1]) - A[0][1] * A[1][0], mu = [((1 - A[1][1]) * WXC[0] + A[0][1] * WXC[1]) / dI, (A[1][0] * WXC[0] + (1 - A[0][0]) * WXC[1]) / dI];
  const y = [mu.slice()];
  for (let t = 1; t < 300; t++) { const u = [z[2 * (t - 1)], z[2 * (t - 1) + 1]], p = y[t - 1];
    y.push([WXC[0] + A[0][0] * p[0] + A[0][1] * p[1] + P[0][0] * u[0], WXC[1] + A[1][0] * p[0] + A[1][1] * p[1] + P[1][0] * u[0] + P[1][1] * u[1]]); }
  const Y = y.slice(D.sim.burn), y1 = Y.map(r => r[0]), y2 = Y.map(r => r[1]);
  return { A, e, mu, y1, y2, m: [mean(y1), mean(y2)], r: corr(y1, y2), Ph: maRep([A], 10) };
}
function blkSim() {
  const a21 = +$('vs-a').value, rho = +$('vs-r').value, h = +$('vs-h').value; set('vs-a', nf(a21, 2)); set('vs-r', nf(rho, 1)); set('vs-h', h);
  const s = simVar(a21, rho), G = grid('vs-chart', [{ r: 0, c: 0, cs: 1, t: 'Simulated VAR(1), T = 200', xt: 't' }, { r: 0, c: 1, t: 'Impulse responses Phi(h) = A^h', xt: 'horizon h' }], { cols: 2, h: 380, legendRows: 2 });
  const [a0, a1] = G.ax, t = range(0, 199, 1), hh = range(0, 10, 1);
  plot('vs-chart', [ln(t, s.y1, a0, K.blue, 'y1(t)', { line: { color: K.blue, width: 1 }, hovertemplate: 't = %{x}: %{y:.2f}<extra>y1</extra>' }),
    ln(t, s.y2, a0, K.red, 'y2(t)', { line: { color: K.red, width: 1 }, hovertemplate: 't = %{x}: %{y:.2f}<extra>y2</extra>' }),
    ln(hh, s.Ph.map(M => M[0][0]), a1, K.blue, 'y1 after a unit shock in e1', { mode: 'lines+markers', marker: { size: 6 }, hovertemplate: 'h = %{x}: %{y:.3f}<extra>y1</extra>' }),
    ln(hh, s.Ph.map(M => M[1][0]), a1, K.red, 'y2 after a unit shock in e1', { mode: 'lines+markers', marker: { size: 6, symbol: 'square' }, hovertemplate: 'h = %{x}: %{y:.3f}<extra>y2</extra>' }),
    leg('means mu1, mu2', { color: K.navy, dash: 'dash', width: 1 })],
  Object.assign(G.lay, { shapes: [hl(a0, s.mu[0], K.blue, 'dash', 1), hl(a0, s.mu[1], K.red, 'dot', 1.3), hl(a1, 0, K.navy, 'solid', 0.6), vl(a1, h, K.teal, 'dot', 1.2)] }));
  out('vs-out', [T.vs1(nf(a21, 2), nf(rho, 1), nf(s.m[0], 2), nf(s.m[1], 2), nf(s.mu[0], 2), nf(s.mu[1], 2), nf(s.r, 2)),
    T.vs2(nf(s.e.l[0], 3), nf(s.e.l[1], 3), h, nf(s.Ph[h][0][0], 3), nf(s.Ph[h][1][0], 3), nf(s.e.l[0] ** h, 3))]);
  // spectral form of the first column: Phi_h e1 = w1 lambda1^h v1 + w2 lambda2^h v2, with v_k = (a12, lambda_k - a11)'
  const [l1, l2] = s.e.l, A = s.A, v1 = [A[0][1], l1 - A[0][0]], v2 = [A[0][1], l2 - A[0][0]], dv = v1[0] * v2[1] - v2[0] * v1[1];
  const w1 = v2[1] / dv, w2 = -v1[1] / dv, hp = Math.max(h, 1), P0 = s.Ph[hp - 1];
  steps('vs-ws', [T.vss1(hp, pn(P0[0][0], 4), pn(P0[1][0], 4), nf(A[1][0], 2), nf(s.Ph[hp][0][0], 4), nf(s.Ph[hp][1][0], 4)),
    T.vss2(nf(w1 * v1[0], 4), pn(l1, 3), pn(w2 * v2[0], 4), pn(l2, 3), nf(w1 * v1[1], 4), pn(w2 * v2[1], 4)),
    T.vss3(h, nf(w1 * v1[0], 3), nf(l1, 2), nf(l1 ** h, 3), nf(w1 * v1[0] * l1 ** h + w2 * v2[0] * l2 ** h, 3), nf(s.Ph[h][0][0], 3))]);
  table('vs-tb', ['h', ...hh.map(k => ({ v: String(k), cls: 'r' }))], [['Φ<sub>11</sub>(h)', ...s.Ph.map(M => num(M[0][0], 3))], ['Φ<sub>21</sub>(h)', ...s.Ph.map(M => num(M[1][0], 3))],
    ['λ<sub>1</sub><sup>h</sup>', ...hh.map(k => num(l1 ** k, 3))], ['Φ<sub>11</sub>(h)/Φ<sub>11</sub>(h−1)', '', ...hh.slice(1).map(k => num(s.Ph[k][0][0] / s.Ph[k - 1][0][0], 3))]]);
}

// ================================================================= 4. eigenvalues of three VARs
const usFit = (end, p, ord) => varFit(usY(end === undefined ? US_SW : end, ord), p || 4);
const MF = { 0: {} };              // market VAR fits: the full sample, and one other start date (the fits keep their regressors)
function mktFit(p, i0) { i0 = i0 || 0; if (!MF[i0]) { Object.keys(MF).forEach(k => { if (+k) delete MF[k]; }); MF[i0] = {}; }
  return MF[i0][p] || (MF[i0][p] = varFit(mktY(i0, MKT.d.length), p)); }
function blkRoots() {
  const p = +$('ro-p').value, ey = +$('ro-e').value; set('ro-p', p); set('ro-e', ey);
  const end = ey >= 2026 ? ROE : roIdx(`${ey}-10-01`), f = varFit(roY(end), p), ev = eigvals(companion(f));
  const eu = memo('eus', () => eigvals(companion(usFit()))), em = memo('emk', () => eigvals(companion(mktFit(4))));
  const G = grid('ro-chart', [{ r: 0, c: 0, xt: 'real part', yt: 'imaginary part', x: { range: [-1.15, 1.15], scaleanchor: 'y' }, y: { range: [-1.15, 1.15] } }], { cols: 1, h: 470, legendRows: 2 });
  const a = G.ax[0], th = range(0, 400, 1).map(k => 2 * Math.PI * k / 400);
  plot('ro-chart', [ln(th.map(Math.cos), th.map(Math.sin), a, K.navy, 'unit circle', { line: { color: K.navy, width: 1 }, hoverinfo: 'skip' }),
    { x: ev.map(e => e.re), y: ev.map(e => e.im), mode: 'markers', marker: { color: K.red, size: 9 }, name: `Romania VAR(${p}), 2005Q1-${ey >= 2026 ? '2026Q2' : ey + 'Q4'}`, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x:.3f} %{y:+.3f}i<extra>Romania</extra>' },
    { x: eu.map(e => e.re), y: eu.map(e => e.im), mode: 'markers', marker: { color: K.blue, size: 8, symbol: 'square' }, name: 'US VAR(4), 1960-2000', xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x:.3f} %{y:+.3f}i<extra>US</extra>' },
    { x: em.map(e => e.re), y: em.map(e => e.im), mode: 'markers', marker: { color: K.green, size: 9, symbol: 'triangle-up' }, name: 'S&P 500, DAX, BET VAR(4)', xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x:.3f} %{y:+.3f}i<extra>markets</extra>' }],
  Object.assign(G.lay, { shapes: [hl(a, 0, K.navy, 'solid', 0.5), vl(a, 0, K.navy, 'solid', 0.5)] }));
  const hlf = m => (m < 1 ? Math.log(0.5) / Math.log(m) : Infinity);
  out('ro-out', [T.ro1(p, T.q(RO.d[end]), f.n, ev.length, nf(ev[0].mod, 3), ev[0].mod < 1, hlf(ev[0].mod) < Infinity ? nf(hlf(ev[0].mod), 1) : null),
    T.ro2(nf(eu[0].mod, 3), nf(em[0].mod, 3), nf(eu[0].mod ** 20, 2), nf(em[0].mod ** 2, 2))]);
  table('ro-tb', ['', L.re, L.im, L.mod, L.half], ev.map((e, k) => [`λ<sub>${k + 1}</sub>`, num(e.re, 3), num(e.im, 3), num(e.mod, 3, k === 0), num(hlf(e.mod), 1)].map(c => (c.v === 'NaN' || c.v === 'Infinity' ? '' : c))),
    { cap: T.roCap(p, ev.length) });
}

// ================================================================= 5. information criteria
function icTable(Y, pmax) {
  const N = Y.length, Kk = Y[0].length, Tn = N - pmax, out_ = [];
  for (let p = 0; p <= pmax; p++) {
    const X = [], Z = Y.slice(pmax);
    for (let t = pmax; t < N; t++) { const r = [1]; for (let j = 1; j <= p; j++) r.push(...Y[t - j]); X.push(r); }
    const { G, H } = crossp(X, Z), B = mm(inv(G), H), S = zeros(Kk, Kk);
    Z.forEach((z, t) => { const u = z.map((v, k) => v - X[t].reduce((s, x, a) => s + x * B[a][k], 0)); for (let a = 0; a < Kk; a++) for (let b = 0; b < Kk; b++) S[a][b] += u[a] * u[b] / Tn; });
    const ld = Math.log(det(S)), pen = p * Kk * Kk / Tn;
    out_.push({ p, ld, aic: ld + 2 * pen, bic: ld + Math.log(Tn) * pen, hq: ld + 2 * Math.log(Math.log(Tn)) * pen, pen });
  }
  const best = c => out_.reduce((m, r) => (r[c] < out_[m][c] ? r.p : m), 0);
  return { T: Tn, K: Kk, rows: out_, best: { aic: best('aic'), bic: best('bic'), hq: best('hq') } };
}
function blkIc() {
  const pm = +$('ic-m').value, ey = +$('ic-e').value; let pp = +$('ic-p').value; $('ic-p').max = pm; if (pp > pm) { pp = pm; $('ic-p').value = pm; }
  set('ic-m', pm); set('ic-e', ey); set('ic-p', pp);
  const end = ey >= 2026 ? ROE : roIdx(`${ey}-10-01`), I = icTable(roY(end), pm), ps = range(0, pm, 1);
  const G = grid('ic-chart', [{ r: 0, c: 0, xt: 'lag order p', yt: 'information criterion', t: 'Romanian VAR (GDP growth, inflation, ROBOR 3M): lag selection' }], { cols: 1, h: 380, legendRows: 2 });
  const a = G.ax[0], tr_ = [];
  [['aic', K.blue, 'circle', 'AIC'], ['hq', K.green, 'square', 'HQ'], ['bic', K.red, 'triangle-up', 'BIC']].forEach(([c, col, mk, lab]) => {
    const v = I.rows.map(r => r[c]), b = I.best[c];
    tr_.push(ln(ps, v, a, col, `${lab} (minimum at p = ${b})`, { mode: 'lines+markers', marker: { symbol: mk, size: 7 }, hovertemplate: 'p = %{x}: %{y:.3f}<extra>' + lab + '</extra>' }),
      { x: [b], y: [v[b]], mode: 'markers', marker: { symbol: mk + '-open', size: 16, color: col, line: { width: 1.5 } }, showlegend: false, hoverinfo: 'skip', xaxis: a.xa, yaxis: a.ya });
  });
  G.lay.shapes = [vl(a, pp, K.teal, 'dot', 1.2)];
  plot('ic-chart', tr_, G.lay);
  const R = I.rows;
  out('ic-out', [T.ic1(pm, I.T, T.q(RO.d[end]), I.best.aic, I.best.hq, I.best.bic), T.ic2(nf(R[1].aic, 3), nf(R[2].aic, 3), pm >= 5 ? nf(R[5].aic, 3) : null, I.best.aic)]);
  table('ic-tb', ['p', L.ldet, 'AIC', 'HQ', 'BIC', L.ncoef], R.map(r => [{ v: String(r.p), cls: 'r' }, num(r.ld, 3), num(r.aic, 3, r.p === I.best.aic), num(r.hq, 3, r.p === I.best.hq), num(r.bic, 3, r.p === I.best.bic), { v: String(3 + 9 * r.p), cls: 'r' }]),
    { cap: T.icCap(I.T) });
  const r = R[pp], lt = Math.log(I.T);
  steps('ic-ws', [T.ics1(I.T, pm, pp, 3, pp * 9), T.ics2(pp, nf(r.ld, 4)), T.ics3(pp * 9, I.T, nf(r.pen, 4), nf(lt, 3), nf(2 * Math.log(lt), 3)),
    T.ics4(nf(r.ld, 4), nf(2 * r.pen, 4), nf(r.aic, 4), nf(2 * Math.log(lt) * r.pen, 4), nf(r.hq, 4), nf(lt * r.pen, 4), nf(r.bic, 4))]);
}

// ================================================================= table: the estimated Romanian VAR(2)
function blkVt() {
  const p = +$('vt-p').value, e = $('vt-e').value; set('vt-p', p);
  const end = roIdx(e), Y = roY(end), f = varFit(Y, p), U = f.U;
  const r2 = RV.map((_, j) => 1 - pvar(U.map(u => u[j])) / pvar(f.Z.map(z => z[j])));
  const names = [L.const]; for (let j = 1; j <= p; j++) RV.forEach(v => names.push(`${T.sym[v]}<sub>t−${j}</sub>`));
  table('vt-tb', [L.reg, ...RV.map(v => `${L.eq} ${T.sym[v]}<sub>t</sub>`)],
    [...names.map((nm, a) => [nm, ...RV.map((_, i) => { const t = f.B[a][i] / f.se[a][i]; return { v: (Math.abs(t) >= 2 ? '<b>' : '') + `${nf(f.B[a][i], 2)} (${nf(t, 1)})` + (Math.abs(t) >= 2 ? '</b>' : ''), cls: 'r' }; })]),
      ['R²', ...r2.map(v => num(v, 2))]], { cap: T.vtCap(f.n, T.q(RO.d[RO0 + p]), T.q(RO.d[end]), 3 + 9 * p) });
  const sdu = RV.map((_, j) => Math.sqrt(f.S[j][j])), cu = corr(U.map(u => u[1]), U.map(u => u[2]));
  const kg = 1, tg = f.B[kg][2] / f.se[kg][2];
  out('vt-out', [T.vt1(nf(sdu[0], 2), nf(sdu[1], 2), nf(sdu[2], 2), nf(cu, 2)), T.vt2(nf(f.B[kg][2], 2), nf(tg, 1), nf(f.B[2][1], 2), nf(r2[0], 2), f.n, 3 + 9 * p, f.n - 1 - 3 * p)]);
}

// ================================================================= 6. residuals of the Romanian VAR(2)
function portmanteau(U, h, p) {
  const n = U.length, Kk = U[0].length, m = Kk === 0 ? [] : U[0].map((_, k) => mean(U.map(u => u[k]))), Xc = U.map(u => u.map((v, k) => v - m[k]));
  const C = j => { const M = zeros(Kk, Kk); for (let t = j; t < n; t++) for (let a = 0; a < Kk; a++) for (let b = 0; b < Kk; b++) M[a][b] += Xc[t][a] * Xc[t - j][b] / n; return M; };
  const C0i = inv(C(0)); let q = 0; const terms = [];
  for (let j = 1; j <= h; j++) { const Cj = C(j), M = mm(mm(mm(tr(Cj), C0i), Cj), C0i); const tt = sum(M.map((r, i) => r[i])); terms.push(n * n * tt / (n - j)); q += tt / (n - j); }
  q *= n * n; const df = Kk * Kk * (h - p);
  return { q, df, p: chi2sf(q, df), terms };
}
function jbTest(U) {
  const n = U.length, Kk = U[0].length, m = U[0].map((_, k) => mean(U.map(u => u[k]))), Xc = U.map(u => u.map((v, k) => v - m[k]));
  const S = zeros(Kk, Kk); Xc.forEach(u => { for (let a = 0; a < Kk; a++) for (let b = 0; b < Kk; b++) S[a][b] += u[a] * u[b] / n; });
  const Pi = inv(chol(S)), W = Xc.map(u => mv(Pi, u));
  const b1 = [...Array(Kk)].map((_, k) => sum(W.map(w => w[k] ** 3)) / n), b2 = [...Array(Kk)].map((_, k) => sum(W.map(w => w[k] ** 4)) / n - 3);
  const ls = n * sum(b1.map(v => v * v)) / 6, lk = n * sum(b2.map(v => v * v)) / 24;
  return { stat: ls + lk, df: 2 * Kk, p: chi2sf(ls + lk, 2 * Kk), ls, lk };
}
function residFit(p, dum) {
  const Y = roY(), ex = dum ? RO.d.slice(RO0).map(d => [d === '2009-01-01' ? 1 : 0, d === '2020-04-01' ? 1 : 0]) : null;
  return varFit(Y, p, ex);
}
function blkResid() {
  const p = +$('re-p').value, dum = $('re-d').value === 'dum'; let h = +$('re-h').value; if (h <= p) { h = p + 1; $('re-h').value = h; }
  set('re-p', p); set('re-h', h);
  const f = residFit(p, dum), d = RO.d.slice(RO0 + p), U = f.U;
  const G = grid('re-chart', RV.map((v, i) => ({ r: 0, c: i, t: `${LAB[v]} equation`, x: { type: 'date' } })), { cols: 3, h: 330, gx: 0.05, legendRows: 2 });
  const tr_ = [], sds = [];
  RV.forEach((v, i) => { const u = U.map(r => r[i]), s = sd1(u), a = G.ax[i]; sds.push(s);
    tr_.push(bar(d, u, a, COLV[v], `residual, ${LAB[v]} equation`, { hovertemplate: '%{x|%Y-%m}: %{y:.2f}<extra></extra>' }));
    G.lay.shapes.push(hl(a, 2 * s, K.teal, 'dash', 1), hl(a, -2 * s, K.teal, 'dash', 1), hl(a, 0, K.navy, 'solid', 0.6)); });
  tr_.push(leg('+/- 2 standard deviations', { color: K.teal, dash: 'dash', width: 1.2 }));
  plot('re-chart', tr_, G.lay);
  const qh = portmanteau(U, h, p), q12 = portmanteau(U, 12, p), jb = jbTest(U);
  const big = RV.map((_, i) => U.reduce((m, r, t) => (Math.abs(r[i]) > Math.abs(U[m][i]) ? t : m), 0));
  out('re-out', [T.re1(...RV.map((v, i) => [nf(U[big[i]][i], 1), T.q(d[big[i]])]).flat()),
    T.re2(h, nf(qh.q, 1), qh.df, pe(qh.p), nf(q12.q, 1), q12.df, pe(q12.p)), T.re3(nf(jb.stat, 0), jb.df, pe(jb.p), dum)]);
  const sk = RV.map((_, i) => skewK(U.map(r => r[i])));
  table('re-tb', [L.eq, 'σ̂', L.skew, L.kurt, L.big], RV.map((v, i) => [`${T.sym[v]}<sub>t</sub>`, num(Math.sqrt(f.S[i][i]), 2), num(sk[i].s, 2), num(sk[i].k, 1, sk[i].k > 4), { v: `${nf(U[big[i]][i], 1)} (${T.q(d[big[i]])})`, cls: 'r' }]),
    { note: T.reNote(h, nf(qh.q, 1), qh.df, pe(qh.p), nf(q12.q, 1), q12.df, pe(q12.p), nf(jb.stat, 1), pe(jb.p)) });
  steps('re-ws', [T.res1(f.n, 3, h, p, qh.df), T.res2(qh.terms.slice(0, 3).map(x => nf(x, 2)), nf(qh.q, 1)), T.res3(nf(qh.q, 1), qh.df, pv(qh.p), qh.p < 0.05)]);
}

// ================================================================= worked example: the Granger F test by hand
function blkGw() {
  let c = +$('gw-c').value, e = +$('gw-e').value; const p = +$('gw-p').value; set('gw-p', p);
  if (c === e) { e = (c + 2) % 3; $('gw-e').value = e; }
  const f = varFit(roY(), p), G = granger(f, null, c, e), W = p * G.F;
  steps('gw-ws', [T.gws1(T.sym[RV[e]], T.sym[RV[c]], f.n, p, 1 + 3 * p, nf(G.rssU, 2)), T.gws2(T.sym[RV[c]], p, 1 + 2 * p, nf(G.rssR, 2)),
    T.gws3(nf(G.rssR, 2), nf(G.rssU, 2), p, G.df2, nf((G.rssR - G.rssU) / p, 3), nf(G.rssU / G.df2, 4), nf(G.F, 2)),
    T.gws4(p, G.df2, nf(G.crit, 2), pv(G.p), G.p < 0.05), T.gws5(p, nf(G.F, 2), nf(W, 2), pv(chi2sf(W, p))),
    T.gws6(G.p < 0.05, T.v[RV[c]], T.v[RV[e]])]);
  out('gw-out', [T.gw1(T.v[RV[c]], T.v[RV[e]], p, nf(G.F, 2), p, G.df2, pe(G.p))]);
}

// ================================================================= table: Granger causality in the Romanian VAR(2)
const PAIRS = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]];
function grRo(p, sys, end) {
  return PAIRS.map(([c, e]) => {
    if (sys === 'bi') { const Y = roY(end).map(r => [r[c], r[e]]), f = varFit(Y, p); return granger(f, null, 0, 1); }
    return granger(memo(`rf${p}-${end}`, () => varFit(roY(end), p)), null, c, e);
  });
}
function blkGr() {
  const p = +$('gr-p').value, sys = $('gr-s').value, end = roIdx($('gr-e').value); set('gr-p', p);
  const R = grRo(p, sys, end);
  table('gr-tb', [L.cause, L.effect, { v: 'F', cls: 'r' }, { v: L.df, cls: 'r' }, { v: 'p', cls: 'r' }], PAIRS.map(([c, e], k) => [T.v[RV[c]], T.v[RV[e]], num(R[k].F, 2, R[k].p < 0.05),
    { v: `(${R[k].df1}, ${R[k].df2})`, cls: 'r' }, { v: (R[k].p < 0.05 ? '<b>' : '') + pv(R[k].p) + (R[k].p < 0.05 ? '</b>' : ''), cls: 'r' }]), { cap: T.grCap(sys, p, T.q(RO.d[end]), nf(R[0].crit, 2)) });
  const ps = range(1, 6, 1), byp = ps.map(q => grRo(q, sys, end));
  const G = grid('gr-chart', [{ r: 0, c: 0, xt: 'lag order p', yt: 'p-value (log scale)', y: { type: 'log', tickvals: [0.001, 0.01, 0.05, 0.1, 0.5, 1], ticktext: ['0.001', '0.01', '0.05', '0.1', '0.5', '1'] }, t: 'Granger F tests by lag order' }], { cols: 1, h: 380, legendRows: 3 });
  const a = G.ax[0], cols = [K.green, K.blue, K.red, K.amber, K.purple, K.teal];
  plot('gr-chart', PAIRS.map(([c, e], k) => ln(ps, byp.map(r => Math.max(r[k].p, 1e-4)), a, cols[k], `${LAB[RV[c]]} -> ${LAB[RV[e]]}`, { mode: 'lines+markers', marker: { size: 6 }, hovertemplate: 'p = %{x}: %{y:.3f}<extra></extra>' })),
    Object.assign(G.lay, { shapes: [hl(a, 0.05, K.navy, 'dash', 1), vl(a, p, K.teal, 'dot', 1.2)] }));
  const sig = PAIRS.filter((_, k) => R[k].p < 0.05).map(([c, e]) => `${T.v[RV[c]]} → ${T.v[RV[e]]}`);
  out('gr-out', [T.gr1(sig), T.gr2(nf(R[2].F, 2), pe(R[2].p), nf(R[5].F, 2), pe(R[5].p))]);
}

// ================================================================= table: markets, does New York lead?
function grMkt(p, i0) { const f = mktFit(p, i0); return { f, R: PAIRS.map(([c, e]) => granger(f, null, c, e)) }; }
function blkGm() {
  const p = +$('gm-p').value, y0 = +$('gm-y').value; set('gm-p', p); set('gm-y', y0);
  const i0 = mIdx(`${y0}-01-01`), { f, R } = grMkt(p, i0);
  table('gm-tb', [L.cause, L.effect, { v: 'F', cls: 'r' }, { v: L.df, cls: 'r' }, { v: 'p', cls: 'r' }], PAIRS.map(([c, e], k) => [LAB[MK[c]], LAB[MK[e]], num(R[k].F, 1, R[k].p < 0.05),
    { v: `(${R[k].df1}, ${R[k].df2})`, cls: 'r' }, { v: (R[k].p < 0.05 ? '<b>' : '') + pv(R[k].p) + (R[k].p < 0.05 ? '</b>' : ''), cls: 'r' }]), { cap: T.gmCap(p, f.n, y0) });
  const ps = range(1, 10, 1), Fs = ps.map(q => grMkt(q, i0).R);
  const G = grid('gm-chart', [{ r: 0, c: 0, xt: 'lag order p', yt: 'F statistic (log scale)', y: { type: 'log' }, t: 'Granger F tests by lag order, daily returns' }], { cols: 1, h: 380, legendRows: 3 });
  const a = G.ax[0], cols = [K.teal, K.red, K.blue, K.amber, K.purple, K.green];
  plot('gm-chart', [...PAIRS.map(([c, e], k) => ln(ps, Fs.map(r => Math.max(r[k].F, 0.05)), a, cols[k], `${LAB[MK[c]]} -> ${LAB[MK[e]]}`, { mode: 'lines+markers', marker: { size: 6 }, hovertemplate: 'p = %{x}: F = %{y:.2f}<extra></extra>' })),
    ln(ps, Fs.map(r => r[0].crit), a, K.navy, '5% critical value', { line: { color: K.navy, dash: 'dash', width: 1.2 }, hovertemplate: 'p = %{x}: %{y:.2f}<extra>5% critical value</extra>' })],
  Object.assign(G.lay, { shapes: [vl(a, p, K.teal, 'dot', 1.2)] }));
  const cb = f.B[1][2], sb = f.se[1][2], cd = f.B[1][1];
  out('gm-out', [T.gm1(nf(cb, 3), nf(sb, 3), nf(cd, 3), p), T.gm2(nf(R[4].F, 2), pe(R[4].p), nf(R[2].F, 1), pe(R[2].p))]);
}

// ================================================================= 7. spurious Granger causality from an omitted variable
function blkGs() {
  const al = $('gs-a').value, p = $('gs-p').value, c = $('gs-c').value, S = D.gsim, ka = S.al.indexOf(+al), nn = S.n[c][p];
  const G = grid('gs-chart', [{ r: 0, c: 0, yt: 'rejection rate (%)', t: 'rejections of "x does not Granger-cause y"', x: { type: 'category' } }], { cols: 1, h: 360, legendRows: 2 });
  const a = G.ax[0], xs = S.T.map(t => `T = ${t}`), bi = S.T.map(t => 100 * nn[t].bi[ka] / S.R), tri = S.T.map(t => 100 * nn[t].tri[ka] / S.R);
  plot('gs-chart', [bar(xs, bi, a, K.red, 'bivariate VAR (x, y): z omitted', { offsetgroup: 1, hovertemplate: '%{x}: %{y:.1f}%<extra>bivariate</extra>' }),
    bar(xs, tri, a, K.blue, 'trivariate VAR (x, y, z)', { offsetgroup: 2, hovertemplate: '%{x}: %{y:.1f}%<extra>trivariate</extra>' }),
    leg(`nominal size ${+(100 * al).toFixed(0)}%`, { color: K.teal, dash: 'dash', width: 1.2 })],
  Object.assign(G.lay, { barmode: 'group', shapes: [hl(a, 100 * al, K.teal, 'dash', 1.2)], yaxis: Object.assign(G.lay.yaxis, { range: [0, 105] }) }));
  out('gs-out', [T.gs1(nf(+c, 1), p, nf(100 * al, 0), S.T.map((t, k) => [t, nf(bi[k], 1), nf(tri[k], 1)]))]);
  table('gs-tb', ['T', ...S.T.map(t => ({ v: String(t), cls: 'r' }))], [[T.L.bi, ...bi.map(v => num(v, 1))], [T.L.tri, ...tri.map(v => num(v, 1))]], { cap: T.gsCap(nf(+c, 1), p, nf(100 * al, 0), S.R) });
}

// ================================================================= worked example: orthogonalised responses of the VAR(1)
function wxOrth(rho, ord, H) {
  const S = [[1, rho], [rho, 1]], A = wxA(0.2, 0.3);
  const P = ord === 'y2' ? (() => { const Lr = chol([[S[1][1], S[1][0]], [S[0][1], S[0][0]]]); return [[Lr[1][1], Lr[1][0]], [Lr[0][1], Lr[0][0]]]; })() : chol(S);
  const Th = [P]; for (let h = 1; h <= H; h++) Th.push(mm(A, Th[h - 1]));
  return { S, A, P, Th };
}
function blkWo() {
  const rho = +$('wo-r').value, ord = $('wo-o').value, h = +$('wo-h').value; set('wo-r', nf(rho, 1)); set('wo-h', h);
  const s = wxOrth(rho, ord, 8), P = s.P, hp = Math.max(h, 1), Tp = s.Th[hp - 1], Th = s.Th[hp];
  steps('wo-ws', [ord === 'y2' ? T.wos1b(nf(rho, 2), nf(P[0][0], 3), nf(P[0][1], 3)) : T.wos1(nf(rho, 2), nf(P[1][0], 3), nf(P[1][1], 3)),
    T.wos2(mat(P, 3), nf(P[0][0], 3), nf(P[1][0], 3), nf(P[0][1], 3), nf(P[1][1], 3)),
    T.wos3(hp, pn(Tp[0][0], 3), pn(Tp[1][0], 3), nf(Th[0][0], 3), nf(Th[1][0], 3), pn(Tp[0][1], 3), pn(Tp[1][1], 3), nf(Th[0][1], 3), nf(Th[1][1], 3)),
    T.wos4(h, mat(s.Th[h], 3))]);
  const G = grid('wo-chart', [{ r: 0, c: 0, xt: 'horizon h', yt: 'response', t: `Orthogonalised responses, order ${ord === 'y2' ? '(y2, y1)' : '(y1, y2)'}` }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0], hh = range(0, 8, 1);
  plot('wo-chart', [[0, 0, K.blue, 'y1 <- u1', 'circle', 'solid'], [1, 0, K.red, 'y2 <- u1', 'square', 'solid'], [0, 1, K.blue, 'y1 <- u2', 'circle-open', 'dash'], [1, 1, K.red, 'y2 <- u2', 'square-open', 'dash']]
    .map(([i, j, col, nm, mk, ds]) => ln(hh, s.Th.map(M => M[i][j]), a, col, nm, { mode: 'lines+markers', marker: { symbol: mk, size: 7 }, line: { color: col, width: 1.5, dash: ds }, hovertemplate: 'h = %{x}: %{y:.3f}<extra>' + nm + '</extra>' })),
  Object.assign(G.lay, { shapes: [hl(a, 0, K.navy, 'solid', 0.6), vl(a, h, K.teal, 'dot', 1.2)] }));
  out('wo-out', [T.wo1(ord, nf(s.Th[0][1][0], 3), nf(s.Th[1][1][0], 3), nf(s.Th[0][0][1], 3), nf(s.Th[1][0][1], 3))]);
}

// ================================================================= 8. impulse responses of the Romanian VAR(2)
function roIrf(ord, p, H, end) {
  return memo(`irf${ord}${p}${H}${end || ''}`, () => { const o = ORD[ord], f = varFit(roY(end, ord), p), Th = orthIrf(f, H);
    return { f, Th: Th.map(M => reord(M, o)), o }; });
}
function roBoot(ord, H) {      // the chapter's residual bootstrap (500 replications, 90% band), the draws of its seed; VAR(2), full sample
  return memo(`boot${ord}${H}`, () => {
    const o = ORD[ord], Y = roY(undefined, ord), p = 2, f = varFit(Y, p), n = f.U.length, mu = [0, 1, 2].map(k => mean(f.U.map(u => u[k]))), Uc = f.U.map(u => u.map((v, k) => v - mu[k]));
    const draws = [];
    for (let b = 0; b < 500; b++) {
      const y = Y.slice(0, p).map(r => [...r]);
      for (let t = p; t < Y.length; t++) { const e = Uc[BOOT[b * n + t - p]], v = f.c.slice(); for (let j = 1; j <= p; j++) { const z = y[t - j]; for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) v[i] += f.A[j - 1][i][k] * z[k]; } y.push(v.map((x, i) => x + e[i])); }
      draws.push(orthIrf(varFit(y, p), H).map(M => reord(M, o)));
    }
    const q = a => [...Array(H + 1)].map((_, h) => [0, 1, 2].map(i => [0, 1, 2].map(j => quantile(draws.map(d => d[h][i][j]), a))));
    return { lo: q(0.05), hi: q(0.95) };
  });
}
function blkIrf() {
  const ord = $('ir-o').value, p = +$('ir-p').value, pr = $('ir-j').value.split('-').map(Number), h = +$('ir-h').value; set('ir-p', p); set('ir-h', h);
  const H = 12, R = roIrf(ord, p, H), B = p === 2 ? roBoot(ord, H) : null, hh = range(0, H, 1);
  const cells = []; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cells.push({ r: i, c: j, t: `${LAB[RV[i]]} <- ${LAB[RV[j]]}`, xt: i === 2 ? 'quarters' : '' });
  const G = grid('ir-chart', cells, { cols: 3, h: 560, gx: 0.05, gyPx: 46, rowPx: 170, legendRows: 1 }), tr_ = [];
  cells.forEach((c, k) => { const a = G.ax[k], i = c.r, j = c.c;
    if (B) tr_.push(band(hh, hh.map(x => B.lo[x][i][j]), hh.map(x => B.hi[x][i][j]), a, K.teal, 0.25, k === 0 ? '90% bootstrap band' : ''));
    tr_.push(ln(hh, R.Th.map(M => M[i][j]), a, COLV[RV[j]], k === 0 ? 'orthogonalised response' : null, { line: { color: COLV[RV[j]], width: 1.8 }, hovertemplate: 'h = %{x}: %{y:.3f}<extra></extra>' }));
    G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.6)); if (i === pr[0] && j === pr[1]) G.lay.shapes.push(vl(a, h, K.navy, 'dot', 1)); });
  plot('ir-chart', tr_, G.lay);
  const r = R.Th.map(M => M[2][1]), mx = r.reduce((m, v, k) => (v > r[m] ? k : m), 0), sig = B ? hh.filter(x => B.lo[x][2][1] > 0 || B.hi[x][2][1] < 0).length : null;
  const gi = R.Th.map(M => M[0][2]), pii = R.Th.map(M => M[1][2]), pmx = pii.reduce((m, v, k) => (v > pii[m] ? k : m), 0);
  out('ir-out', [T.ir1(ORDN(ord), p, nf(R.Th[0][1][1], 2), nf(r[0], 2), nf(r[mx], 2), mx, nf(r[12], 2), sig),
    T.ir2(nf(gi[1], 2), nf(pii[pmx], 2), pmx, nf(R.Th[4][1][1], 2), nf(R.Th[12][1][1], 2)), ...(B ? [] : [T.ir3(p)])]);
  const [i, j] = pr, sel = [0, 1, 2, 4, 8, 12];
  table('ir-tb', ['h', ...sel.map(x => ({ v: String(x), cls: 'r' }))], [[L.resp, ...sel.map(x => num(R.Th[x][i][j], 3, B && (B.lo[x][i][j] > 0 || B.hi[x][i][j] < 0)))],
    ...(B ? [[L.lo, ...sel.map(x => num(B.lo[x][i][j], 3))], [L.hi, ...sel.map(x => num(B.hi[x][i][j], 3))]] : [])], { cap: T.irCap(T.v[RV[i]], T.vg[RV[j]]) });
  // worked: theta_{ij,h} = sum_m phi_{im,h} p_{mj} in the estimation order
  const o = R.o, oi = o.indexOf(i), oj = o.indexOf(j), Ph = maRep(R.f.A, h)[h], P = chol(R.f.S);
  const terms = [0, 1, 2].map(m => [Ph[oi][m], P[m][oj]]);
  steps('ir-ws', [T.irs1(ORDN(ord), mat(P, 3)), T.irs2(h, T.sym[RV[i]], T.sym[RV[j]], terms.map(([a, b]) => `${pn(a, 3)} × ${pn(b, 3)}`), nf(sum(terms.map(([a, b]) => a * b)), 3)),
    T.irs3(o.map(k => T.sym[RV[k]]), oj, T.sym[RV[j]])]);
}

// ================================================================= 9. the effect of the ordering
function blkIo() {
  const p = +$('io-p').value, end = roIdx($('io-e').value); set('io-p', p);
  const H = 12, Y = roY(end), fa = varFit(Y, p), fb = varFit(roY(end, 'gip'), p), oa = orthIrf(fa, H), ob = orthIrf(fb, H), G0 = girfOf(fa, H), hh = range(0, H, 1);
  const s = [[oa.map(M => M[2][1]), ob.map(M => M[1][2]), G0.map(M => M[2][1])], [oa.map(M => M[1][2]), ob.map(M => M[2][1]), G0.map(M => M[1][2])]];
  const G = grid('io-chart', [{ r: 0, c: 0, t: 'ROBOR 3M <- inflation shock', xt: 'quarters', yt: 'percentage points' }, { r: 0, c: 1, t: 'inflation <- ROBOR 3M shock', xt: 'quarters' }], { cols: 2, h: 360, legendRows: 2 });
  const tr_ = [];
  [0, 1].forEach(k => { const a = G.ax[k];
    tr_.push(ln(hh, s[k][0], a, K.blue, k ? null : 'order g, inflation, ROBOR', { mode: 'lines+markers', marker: { size: 5 }, hovertemplate: 'h = %{x}: %{y:.3f}<extra></extra>' }),
      ln(hh, s[k][1], a, K.red, k ? null : 'order g, ROBOR, inflation', { mode: 'lines+markers', marker: { size: 5, symbol: 'square' }, line: { color: K.red, dash: 'dash', width: 1.5 }, hovertemplate: 'h = %{x}: %{y:.3f}<extra></extra>' }),
      ln(hh, s[k][2], a, K.green, k ? null : 'generalised (order-free)', { mode: 'lines+markers', marker: { size: 5, symbol: 'triangle-up' }, line: { color: K.green, dash: 'dot', width: 1.5 }, hovertemplate: 'h = %{x}: %{y:.3f}<extra></extra>' }));
    G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.6)); });
  plot('io-chart', tr_, G.lay);
  const mx = v => Math.max(...v), cu = corr(fa.U.map(u => u[1]), fa.U.map(u => u[2]));
  out('io-out', [T.io1(p, T.q(RO.d[end]), nf(mx(s[0][0]), 2), nf(mx(s[0][1]), 2), nf(mx(s[0][2]), 2)), T.io2(nf(mx(s[1][0]), 2), nf(mx(s[1][1]), 2), nf(s[1][1][0], 2), nf(cu, 2))]);
  const S = fa.S, Pa = chol(S);
  steps('io-ws', [T.ios1(mat(S, 3)), T.ios2(nf(S[2][1], 3), pn(Pa[2][0], 3), pn(Pa[1][0], 3), nf(Pa[1][1], 3), nf(Pa[2][1], 3)), T.ios3(nf(S[2][1], 3), nf(S[1][1], 3), nf(G0[0][2][1], 3)),
    T.ios4(nf(Pa[2][1], 3), nf(G0[0][2][1], 3))]);
}

// ================================================================= 10. markets: responses to an S&P 500 shock
function mktIrf(p, i0, H) {
  const fa = mktFit(p, i0), fb = varFit(mktY(i0, MKT.d.length, [2, 1, 0]), p);
  return { fa, G: girfOf(fa, H), A: orthIrf(fa, H), B: orthIrf(fb, H) };
}
function blkMi() {
  const p = +$('mi-p').value, y0 = +$('mi-y').value; set('mi-p', p); set('mi-y', y0);
  const i0 = mIdx(`${y0}-01-01`), H = 5, M = mktIrf(p, i0, H), hh = range(0, H, 1);
  const S = [{ G: M.G.map(m => m[1][0]), A: M.A.map(m => m[1][0]), B: M.B.map(m => m[1][2]) }, { G: M.G.map(m => m[2][0]), A: M.A.map(m => m[2][0]), B: M.B.map(m => m[0][2]) }];
  const G = grid('mi-chart', [{ r: 0, c: 0, t: 'DAX <- S&P 500 shock', xt: 'days', yt: '% return' }, { r: 0, c: 1, t: 'BET <- S&P 500 shock', xt: 'days' }], { cols: 2, h: 360, legendRows: 2 });
  const tr_ = [];
  S.forEach((s, k) => { const a = G.ax[k];
    tr_.push(bar(hh.map(x => x - 0.25), s.G, a, K.green, k ? null : 'generalised', { width: 0.25, hovertemplate: 'day %{x:.0f}: %{y:.3f}<extra>generalised</extra>' }),
      bar(hh, s.A, a, K.blue, k ? null : 'Cholesky, S&P 500 first', { width: 0.25, hovertemplate: 'day %{x}: %{y:.3f}<extra>S&P 500 first</extra>' }),
      bar(hh.map(x => x + 0.25), s.B, a, K.red, k ? null : 'Cholesky, S&P 500 last (closing-time order)', { width: 0.25, hovertemplate: 'day %{x:.0f}: %{y:.3f}<extra>S&P 500 last</extra>' }));
    G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.6)); });
  G.lay.barmode = 'overlay';
  plot('mi-chart', tr_, G.lay);
  const Sg = M.fa.S, sd = Sg.map((r, i) => Math.sqrt(r[i])), cu = Sg[1][0] / (sd[0] * sd[1]);
  out('mi-out', [T.mi1(p, y0, nf(sd[0], 2), nf(S[0].G[0], 2), nf(S[0].G[1], 2), nf(S[1].G[0], 2), nf(S[1].G[1], 2)), T.mi2(nf(S[0].B[1], 2), nf(S[1].B[1], 2), nf(cu, 2))]);
  table('mi-tb', ['', L.day, ...hh.map(x => ({ v: String(x), cls: 'r' }))], [['DAX', L.gen, ...S[0].G.map(v => num(v, 3))], ['DAX', L.chf, ...S[0].A.map(v => num(v, 3))], ['DAX', L.chl, ...S[0].B.map(v => num(v, 3))],
    ['BET', L.gen, ...S[1].G.map(v => num(v, 3))], ['BET', L.chf, ...S[1].A.map(v => num(v, 3))], ['BET', L.chl, ...S[1].B.map(v => num(v, 3))]]);
  steps('mi-ws', [T.mis1(nf(Sg[0][0], 4), nf(Sg[1][0], 4), nf(Sg[2][0], 4)), T.mis2(nf(Sg[1][0], 4), nf(sd[0], 4), nf(S[0].G[0], 3), nf(Sg[2][0], 4), nf(S[1].G[0], 3)), T.mis3(nf(cu, 3), nf(sd[1], 3), nf(cu * sd[1], 3))]);
}

// ================================================================= worked example: the FEVD of the VAR(1)
function blkWf() {
  const rho = +$('wf-r').value, ord = $('wf-o').value, h = +$('wf-h').value, j = +$('wf-v').value; set('wf-r', nf(rho, 1)); set('wf-h', h);
  const s = wxOrth(rho, ord, 10), Th = s.Th, fe = fevdOf(Th);
  const r1 = Th.slice(0, h).map(M => M[j][0] ** 2), r2 = Th.slice(0, h).map(M => M[j][1] ** 2), mse = sum(r1) + sum(r2);
  steps('wf-ws', [T.wfs1(h, j + 1, Th.slice(0, h).map((M, k) => `θ<sub>${j + 1}1,${k}</sub> = ${nf(M[j][0], 3)}, θ<sub>${j + 1}2,${k}</sub> = ${nf(M[j][1], 3)}`)),
    T.wfs2(j + 1, r1.map(v => nf(v, 3)), nf(sum(r1), 3), r2.map(v => nf(v, 3)), nf(sum(r2), 3), nf(mse, 3)), T.wfs3(j + 1, h, nf(sum(r1), 3), nf(mse, 3), nf(100 * sum(r1) / mse, 1), nf(100 * sum(r2) / mse, 1))]);
  const hs = range(1, 10, 1), G = grid('wf-chart', [{ r: 0, c: 0, xt: 'horizon h', yt: 'share (%)', y: { range: [0, 100] }, t: `FEVD of y${j + 1}, order ${ord === 'y2' ? '(y2, y1)' : '(y1, y2)'}` }], { cols: 1, h: 330, legendRows: 1 });
  const a = G.ax[0];
  plot('wf-chart', [bar(hs, hs.map(x => 100 * fe[j][x - 1][0]), a, K.blue, 'u1 shock', { hovertemplate: 'h = %{x}: %{y:.1f}%<extra>u1</extra>' }), bar(hs, hs.map(x => 100 * fe[j][x - 1][1]), a, K.red, 'u2 shock', { hovertemplate: 'h = %{x}: %{y:.1f}%<extra>u2</extra>' })],
    Object.assign(G.lay, { barmode: 'stack', shapes: [vl(a, h, K.navy, 'dot', 1.2)] }));
  out('wf-out', [T.wf1(j + 1, h, nf(mse, 2), nf(100 * fe[j][h - 1][0], 1), nf(100 * fe[j][9][0], 1))]);
}

// ================================================================= 11. FEVD of the Romanian VAR(2)
function blkFevd() {
  const ord = $('fv-o').value, p = +$('fv-p').value, h = +$('fv-h').value; set('fv-p', p); set('fv-h', h);
  const H = 12, R = roIrf(ord, p, H - 1), fe = fevdOf(R.Th), hs = range(1, H, 1);
  const G = grid('fv-chart', RV.map((v, i) => ({ r: 0, c: i, t: LAB[v], xt: 'horizon (quarters)', yt: i ? '' : 'share (%)', y: { range: [0, 100] } })), { cols: 3, h: 340, gx: 0.04, legendRows: 1 });
  const tr_ = [];
  RV.forEach((v, i) => RV.forEach((s, j) => tr_.push(bar(hs, hs.map(x => 100 * fe[i][x - 1][j]), G.ax[i], COLV[s], i ? null : `${LAB[s]} shock`, { hovertemplate: 'h = %{x}: %{y:.1f}%<extra>' + LAB[s] + '</extra>' }))));
  G.ax.forEach(a => G.lay.shapes.push(vl(a, h, K.navy, 'dot', 1)));
  G.lay.barmode = 'stack';
  plot('fv-chart', tr_, G.lay);
  out('fv-out', [T.fv1(ORDN(ord), p, nf(100 * fe[2][0][2], 0), nf(100 * fe[2][11][2], 0), nf(100 * fe[2][11][1], 0), nf(100 * fe[2][11][0], 0)),
    T.fv2(nf(100 * fe[1][11][1], 0), nf(100 * fe[0][11][0], 0))]);
  const hsel = [1, 4, 8, 12];
  table('fv-tb', [L.var, 'h', ...RV.map(s => ({ v: `${T.v[s]}`, cls: 'r' }))], RV.flatMap((v, i) => hsel.map(x => [x === 1 ? T.v[v] : '', { v: String(x), cls: 'r' }, ...RV.map((_, j) => num(100 * fe[i][x - 1][j], 1, x === h))])), { cap: T.fvCap(ORDN(ord)) });
  const Th = R.Th.slice(0, h), num_ = sum(Th.map(M => M[2][1] ** 2)), den = sum(Th.map(M => sum(M[2].map(v => v * v))));
  steps('fv-ws', [T.fvs1(h, Th.slice(0, 4).map(M => nf(M[2][1], 3)), h > 4, nf(num_, 3)), T.fvs2(h, nf(den, 3)), T.fvs3(h, nf(num_, 3), nf(den, 3), nf(100 * num_ / den, 1))]);
}

// ================================================================= table: the spillover table
function spillTab(th) {
  const Kk = th.length, from = th.map((r, i) => 100 * (sum(r) - r[i])), to = th[0].map((_, j) => 100 * (sum(th.map(r => r[j])) - th[j][j]));
  return { th, from, to, S: spillIdx(th), net: to.map((v, j) => v - from[j]), K: Kk };
}
function spillTable(id, s, cap) {
  table(id, [L.tofrom, ...MK.map(k => ({ v: LAB[k], cls: 'r' })), { v: L.fromo, cls: 'r' }],
    [...MK.map((k, i) => [LAB[k], ...s.th[i].map((v, j) => num(100 * v, 1, i === j)), num(s.from[i], 1)]), [`<b>${L.too}</b>`, ...s.to.map(v => num(v, 1)), { v: `${L.index} <b>${nf(s.S, 1)}</b>`, cls: 'r' }],
      [L.net, ...s.net.map(v => num(v, 1)), '']], { cap });
}
function blkSt() {
  const H = +$('st-h').value, p = +$('st-p').value, y = $('st-y').value; set('st-h', H); set('st-p', p);
  const [i0, i1] = y === 'all' ? [0, MKT.d.length] : [mIdx(`${y}-01-01`), mIdx(`${+y + 1}-01-01`)];
  const f = y === 'all' ? mktFit(p) : varFit(mktY(i0, i1), p), s = spillTab(gfevd(f, H));
  spillTable('st-tb', s, T.stCap(H, p, y === 'all' ? null : y, f.n));
  steps('st-ws', [T.sts1(H, p), T.sts2(s.from.map(v => nf(v, 1)), nf(sum(s.from), 1), nf(s.S, 1)), T.sts3(MK.map((k, j) => [LAB[k], nf(s.to[j], 1), nf(s.from[j], 1), nf(s.net[j], 1)]))]);
  const G = grid('st-chart', [{ r: 0, c: 0, yt: '% of forecast error variance', t: 'Directional spillovers, ' + (y === 'all' ? '2000-2026' : y), x: { type: 'category' } }], { cols: 1, h: 340, legendRows: 1 });
  const a = G.ax[0], xs = MK.map(k => LAB[k]);
  plot('st-chart', [bar(xs, s.to, a, K.blue, 'to others', { offsetgroup: 1, hovertemplate: '%{x}: %{y:.1f}<extra>to others</extra>' }), bar(xs, s.from, a, K.amber, 'from others', { offsetgroup: 2, hovertemplate: '%{x}: %{y:.1f}<extra>from others</extra>' }),
    bar(xs, s.net, a, K.red, 'net (to - from)', { offsetgroup: 3, hovertemplate: '%{x}: %{y:.1f}<extra>net</extra>' })], Object.assign(G.lay, { barmode: 'group', shapes: [hl(a, 0, K.navy, 'solid', 0.6)] }));
  out('st-out', [T.st1(nf(s.S, 1), nf(100 * s.th[0][1], 1), nf(100 * s.th[1][0], 1), nf(s.from[2], 1), nf(s.to[2], 1))]);
}

// ================================================================= 12. the spillover index over time
function rollSpill(W, H, step) {
  if (W === 200 && H === 10) return { e: D.mkt.roll.map((_, k) => D.mkt.roll_i0 + k), s: D.mkt.roll, step: 1 };
  return memo(`rs${W}-${H}`, () => { const e = [], s = [];
    for (let k = W; k <= MKT.d.length; k += step) { e.push(k - 1); s.push(spillIdx(gfevd(varFit(mktY(k - W, k), 4), H))); }
    return { e, s, step }; });
}
function blkSp() {
  const W = +$('sp-w').value, H = +$('sp-h').value; set('sp-w', W); set('sp-h', H);
  const R = rollSpill(W, H, 5), d = R.e.map(k => MKT.d[k]), sl = $('sp-d');
  sl.max = R.s.length - 1; if (+sl.value > R.s.length - 1 || sl.dataset.w !== `${W}-${H}`) { const t = d.indexOf(d.find(x => x >= (sl.dataset.day || '2020-03-17')) || d[d.length - 1]); sl.value = t; sl.dataset.w = `${W}-${H}`; }
  const k = +sl.value, day = d[k]; sl.dataset.day = day; set('sp-d', T.dmy(day));
  const G = grid('sp-chart', [{ r: 0, c: 0, x: { type: 'date' }, t: 'Diebold-Yilmaz spillover index of S&P 500, DAX and BET returns' }], { cols: 1, h: 360, legendRows: 2 });
  const a = G.ax[0], full = spillIdx(gfevd(mktFit(4), H)), mx = Math.max(...R.s);
  G.lay.shapes = [hl(a, full, K.red, 'dash', 1), vl(a, day, K.teal, 'solid', 1.2), ...[['2008-09-15'], ['2011-08-05'], ['2020-03-11'], ['2022-02-24']].map(([x]) => vl(a, x, K.amber, 'dot', 0.9))];
  G.lay.annotations.push(...[['2008-09-15', 'Lehman', 'right'], ['2011-08-05', 'US downgrade', 'left'], ['2020-03-11', 'COVID-19', 'right'], ['2022-02-24', 'Ukraine', 'left']].map(([x, t, al]) => ({ x, y: mx * 1.04, xref: a.xa, yref: a.ya, text: t, showarrow: false, xanchor: al, font: { size: 10, color: K.navy } })));
  G.lay.yaxis = Object.assign(G.lay.yaxis, { range: [0, mx * 1.12] });
  plot('sp-chart', [ln(d, R.s, a, K.blue, `total spillover index, ${W}-day windows (%)`, { line: { color: K.blue, width: 1 }, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }), leg('full sample (%)', { color: K.red, dash: 'dash', width: 1.2 })], G.lay);
  const im = R.s.indexOf(mx), imn = R.s.indexOf(Math.min(...R.s)), avg = (a0, b0) => { const v = R.s.filter((_, j) => d[j] >= a0 && d[j] <= b0); return v.length ? mean(v) : NaN; };
  out('sp-out', [T.sp1(W, H, R.step, nf(mean(R.s), 1), nf(mx, 1), T.dmy(d[im]), nf(R.s[imn], 1), T.dmy(d[imn]), nf(R.s[R.s.length - 1], 1), nf(full, 1)),
    T.sp2(nf(avg('2007-01-01', '2007-12-31'), 0), nf(avg('2008-10-01', '2008-12-31'), 0), nf(avg('2020-03-01', '2020-06-30'), 0))]);
  const e = R.e[k], s = spillTab(gfevd(varFit(mktY(e + 1 - W, e + 1), 4), H));
  spillTable('sp-tb', s, T.spCap(W, T.dmy(MKT.d[e + 1 - W]), T.dmy(day), H, nf(s.S, 1)));
}

// ================================================================= 13. forecasts for two years
function blkFc() {
  const p = +$('fc-p').value, o = $('fc-o'), oi = +o.value, j = +$('fc-v').value; set('fc-p', p); set('fc-o', T.q(RO.d[oi]));
  const Y = roY(oi), f = varFit(Y, p), H = 8, F = forecast(f, Y, H), mu = meanOf(f), fd = range(1, H, 1).map(h => RO.d[oi + h] || qiso(2000, oi + h));
  const h0 = Math.max(RO0, oi - 29), G = grid('fc-chart', RV.map((v, i) => ({ r: 0, c: i, t: LAB[v], x: { type: 'date' } })), { cols: 3, h: 330, gx: 0.05, legendRows: 2 }), tr_ = [];
  RV.forEach((v, i) => { const a = G.ax[i], hd = RO.d.slice(h0, oi + 1), hv = RO[v].slice(h0, oi + 1);
    tr_.push(band(fd, F.pt.map((x, h) => x[i] - Z975 * F.sd[h][i]), F.pt.map((x, h) => x[i] + Z975 * F.sd[h][i]), a, K.teal, 0.25, i ? '' : '95% interval'),
      ln(hd, hv, a, COLV[v], i ? null : 'data', { hovertemplate: '%{x|%Y-%m}: %{y:.2f}<extra></extra>' }),
      ln([RO.d[oi], ...fd], [RO[v][oi], ...F.pt.map(x => x[i])], a, K.navy, i ? null : `VAR(${p}) forecast`, { line: { color: K.navy, dash: 'dash', width: 1.5 }, hovertemplate: '%{x|%Y-%m}: %{y:.2f}<extra></extra>' }));
    if (oi < ROE) tr_.push(ln(RO.d.slice(oi, Math.min(ROE, oi + H) + 1), RO[v].slice(oi, Math.min(ROE, oi + H) + 1), a, K.amber, i ? null : 'realised (after the origin)', { line: { color: K.amber, width: 2 }, hovertemplate: '%{x|%Y-%m}: %{y:.2f}<extra>realised</extra>' }));
  });
  plot('fc-chart', tr_, G.lay);
  const k4 = 3, k8 = 7, lab = h => T.q(fd[h]);
  out('fc-out', [T.fc1(T.q(RO.d[oi]), p, nf(RO.pi[oi], 1), nf(F.pt[k4][1], 1), nf(F.pt[k8][1], 1), nf(F.pt[k8][1] - Z975 * F.sd[k8][1], 1), nf(F.pt[k8][1] + Z975 * F.sd[k8][1], 1)),
    T.fc2(nf(F.pt[k4][2], 1), nf(F.pt[k8][2], 1), nf(F.pt[k4][0], 1), nf(F.pt[k4][0] - Z975 * F.sd[k4][0], 1), nf(F.pt[k4][0] + Z975 * F.sd[k4][0], 1), nf(mu[1], 1), nf(mu[2], 1)),
    ...(oi + H <= ROE ? [T.fc3(lab(k8), nf(RO.pi[oi + H], 1), nf(F.pt[k8][1], 1), RO.pi[oi + H] >= F.pt[k8][1] - Z975 * F.sd[k8][1] && RO.pi[oi + H] <= F.pt[k8][1] + Z975 * F.sd[k8][1])] : [])]);
  table('fc-tb', [L.q, ...RV.map(v => ({ v: T.v[v], cls: 'r' }))], F.pt.map((x, h) => [lab(h), ...RV.map((_, i) => ({ v: `${nf(x[i], 2)} [${nf(x[i] - Z975 * F.sd[h][i], 1)}; ${nf(x[i] + Z975 * F.sd[h][i], 1)}]`, cls: 'r' }))]), { cap: T.fcCap(p, T.q(RO.d[oi])) });
  const yT = Y[Y.length - 1], yT1 = Y[Y.length - 2], parts = [`${nf(f.c[j], 3)}`];
  RV.forEach((_, k) => parts.push(`${sgn(f.A[0][j][k], 3).trim()}·${pn(yT[k], 2)}`));
  if (p >= 2) RV.forEach((_, k) => parts.push(`${sgn(f.A[1][j][k], 3).trim()}·${pn(yT1[k], 2)}`));
  steps('fc-ws', [T.fcs1(T.sym[RV[j]], T.q(RO.d[oi]), vec(yT, 2), p >= 2 ? vec(yT1, 2) : null), T.fcs2(T.sym[RV[j]], parts.join(' '), p > 2, nf(F.pt[0][j], 3)),
    T.fcs3(T.sym[RV[j]], nf(F.sd[0][j] ** 2, 3), nf(F.pt[0][j], 2), nf(Z975 * F.sd[0][j], 2), nf(F.sd[H - 1][j] ** 2, 3), H)]);
}

// ================================================================= 14. VAR against univariate benchmarks
function dmTest(e1, e2, h) { const dl = e1.map((v, i) => v * v - e2[i] * e2[i]), n = dl.length, m = mean(dl);
  const gk = k => { let s = 0; for (let t = k; t < n; t++) s += (dl[t] - m) * (dl[t - k] - m); return s / n; };
  let v = gk(0); for (let k = 1; k < h; k++) v += 2 * gk(k); v /= n; const z = m / Math.sqrt(v); return { z, p: 2 * (1 - normCdf(Math.abs(z))) }; }
function oosStats(cty, h, y0) {
  const O = D.oos[cty], tg = O.target[h], keep = tg.map(q => +q.slice(0, 4) >= y0), V = cty === 'ro' ? RV : ['pi', 'u', 'R'];
  const e = (m, v) => O.e[m][v][h].filter((_, k) => keep[k]), rm = x => Math.sqrt(mean(x.map(u => u * u)));
  return V.map(v => { const ev = e('VAR', v), ea = e('AR', v), er = e('RW', v); return { v, n: ev.length, var: rm(ev), ar: rm(ea), rw: rm(er), dm: dmTest(ev, ea, +h), first: tg.find((q, k) => keep[k]) }; });
}
function blkOos() {
  const sel = $('oo-c').value, [cty, h] = sel.split('-'), ys = $('oo-y'), lo = cty === 'ro' ? 2015 : 1985, hiY = cty === 'ro' ? 2022 : 1996;
  if (+ys.min !== lo) { ys.min = lo; ys.max = hiY; ys.value = lo; }
  const y0 = +ys.value; set('oo-y', y0);
  const ro = [1, 4].map(x => oosStats('ro', String(x), cty === 'ro' ? y0 : 2015)), us = [2, 4, 8].map(x => oosStats('us', String(x), cty === 'us' ? y0 : 1985));
  const G = grid('oo-chart', [{ r: 0, c: 0, t: 'Romania, forecasts for 2015-2026', yt: 'relative RMSE', x: { type: 'category' } }, { r: 0, c: 1, t: 'United States, forecasts for 1985-2000', x: { type: 'category' } }], { cols: 2, h: 350, legendRows: 2 });
  const tr_ = [];
  [[ro, [1, 4], G.ax[0]], [us, [2, 4, 8], G.ax[1]]].forEach(([S, hs, a], k) => {
    const xs = [], ra = [], rv = []; S[0].forEach((_, vi) => hs.forEach((hh, hi) => { const s = S[hi][vi]; xs.push(`${s.v === 'pi' ? 'π' : s.v} h=${hh}`); ra.push(s.ar / s.rw); rv.push(s.var / s.rw); }));
    tr_.push(bar(xs, ra, a, K.amber, k ? null : 'AR / random walk', { offsetgroup: 1, hovertemplate: '%{x}: %{y:.2f}<extra>AR</extra>' }), bar(xs, rv, a, K.blue, k ? null : 'VAR / random walk', { offsetgroup: 2, hovertemplate: '%{x}: %{y:.2f}<extra>VAR</extra>' }));
    G.lay.shapes.push(hl(a, 1, K.red, 'dash', 1.2)); });
  tr_.push(leg('random walk = 1', { color: K.red, dash: 'dash', width: 1.2 }));
  G.lay.barmode = 'group';
  plot('oo-chart', tr_, G.lay);
  const S = oosStats(cty, h, y0), sym = v => T.sym[v] || v;
  table('oo-tb', [L.var, { v: 'n', cls: 'r' }, { v: 'RMSE VAR', cls: 'r' }, { v: 'RMSE AR', cls: 'r' }, { v: L.rmseRw, cls: 'r' }, { v: 'VAR / RW', cls: 'r' }, { v: 'AR / RW', cls: 'r' }, { v: 'DM', cls: 'r' }, { v: 'p', cls: 'r' }],
    S.map(s => [sym(s.v), { v: String(s.n), cls: 'r' }, num(s.var, 2), num(s.ar, 2), num(s.rw, 2), num(s.var / s.rw, 2, s.var < s.ar), num(s.ar / s.rw, 2, s.ar < s.var), num(s.dm.z, 2), { v: pv(s.dm.p), cls: 'r' }]),
    { cap: T.ooCap(cty, h, T.q(qiso(+S[0].first.slice(0, 4), (+S[0].first.slice(5)) - 1))) });
  const r1 = ro[0], u8 = us[2];
  out('oo-out', [T.oo1(nf(r1[0].var / r1[0].rw, 2), nf(r1[0].ar / r1[0].rw, 2), nf(r1[0].dm.z, 2), pe(r1[0].dm.p), r1[0].n),
    T.oo2(nf(u8[1].var / u8[1].rw, 2), nf(u8[1].ar / u8[1].rw, 2), nf(u8[2].var / u8[2].rw, 2), nf(u8[2].ar / u8[2].rw, 2), nf(u8[0].var / u8[0].rw, 2), nf(u8[0].ar / u8[0].rw, 2))]);
}

// ================================================================= 15. the three US series
function blkUs() {
  const y0 = +$('us-y0').value, y1 = Math.max(+$('us-y1').value, y0 + 5); $('us-y1').value = y1; set('us-y0', y0); set('us-y1', y1);
  const a = `${y0}-01-01`, b = `${y1}-12-31`;
  const G = grid('us-chart', [{ r: 0, c: 0, t: 'United States, quarterly, FRED', x: { type: 'date' } }], { cols: 1, h: 380, legendRows: 2 }), ax = G.ax[0];
  G.lay.shapes = [rect(ax, a, b > '2026-06-30' ? '2026-06-30' : b, K.teal, 0.1), hl(ax, 0, K.navy, 'solid', 0.6)];
  plot('us-chart', [ln(US.d, US.pi, ax, K.red, 'inflation (GDP price index, annualised %)', { line: { color: K.red, width: 1.1 }, hovertemplate: '%{x|%Y-%m}: %{y:.2f}<extra></extra>' }),
    ln(US.d, US.u, ax, K.purple, 'unemployment rate (%)', { line: { color: K.purple, width: 1.1 }, hovertemplate: '%{x|%Y-%m}: %{y:.2f}<extra></extra>' }),
    ln(US.d, US.R, ax, K.blue, 'federal funds rate (%)', { line: { color: K.blue, width: 1.1 }, hovertemplate: '%{x|%Y-%m}: %{y:.2f}<extra></extra>' }),
    leg(`selected window ${y0}-${Math.min(y1, 2026)}`, null, { fill: `rgba(${RGB[K.teal]},0.3)` })], G.lay);
  const ix = US.d.map((_, k) => k).filter(k => US.d[k] >= a && US.d[k] <= b), V = ['pi', 'u', 'R'], X = V.map(v => ix.map(k => US[v][k]));
  out('us-out', [T.us1(T.q(US.d[ix[0]]), T.q(US.d[ix[ix.length - 1]]), ix.length, nf(mean(X[0]), 2), nf(mean(X[1]), 2), nf(mean(X[2]), 2), nf(corr(X[0], X[2]), 2), nf(corr(X[1], X[2]), 2))]);
  table('us-tb', [L.var, L.mean, L.sd, 'min', 'max'], V.map((v, i) => [T.v[v], num(mean(X[i]), 2), num(sd1(X[i]), 2), num(Math.min(...X[i]), 2), num(Math.max(...X[i]), 2)]), { cap: T.usCap(T.q(US.d[ix[0]]), T.q(US.d[ix[ix.length - 1]]), ix.length) });
}

// ================================================================= 16. the Stock-Watson VAR
function blkSw() {
  const end = US.d.indexOf($('sw-e').value), p = +$('sw-p').value; set('sw-p', p);
  const H = 24, Y = usY(end), f = varFit(Y, p), Th = orthIrf(f, H), lec = end === US_SW && p === 4, hh = range(0, H, 1), V = ['pi', 'u', 'R'];
  const cells = []; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cells.push({ r: i, c: j, t: `${LAB[V[i]]} <- ${LAB[V[j]]}`, xt: i === 2 ? 'quarters' : '' });
  const G = grid('sw-chart', cells, { cols: 3, h: 560, gx: 0.05, gyPx: 46, rowPx: 170 }), tr_ = [];
  cells.forEach((c, k) => { const a = G.ax[k], i = c.r, j = c.c;
    if (lec) tr_.push(band(hh, hh.map(x => D.us.lo[x][i][j]), hh.map(x => D.us.hi[x][i][j]), a, K.teal, 0.25, k === 0 ? '90% bootstrap band' : ''));
    tr_.push(ln(hh, Th.map(M => M[i][j]), a, COLV[V[j]], k === 0 ? 'orthogonalised response' : null, { line: { color: COLV[V[j]], width: 1.8 }, hovertemplate: 'h = %{x}: %{y:.3f}<extra></extra>' }));
    G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.6)); });
  plot('sw-chart', tr_, G.lay);
  const uR = Th.map(M => M[1][2]), pR = Th.map(M => M[0][2]), um = uR.reduce((m, v, k) => (v > uR[m] ? k : m), 0), pm = pR.reduce((m, v, k) => (v > pR[m] ? k : m), 0);
  const fe = fevdOf(Th.slice(0, 12)), gr = PAIRS.map(([c, e]) => granger(f, null, c, e)), ev = eigvals(companion(f));
  out('sw-out', [T.sw1(T.q(US.d[end]), p, f.n, nf(Th[0][2][2], 2), nf(uR[um], 2), um, nf(pR[pm], 2), pm, nf(pR[12], 2), nf(uR[2], 2)),
    T.sw2(nf(100 * fe[1][11][2], 0), nf(100 * fe[0][11][2], 0), nf(ev[0].mod, 3), lec)]);
  table('sw-tb', [L.cause, L.effect, { v: 'F', cls: 'r' }, { v: 'p', cls: 'r' }], PAIRS.map(([c, e], k) => [T.v[V[c]], T.v[V[e]], num(gr[k].F, 2, gr[k].p < 0.05), { v: (gr[k].p < 0.05 ? '<b>' : '') + pv(gr[k].p) + (gr[k].p < 0.05 ? '</b>' : ''), cls: 'r' }]),
    { cap: T.swCap(p, T.q(US.d[end]), f.n), note: T.swNote(V.map((v, i) => [T.v[v], nf(100 * fe[i][11][0], 0), nf(100 * fe[i][11][1], 0), nf(100 * fe[i][11][2], 0)])) });
}

// ================================================================= wiring
const BLK = { 'ro-macro': ['rm', blkRoMacro], ccf: ['cc', blkCcf], 'wx-var1': ['w1', blkWx1], 'var-sim': ['vs', blkSim], roots: ['ro', blkRoots], ic: ['ic', blkIc], 'var-table': ['vt', blkVt],
  resid: ['re', blkResid], 'granger-wx': ['gw', blkGw], 'granger-ro': ['gr', blkGr], 'granger-mkt': ['gm', blkGm], 'granger-sim': ['gs', blkGs], 'wx-orth': ['wo', blkWo], 'irf-ro': ['ir', blkIrf],
  'irf-order': ['io', blkIo], 'market-irf': ['mi', blkMi], 'wx-fevd': ['wf', blkWf], 'fevd-ro': ['fv', blkFevd], 'spill-table': ['st', blkSt], spillover: ['sp', blkSp], 'forecast-ro': ['fc', blkFc],
  oos: ['oo', blkOos], 'us-data': ['us', blkUs], 'sw-irf': ['sw', blkSw] };
const PLAY = { rm: ['rm-y0', range(2000, 2020, 1), 600], cc: ['cc-y0', range(2000, 2020, 2), 900], w1: ['w1-a21', range(0, 1.2, 0.1), 700], vs: ['vs-a', range(0, 1.2, 0.1), 700],
  ro: ['ro-p', range(1, 6, 1), 1000], ic: ['ic-e', range(2012, 2026, 1), 700], vt: ['vt-p', range(1, 4, 1), 1200], re: ['re-p', range(1, 4, 1), 1100], gw: ['gw-p', range(1, 4, 1), 1100],
  gr: ['gr-p', range(1, 6, 1), 1000], gm: ['gm-p', range(1, 10, 1), 900], gs: ['gs-c', ['0.0', '0.4', '0.8'], 1300], wo: ['wo-r', range(-0.9, 0.9, 0.1), 500], ir: ['ir-o', ['gip', 'gpi', 'igp', 'ipg', 'pgi', 'pig'], 1500],
  io: ['io-p', range(1, 4, 1), 1200], mi: ['mi-y', range(2000, 2020, 2), 1000], wf: ['wf-h', range(1, 10, 1), 700], fv: ['fv-o', ['gip', 'gpi', 'igp', 'ipg', 'pgi', 'pig'], 1400], st: ['st-h', range(1, 20, 1), 600],
  sp: ['sp-w', range(100, 500, 50), 1500], fc: ['fc-o', null, 500], oo: ['oo-c', ['ro-1', 'ro-4', 'us-2', 'us-4', 'us-8'], 1400], us: ['us-y1', range(1980, 2026, 2), 500], sw: ['sw-e', ['2000-10-01', '2007-10-01', '2019-10-01', '2026-04-01'], 1600] };
function init() {
  const live = Object.keys(BLK).filter(b => has(BLK[b][0] + '-out'));
  const players = {};
  live.forEach(b => { const [x, f] = BLK[b], blk = $(b) || $(x + '-out').closest('.blk, .wlive');
    (blk ? [...blk.querySelectorAll('input, select')] : []).forEach(el => el.addEventListener('input', f));
    const pl = PLAY[x]; if (!pl || !has(x + '-play')) return;
    let vals = pl[1];
    if (x === 'fc') vals = range(+$('fc-o').min, ROE, 1);
    players[b] = new Player(x + '-play', $(pl[0]), vals, pl[2]);
  });
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if (has(b) && live.includes(b)) predict(b, Object.assign({ player: players[b], veil: { out: true } }, sp)); });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  live.forEach(b => BLK[b][1]());
  let w = window.innerWidth, tm = null;
  window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(() => { if ((window.innerWidth < 720) !== (w < 720) || Math.abs(window.innerWidth - w) > 200) { w = window.innerWidth; live.forEach(b => BLK[b][1]()); } }, 250); });
}
window.CH6 = { RO, MKT, US, roY, mktY, usY, varFit, maRep, orthIrf, fevdOf, girfOf, gfevd, spillIdx, companion, eigvals, granger, forecast, meanOf, icTable, portmanteau, jbTest, residFit,
  ccfStats, simVar, wxOrth, wxPath, eig2, wxA, roIrf, roBoot, grRo, grMkt, mktIrf, mktFit, spillTab, rollSpill, oosStats, dmTest, usFit, skewK, fsf, fppf, chi2sf, roIdx, mIdx, ORD, PAIRS, BLK };
if (!window.CH6_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
