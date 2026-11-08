// Shared helpers of the interactive chapter pages of TSA (ported from MFM). Exposed as window.TSAI.
(function () {
'use strict';
const LANG = window.TSA_LANG === 'ro' ? 'ro' : 'en';
const C = { blue: '#2D5BA8', red: '#CD0000', amber: '#B5853F', green: '#2E7D32', navy: '#1F2A44', band: '#C5D2E8', grid: '#E3E9F4' };
const PLAY = LANG === 'ro' ? ['Pornește', 'Pauză'] : ['Play', 'Pause'];
// ---------------------------------------------------------------- linear algebra
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const mv = (M, v) => M.map(r => dot(r, v));
const quad = (w, S) => dot(w, mv(S, w));
const scale = (M, k) => M.map(r => r.map(x => x * k));
const sum = a => a.reduce((s, x) => s + x, 0);
const sub = (S, idx) => idx.map(i => idx.map(j => S[i][j]));
function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
  return x;
}
function chol(A) {
  const n = A.length, L = A.map(() => new Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
    L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-14)) : s / L[j][j];
  }
  return L;
}
function eigSym(A) {                // Jacobi rotations; eigenvalues in decreasing order
  const n = A.length, M = A.map(r => [...r]);
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += M[i][j] * M[i][j];
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(M[p][q]) < 1e-15) continue;
      const th = (M[q][q] - M[p][p]) / (2 * M[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const a = M[k][p], b = M[k][q]; M[k][p] = c * a - s * b; M[k][q] = s * a + c * b; }
      for (let k = 0; k < n; k++) { const a = M[p][k], b = M[q][k]; M[p][k] = c * a - s * b; M[q][k] = s * a + c * b; }
    }
  }
  return M.map((r, i) => r[i]).sort((a, b) => b - a);
}
const cov2corr = S => S.map((r, i) => r.map((v, j) => v / Math.sqrt(S[i][i] * S[j][j])));

// ---------------------------------------------------------------- time series
const diff = (x, d = 1) => x.slice(d).map((v, i) => v - x[i]);          // x_t - x_{t-d}
function acf(x, maxLag) {           // sample autocorrelations r_1..r_maxLag (denominator n, as statsmodels acf)
  const n = x.length, m = sum(x) / n, c0 = sum(x.map(v => (v - m) ** 2)) / n, r = [];
  for (let k = 1; k <= maxLag; k++) { let c = 0; for (let t = k; t < n; t++) c += (x[t] - m) * (x[t - k] - m); r.push(c / n / c0); }
  return r;
}
function pacf(r) {                  // partial autocorrelations from r_1..r_K (Durbin–Levinson)
  const K = r.length, out = [], phi = [];
  for (let k = 1; k <= K; k++) {
    let a;
    if (k === 1) a = r[0];
    else { let num = r[k - 1], den = 1; for (let j = 1; j < k; j++) { num -= phi[j - 1] * r[k - 1 - j]; den -= phi[j - 1] * r[j - 1]; } a = num / den; }
    const nphi = phi.map((p, j) => p - a * phi[k - 2 - j]); nphi.push(a); phi.length = 0; phi.push(...nphi); out.push(a);
  }
  return out;
}
function ljungBox(x, h, fitdf = 0) { // Q = n(n+2) sum r_k^2/(n-k), k = 1..h; degrees of freedom h - fitdf
  const n = x.length, r = acf(x, h); let q = 0;
  for (let k = 1; k <= h; k++) q += r[k - 1] ** 2 / (n - k);
  return { q: n * (n + 2) * q, df: h - fitdf };
}
function armaSim(phi, theta, n, z, burn = 200, c = 0, sigma = 1) {   // X_t = c + sum phi_i X_{t-i} + e_t + sum theta_j e_{t-j}, e ~ N(0, sigma^2)
  const p = phi.length, q = theta.length, x = [], e = [];
  for (let t = 0; t < n + burn; t++) {
    const et = sigma * z(); let v = c + et;
    for (let i = 1; i <= p; i++) v += phi[i - 1] * (t - i >= 0 ? x[t - i] : 0);
    for (let j = 1; j <= q; j++) v += theta[j - 1] * (t - j >= 0 ? e[t - j] : 0);
    x.push(v); e.push(et);
  }
  return x.slice(burn);
}
function garchSim(omega, alpha, beta, n, z, burn = 500) {   // GARCH(1,1): eps_t = sigma_t z_t, sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2
  let s2 = omega / Math.max(1 - alpha - beta, 1e-6), e = 0; const eps = [], sig = [];
  for (let t = 0; t < n + burn; t++) { s2 = omega + alpha * e * e + beta * s2; e = Math.sqrt(s2) * z(); if (t >= burn) { eps.push(e); sig.push(Math.sqrt(s2)); } }
  return { eps, sig };
}
function normCdf(x) {               // standard Normal distribution function (Abramowitz–Stegun 7.1.26 on erf, |error| < 1.5e-7)
  const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x / 2);
  return x >= 0 ? (1 + y) / 2 : (1 - y) / 2;
}
function normPpf(p) {               // its inverse (Acklam), relative error < 1.2e-9
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416], lo = 0.02425;
  if (p < lo) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - lo) return -normPpf(1 - p);
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
const quantile = (a, p) => { const b = [...a].sort((x, y) => x - y), h = (b.length - 1) * p, k = Math.floor(h); return b[k] + (h - k) * ((b[k + 1] ?? b[k]) - b[k]); };

// ---------------------------------------------------------------- random numbers (seeded)
function rng(seed) {
  let a = seed >>> 0;
  const u = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  let spare = null;
  const z = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let x, y, r; do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0);
    const f = Math.sqrt(-2 * Math.log(r) / r); spare = y * f; return x * f;
  };
  z.u = u;
  return z;
}
function sampleMoments(mu, L, T, z) {   // T Normal draws with mean mu and Cholesky factor L
  const n = mu.length, m = new Array(n).fill(0), X = [];
  for (let t = 0; t < T; t++) { const e = Array.from({ length: n }, z), x = mu.map((v, i) => v + dot(L[i], e)); X.push(x); for (let i = 0; i < n; i++) m[i] += x[i] / T; }
  const S = m.map(() => new Array(n).fill(0));
  for (const x of X) for (let i = 0; i < n; i++) { const di = x[i] - m[i]; for (let j = 0; j <= i; j++) S[i][j] += di * (x[j] - m[j]) / (T - 1); }
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) S[j][i] = S[i][j];
  return { m, S };
}
const median = a => { const b = [...a].sort((x, y) => x - y); return (b[(b.length - 1) >> 1] + b[b.length >> 1]) / 2; };
const mean = a => sum(a) / a.length;
const sdev = a => { const m = mean(a); return Math.sqrt(sum(a.map(x => (x - m) ** 2)) / (a.length - 1)); };

// ---------------------------------------------------------------- formatting and charts
const nf = (x, d = 1) => { const s = x.toFixed(d).replace('-', '−'); return LANG === 'ro' ? s.replace('.', ',') : s; };
const pct = (x, d = 1) => nf(100 * x, d) + '%';
const $ = id => document.getElementById(id);
const AX = (title, extra) => Object.assign({ title: { text: title, font: { size: 12, color: C.navy } }, gridcolor: C.grid, zerolinecolor: C.band, color: C.navy, tickfont: { size: 11, color: C.navy }, automargin: true }, extra || {});
function layout(extra) {
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, system-ui, sans-serif', color: C.navy, size: 12 },
    margin: { l: 56, r: 12, t: 30, b: 40 },
    legend: { orientation: 'h', x: 0, xanchor: 'left', y: -0.2, yanchor: 'top', bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } },
    hoverlabel: { bgcolor: '#fff', bordercolor: C.navy, font: { color: C.navy, family: 'Inter, sans-serif' } },
    hovermode: 'closest'
  }, extra || {});
}
const ttl = s => ({ text: s, x: 0, xanchor: 'left', font: { size: 12, color: C.navy } });
const CFG = { displayModeBar: false, responsive: true };
const plot = (id, tr, lay) => Plotly.react(id, tr, lay, CFG);
const out = (id, items) => { $(id).innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
const vline = (x, color, dash) => ({ type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color, width: 2, dash: dash || 'solid' } });
const hline = (y, color, dash) => ({ type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: y, y1: y, line: { color, width: 2, dash: dash || 'solid' } });

// ---------------------------------------------------------------- play buttons
class Player {
  constructor(id, input, values, ms) { this.b = $(id); this.input = input; this.values = values; this.ms = ms; this.timer = null; this.b.addEventListener('click', () => this.toggle()); this.label(); }
  label() { this.b.textContent = this.timer ? '⏸ ' + PLAY[1] : '▶ ' + PLAY[0]; }
  toggle() { this.timer ? this.stop() : this.start(); }
  start() {
    const el = this.input, cur = this.values.findIndex(v => +v >= +el.value - 1e-9);
    let i = cur >= 0 && cur < this.values.length - 1 ? cur + 1 : 0;
    this.timer = setInterval(() => {
      el.value = this.values[i]; el.dispatchEvent(new Event('input'));
      if (++i >= this.values.length) { this.stop(); if (typeof this.onEnd === 'function') this.onEnd(); }
    }, this.ms);
    this.label();
  }
  stop() { clearInterval(this.timer); this.timer = null; this.label(); }
}
const range = (a, b, st) => { const r = []; for (let x = a; st > 0 ? x <= b + 1e-9 : x >= b - 1e-9; x += st) r.push(+x.toFixed(6)); return r; };

// ---------------------------------------------------------------- opt-in: predict, then check (per block) and exit ticket
// Pages that do not call predict() or exitTicket() are unchanged.
const UI = LANG === 'ro'
  ? { pick: 'Ce credeți? Alegeți o variantă:', check: 'Verifică', right: 'Corect.', wrong: 'Nu: varianta corectă este marcată.', checkAll: 'Verifică răspunsurile',
      score: (k, n) => `${k} din ${n} răspunsuri corecte.`, none: 'Fără răspuns.', ok: 'Corect', bad: 'Greșit', title: 'Verifică-te',
      show: 'Arată slide-ul', hidden: 'Rezultatul apare după „Verifică”, după ▶ sau cînd modificați parametrii.' }
  : { pick: 'What do you think? Pick an option:', check: 'Check', right: 'Correct.', wrong: 'No: the correct option is marked.', checkAll: 'Check answers',
      score: (k, n) => `${k} of ${n} answers correct.`, none: 'No answer.', ok: 'Correct', bad: 'Wrong', title: 'Check Yourself',
      show: 'Show slide', hidden: 'The result appears after “Check”, after ▶, or when you change the parameters.' };
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
// Opt-in veil of a block: the slide image is blurred under a „Arată slide-ul” / "Show slide" button and, with { out: true },
// the readout of the live figure is blurred too; with { charts: true } (all .chart divs of the block) or { charts: [ids] } the
// live charts are blurred and made inert (no hover), for questions whose answer the initial chart shows. Returns reveal(),
// which removes all of it. A trusted (user) input in the block's controls reveals the readout and the charts only, since they
// then describe the reader's own choice.
function veil(blk, opt) {
  const fig = blk.querySelector('figure.slide'), outs = opt && opt.out ? [...blk.querySelectorAll('.out')] : [];
  const cs = !opt || !opt.charts ? [] : Array.isArray(opt.charts) ? opt.charts.map(id => $(id)).filter(e => e && blk.contains(e))
    : [...blk.querySelectorAll('.chart')];
  if (!fig && !outs.length && !cs.length) return () => {};
  const a = fig && fig.querySelector('a'), b = document.createElement('button');
  if (fig) {
    fig.classList.add('veiled');
    b.type = 'button'; b.className = 'btn play veil-btn'; b.textContent = UI.show;
    fig.appendChild(b);
    if (a) { a.setAttribute('tabindex', '-1'); a.setAttribute('aria-hidden', 'true'); }
  }
  const showOut = () => {
    outs.forEach(o => { o.classList.remove('veiled-out'); o.removeAttribute('title'); });
    cs.forEach(c => { c.classList.remove('veiled-chart'); c.removeAttribute('aria-hidden'); c.removeAttribute('title'); });
  };
  outs.forEach(o => { o.classList.add('veiled-out'); o.title = UI.hidden; });
  cs.forEach(c => { c.classList.add('veiled-chart'); c.setAttribute('aria-hidden', 'true'); c.title = UI.hidden; });
  let on = true;
  const reveal = () => {
    showOut(); if (!on) return; on = false;
    if (fig) { fig.classList.remove('veiled'); b.remove(); if (a) { a.removeAttribute('tabindex'); a.removeAttribute('aria-hidden'); } }
  };
  b.addEventListener('click', () => { reveal(); if (a) a.focus(); });
  const ctls = blk.querySelector('.ctls');
  if (ctls && (outs.length || cs.length)) ['input', 'change'].forEach(ev => ctls.addEventListener(ev, e => { if (e.isTrusted) showOut(); }));
  return reveal;
}
// Deterministic shuffle of the options, seeded by a string, so the correct option does not sit at a predictable position;
// returns the options in display order and the display index of the key.
function shuffleOpts(seed, opts, key) {
  let h = 2166136261;
  for (const ch of String(seed)) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619) >>> 0; }
  const idx = opts.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) { h = Math.imul(h ^ (h >>> 15), 2246822519) >>> 0; const j = h % (i + 1); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  return { opts: idx.map(i => opts[i]), key: idx.indexOf(key) };
}
// spec: { opts: [text, ...], key: index of the correct option, player: Player (optional),
//         veil: true | { out: true, charts: true | [chart ids] } (optional: blur the slide, the readout with out: true and the
//         live charts with charts, until the reader presses „Arată slide-ul”, presses „Verifică” or runs ▶ to the end) };
//         options are shown in a fixed shuffled order
function predict(blockId, spec) {
  const blk = $(blockId), ask = blk && blk.querySelector('.ask'), det = blk && blk.querySelector('details');
  if (!ask || !spec || !spec.opts) return null;
  spec = Object.assign({}, spec, shuffleOpts('p:' + blockId, spec.opts, spec.key));
  const reveal = spec.veil ? veil(blk, spec.veil) : () => {};
  const nm = 'pred-' + blockId;
  const fs = document.createElement('fieldset'); fs.className = 'pred';
  fs.innerHTML = `<legend>${UI.pick}</legend>` + spec.opts.map((t, i) =>
    `<label class="pick"><input type="radio" name="${nm}" value="${i}"><span>${esc(t)}</span></label>`).join('') +
    `<div class="pred-act"><button class="btn" type="button">${UI.check}</button><span class="pred-res" role="status" aria-live="polite"></span></div>`;
  ask.appendChild(fs);
  const labels = [...fs.querySelectorAll('label.pick')], res = fs.querySelector('.pred-res');
  const chosen = () => { const r = fs.querySelector('input:checked'); return r ? +r.value : null; };
  const clear = () => { labels.forEach(l => l.classList.remove('ok', 'bad')); res.textContent = ''; res.className = 'pred-res'; };
  const check = () => {
    const c = chosen(); clear(); reveal();
    if (c !== null) {
      labels[spec.key].classList.add('ok');
      if (c !== spec.key) labels[c].classList.add('bad');
      res.textContent = c === spec.key ? UI.right : UI.wrong; res.classList.add(c === spec.key ? 'is-ok' : 'is-bad');
    }
    if (det) det.open = true;
  };
  fs.querySelector('button').addEventListener('click', check);
  fs.addEventListener('change', clear);
  if (spec.player) spec.player.onEnd = () => { reveal(); if (chosen() !== null) check(); };
  return { check, chosen, reveal };
}
// spec: { title (optional), id (default 'check'), store (localStorage key, optional),
//         qs: [{ q, key, opts: [{ t, why }, ...] }, ...] }; inserted before the page footnote (or at the end of <main>)
function exitTicket(spec) {
  const main = document.querySelector('main'); if (!main || !spec || !spec.qs) return null;
  const id = spec.id || 'check', title = spec.title || UI.title;
  spec = Object.assign({}, spec, { qs: spec.qs.map((q, k) => Object.assign({}, q, shuffleOpts(`t:${id}:${k}`, q.opts, q.key))) });
  const sec = document.createElement('section'); sec.className = 'ticket'; sec.id = id;
  sec.innerHTML = `<h2 class="sec">${esc(title)}</h2><ol class="tq">` + spec.qs.map((q, k) =>
    `<li><fieldset><legend>${esc(q.q)}</legend>` + q.opts.map((o, i) =>
      `<label class="pick"><input type="radio" name="${id}-q${k}" value="${i}"><span>${esc(o.t)}<span class="why" hidden></span></span></label>`).join('') +
    `</fieldset></li>`).join('') + `</ol><div class="pred-act"><button class="btn play" type="button">${UI.checkAll}</button><span class="pred-res" role="status" aria-live="polite"></span></div>`;
  const foot = main.querySelector('.src.foot');
  foot ? main.insertBefore(sec, foot) : main.appendChild(sec);
  const toc = main.querySelector('.toc');
  if (toc && !toc.querySelector(`a[href="#${id}"]`)) toc.insertAdjacentHTML('beforeend', `${toc.children.length ? ' · ' : ''}<a href="#${id}">${esc(title)}</a>`);
  const items = [...sec.querySelectorAll('ol.tq > li')], res = sec.querySelector('.pred-res');
  const save = () => { if (!spec.store) return; try { localStorage.setItem(spec.store, JSON.stringify(items.map(li => { const r = li.querySelector('input:checked'); return r ? +r.value : null; }))); } catch (e) { /* storage unavailable */ } };
  const clear = li => { li.querySelectorAll('label.pick').forEach(l => l.classList.remove('ok', 'bad', 'no', 'mine')); li.querySelectorAll('.why').forEach(w => { w.hidden = true; w.textContent = ''; }); };
  const check = () => {
    let n = 0;
    items.forEach((li, k) => {
      const q = spec.qs[k], r = li.querySelector('input:checked'), c = r ? +r.value : null;
      clear(li);
      li.querySelectorAll('label.pick').forEach((l, i) => {
        l.classList.add(i === q.key ? 'ok' : i === c ? 'bad' : 'no'); if (i === c) l.classList.add('mine');
        const w = l.querySelector('.why'); w.textContent = `${i === q.key ? UI.ok : UI.bad}. ${q.opts[i].why}`; w.hidden = false;
      });
      if (c === q.key) n++;
    });
    res.textContent = UI.score(n, items.length); res.className = 'pred-res';
  };
  sec.querySelector('button').addEventListener('click', check);
  items.forEach(li => li.addEventListener('change', () => { clear(li); res.textContent = ''; save(); }));
  if (spec.store) {
    try { const v = JSON.parse(localStorage.getItem(spec.store) || 'null'); if (Array.isArray(v)) v.forEach((c, k) => { const r = c !== null && items[k] && items[k].querySelectorAll('input')[c]; if (r) r.checked = true; }); } catch (e) { /* storage unavailable */ }
  }
  if (location.hash === '#' + id) sec.scrollIntoView();
  return { check };
}

window.TSAI = { LANG, C, dot, mv, quad, scale, sum, sub, solve, chol, eigSym, cov2corr, diff, acf, pacf, ljungBox, armaSim, garchSim,
  normCdf, normPpf, quantile, rng, sampleMoments, median, mean, sdev, nf, pct, $, AX, layout, ttl, CFG, plot, out,
  vline, hline, Player, range, predict, exitTicket };
})();

// left-hand menu on every page (assets/sidemenu.js)
;(function () { var c = document.currentScript; if (!c || !c.src) return; var e = document.createElement('script'); e.src = new URL('../assets/sidemenu.js?v=20261108a', c.src).href; document.head.appendChild(e); })();
