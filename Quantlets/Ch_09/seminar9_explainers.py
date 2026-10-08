"""
seminar9_explainers.py -- explanatory (primer) charts for Seminar 9 (TSA): machine learning for time series
==========================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 9, which takes place
BEFORE Lecture 9. All charts use SIMULATED data only (fixed seeds): they illustrate the concepts (a lag table,
walk-forward validation, ridge and lasso shrinkage, one split of a regression tree, a random forest and gradient
boosting, the pinball loss, a split conformal interval, a Diebold-Mariano loss differential, a ROC curve) and contain
no exercise answers.

Charts are drawn at the size of their box on the slide (1 pt in the figure = 1 pt on the slide).
Output: charts/ch9_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).
Run:  python3 Quantlets/Ch_09/seminar9_explainers.py
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import tsa_style as st                     # noqa: E402

warnings.filterwarnings('ignore')
st.apply()
# sizes for the slide: full width 5.45 in (box 0.98\textwidth x 0.66\textheight), a column chart 2.6 in wide
# (0.47\textwidth x 0.86\textheight); the legend is added below
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.0, 'pdf.fonttype': 42})
FULL = (5.45, 1.75)
FULL_L = (5.45, 1.55)
HALF = (2.6, 1.95)


def legend(fig, ncol=3):
    fig.tight_layout()
    st.fig_legend_bottom(fig, ncol=ncol)


def save(name):
    st.check_no_grey(plt.gcf())
    st.save_fig(f'ch9_sem_primer_{name}')


def seasonal_series(n, rng):
    t = np.arange(n)
    return 10 + 2 * np.sin(2 * np.pi * t / 7) + np.cumsum(0.25 * rng.standard_normal(n)) + 0.4 * rng.standard_normal(n)


# =============================================================================
# 1. from a series to a table: lags as features, the value h steps ahead as target
# =============================================================================
def fig_lag_table(seed=1, n=40, origin=24, p=4, h=3):
    rng = np.random.default_rng(seed)
    y = seasonal_series(n, rng)
    t = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=FULL_L)
    ax.plot(t, y, color=st.MainBlue, lw=0.9, marker='o', ms=2, label='Series $y_t$')
    lag = np.arange(origin - p + 1, origin + 1)
    ax.plot(lag, y[lag - 1], 'o', ms=5, color=st.Forest, label=f'Features: lags $y_t, \\dots, y_{{t-{p - 1}}}$')
    ax.plot(origin + h, y[origin + h - 1], 's', ms=6, color=st.IDAred, label=f'Target $y_{{t+h}}$, $h$ = {h}')
    ax.axvline(origin, color=st.Amber, lw=1.0, ls='--', label='Forecast origin $t$')
    ax.annotate('', xy=(origin + h, y[origin + h - 1] + 1.2), xytext=(origin, y[origin + h - 1] + 1.2),
                arrowprops=dict(arrowstyle='->', color=st.IDAred, lw=0.8))
    ax.text(origin + h / 2, y[origin + h - 1] + 1.5, '$h$ steps', color=st.IDAred, ha='center', fontsize=7)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$y_t$')
    ax.set_title('One row of the table: features known at $t$, target $h$ steps later', loc='left')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=4)
    save('lag_table')


# =============================================================================
# 2. walk-forward validation against shuffled K-fold
# =============================================================================
def fig_walk_forward(n=24, K=5, seed=2):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(5.45, 1.35))
    ax = axes[0]
    for i in range(K):
        o = 10 + 2 * i
        ax.add_patch(Rectangle((0, i), o, 0.7, color=st.MainBlue, lw=0))
        ax.add_patch(Rectangle((o, i), 2, 0.7, color=st.IDAred, lw=0))
    ax.set_xlim(0, n)
    ax.set_ylim(K, -0.3)
    ax.set_yticks(np.arange(K) + 0.35)
    ax.set_yticklabels([f'origin {i + 1}' for i in range(K)])
    ax.set_xlabel('Time')
    ax.set_title('Walk-forward: train on the past only', loc='left')
    ax = axes[1]
    idx = rng.permutation(n)
    folds = np.array_split(idx, K)
    for i, f in enumerate(folds):
        for j in range(n):
            ax.add_patch(Rectangle((j, i), 0.95, 0.7, color=st.IDAred if j in f else st.MainBlue, lw=0))
    ax.set_xlim(0, n)
    ax.set_ylim(K, -0.3)
    ax.set_yticks(np.arange(K) + 0.35)
    ax.set_yticklabels([f'fold {i + 1}' for i in range(K)])
    ax.set_xlabel('Time')
    ax.set_title('Shuffled $K$-fold: the future trains the model', loc='left')
    h = [Rectangle((0, 0), 1, 1, color=st.MainBlue), Rectangle((0, 0), 1, 1, color=st.IDAred)]
    fig.tight_layout()
    st.fig_legend_bottom(fig, h, ['Training data', 'Test data'], ncol=2)
    save('walk_forward')


# =============================================================================
# 3. ridge and lasso with one standardised feature: coefficient as a function of lambda
# =============================================================================
def fig_shrinkage(Sxy=30.0, Sxx=40.0):
    lam = np.linspace(0, 100, 401)
    ols = Sxy / Sxx
    ridge = Sxy / (Sxx + lam)
    lasso = np.sign(Sxy) * np.maximum(np.abs(Sxy) - lam / 2, 0) / Sxx
    fig, ax = plt.subplots(figsize=HALF)
    ax.axhline(ols, color=st.DarkText, lw=0.7, ls=':', label=f'OLS = {ols:.2f}')
    ax.plot(lam, ridge, color=st.MainBlue, label='Ridge $S_{xy}/(S_{xx}+\\lambda)$')
    ax.plot(lam, lasso, color=st.IDAred, label='Lasso (soft thresholding)')
    ax.axvline(2 * Sxy, color=st.IDAred, lw=0.7, ls='--')
    ax.text(2 * Sxy + 2, 0.45, '$\\lambda = 2|S_{xy}|$:\nlasso = 0', color=st.IDAred, fontsize=7)
    ax.set_xlabel('Penalty $\\lambda$')
    ax.set_ylabel('Coefficient $\\hat\\beta$')
    ax.set_title(f'$S_{{xy}}$ = {Sxy:g}, $S_{{xx}}$ = {Sxx:g}', loc='left')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=1)
    save('shrinkage')


# =============================================================================
# 4. one split of a regression tree: total SSE as a function of the threshold
# =============================================================================
def fig_tree_split(seed=4, n=30):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, n))
    y = np.where(x < 6, 3.0, 6.0) + 0.6 * rng.standard_normal(n)
    cands = (x[1:] + x[:-1]) / 2
    sse = np.array([((y[x <= c] - y[x <= c].mean()) ** 2).sum() + ((y[x > c] - y[x > c].mean()) ** 2).sum() for c in cands])
    c = cands[np.argmin(sse)]
    fig, axes = plt.subplots(1, 2, figsize=FULL_L)
    ax = axes[0]
    ax.plot(x, y, 'o', ms=3, color=st.MainBlue, label='Data $(x_i, y_i)$')
    ax.plot([0, c], [y[x <= c].mean()] * 2, color=st.IDAred, lw=1.5, label='Leaf means (the forecast)')
    ax.plot([c, 10], [y[x > c].mean()] * 2, color=st.IDAred, lw=1.5)
    ax.axvline(c, color=st.Amber, lw=1.0, ls='--', label=f'Best threshold $c$ = {c:.2f}')
    ax.set_xlabel('Feature $x$')
    ax.set_ylabel('Target $y$')
    ax.set_title('Split "$x \\leq c$?"', loc='left')
    ax = axes[1]
    ax.plot(cands, sse, color=st.Forest, marker='o', ms=2, label='Total SSE of the two leaves')
    ax.axhline(((y - y.mean()) ** 2).sum(), color=st.Purple, lw=0.8, ls=':', label='SSE without a split')
    ax.axvline(c, color=st.Amber, lw=1.0, ls='--')
    ax.set_xlabel('Candidate threshold $c$')
    ax.set_ylabel('SSE')
    ax.set_title('Choose the $c$ with the smallest SSE', loc='left')
    legend(fig, ncol=3)
    save('tree_split')


# =============================================================================
# 5. ensembles: one deep tree, a random forest, gradient boosting after 1, 10, 100 trees
# =============================================================================
def fig_ensembles(seed=5, n=120):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, n))
    f = np.sin(x) + 0.3 * x
    y = f + 0.4 * rng.standard_normal(n)
    X = x[:, None]
    g = np.linspace(0, 10, 400)[:, None]
    fig, axes = plt.subplots(1, 2, figsize=FULL_L, sharey=True)
    ax = axes[0]
    ax.plot(x, y, 'o', ms=1.8, color=st.MainBlue, label='Data')
    ax.plot(g, DecisionTreeRegressor(random_state=0).fit(X, y).predict(g), color=st.Amber, lw=1.0, label='One deep tree')
    ax.plot(g, RandomForestRegressor(300, min_samples_leaf=8, random_state=0).fit(X, y).predict(g), color=st.IDAred, lw=1.3,
            label='Random forest (300 trees)')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Averaging trees reduces the noise', loc='left')
    ax = axes[1]
    ax.plot(x, y, 'o', ms=1.8, color=st.MainBlue)
    for M, c in ((1, st.Purple), (10, st.Forest), (100, st.Orange)):
        gb = GradientBoostingRegressor(n_estimators=M, learning_rate=0.1, max_depth=2, random_state=0).fit(X, y)
        ax.plot(g, gb.predict(g), color=c, lw=1.1, label=f'Boosting, {M} tree' + ('s' if M > 1 else ''))
    ax.set_xlabel('$x$')
    ax.set_title('Boosting: small trees on the residuals, $\\nu$ = 0.1', loc='left')
    legend(fig, ncol=3)
    save('ensembles')


# =============================================================================
# 6. the pinball loss for three quantile levels
# =============================================================================
def fig_pinball():
    u = np.linspace(-2, 2, 401)                  # u = y - q
    fig, ax = plt.subplots(figsize=HALF)
    for tau, c in ((0.1, st.MainBlue), (0.5, st.Forest), (0.9, st.IDAred)):
        ax.plot(u, np.where(u >= 0, tau * u, (tau - 1) * u), color=c, label=f'$\\tau$ = {tau}')
    ax.axvline(0, color=st.DarkText, lw=0.5, ls=':')
    ax.set_xlabel('Error $y - q$')
    ax.set_ylabel('Pinball loss')
    ax.set_title('Slopes $\\tau$ (right) and $1-\\tau$ (left)', loc='left')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=3)
    save('pinball')


# =============================================================================
# 7. split conformal: the quantile of the absolute calibration errors
# =============================================================================
def fig_conformal(seed=7, n=99, alpha=0.1):
    rng = np.random.default_rng(seed)
    e = np.abs(0.3 * rng.standard_t(5, n))
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q = np.sort(e)[k - 1]
    fig, ax = plt.subplots(figsize=HALF)
    ax.hist(e, bins=20, color=st.MainBlue, alpha=0.8, label=f'{n} absolute calibration errors')
    ax.axvline(q, color=st.IDAred, lw=1.3, label=f'$\\hat q$ = {k}-th smallest = {q:.2f}')
    ax.set_xlabel('$|y_i - \\hat y_i|$')
    ax.set_ylabel('Number of days')
    ax.set_title(f'90% interval: $\\hat y \\pm \\hat q$', loc='left')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=1)
    save('conformal')
    return dict(k=k, q=float(q), share=float(np.mean(e <= q)))


# =============================================================================
# 8. Diebold-Mariano: the loss differential and its mean
# =============================================================================
def fig_dm(seed=8, n=60):
    rng = np.random.default_rng(seed)
    e1 = rng.standard_normal(n) * 0.9
    e2 = rng.standard_normal(n) * 1.3
    d = np.abs(e1) - np.abs(e2)
    m, s = d.mean(), d.std(ddof=1)
    fig, ax = plt.subplots(figsize=HALF)
    ax.bar(np.arange(1, n + 1), d, color=np.where(d < 0, st.Forest, st.IDAred), width=0.8)
    ax.axhline(m, color=st.MainBlue, lw=1.3, label=f'$\\bar d = {m:.2f}$')
    ax.axhspan(m - 1.96 * s / np.sqrt(n), m + 1.96 * s / np.sqrt(n), color=st.MainBlue, alpha=0.15, lw=0,
               label='$\\bar d \\pm 1.96\\,\\hat s/\\sqrt{n}$')
    ax.axhline(0, color=st.DarkText, lw=0.5)
    ax.set_xlabel('Forecast origin')
    ax.set_ylabel('$d_t = |e_{1t}| - |e_{2t}|$')
    ax.set_title(f'DM = ${m / (s / np.sqrt(n)):.2f}$', loc='left')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=2)
    save('dm')
    return dict(dbar=float(m), dm=float(m / (s / np.sqrt(n))))


# =============================================================================
# 9. classification of the sign: a ROC curve of a weak classifier
# =============================================================================
def fig_roc(seed=9, n=3000):
    rng = np.random.default_rng(seed)
    yv = rng.random(n) < 0.55
    score = 0.35 * yv + rng.standard_normal(n)
    order = np.argsort(-score)
    tp = np.cumsum(yv[order]) / yv.sum()
    fp = np.cumsum(~yv[order]) / (~yv).sum()
    auc = np.trapz(np.r_[0, tp], np.r_[0, fp])
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(np.r_[0, fp], np.r_[0, tp], color=st.IDAred, label=f'Weak classifier, AUC = {auc:.2f}')
    ax.plot([0, 1], [0, 1], color=st.MainBlue, ls='--', lw=0.9, label='No skill, AUC = 0.5')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve: all thresholds of the score', loc='left')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=1)
    save('roc')
    return dict(auc=float(auc))


if __name__ == '__main__':
    for fn in (fig_lag_table, fig_walk_forward, fig_shrinkage, fig_tree_split, fig_ensembles, fig_pinball,
               fig_conformal, fig_dm, fig_roc):
        print(fn.__name__, fn())
