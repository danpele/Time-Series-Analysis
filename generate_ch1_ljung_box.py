"""Generate Ljung-Box test illustration chart for ch1."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

np.random.seed(42)
n = 200

# Series 1: White noise
wn = np.random.normal(0, 1, n)

# Series 2: AR(1) with phi=0.7
ar1 = np.zeros(n)
ar1[0] = np.random.normal()
for t in range(1, n):
    ar1[t] = 0.7 * ar1[t-1] + np.random.normal(0, np.sqrt(1 - 0.7**2))

def ljung_box_pvalues(x, max_lag=20):
    """Compute Ljung-Box p-values for lags 1..max_lag."""
    n = len(x)
    # sample ACF
    xm = x - x.mean()
    acf_vals = np.correlate(xm, xm, mode='full') / (n * xm.var())
    acf_vals = acf_vals[n-1:]  # lags 0,1,2,...
    lags = np.arange(1, max_lag + 1)
    Q = np.zeros(max_lag)
    for k in range(max_lag):
        Q[k] = n * (n + 2) * np.sum(
            acf_vals[1:k+2]**2 / (n - np.arange(1, k+2))
        )
    pvals = [1 - stats.chi2.cdf(Q[k], df=k+1) for k in range(max_lag)]
    return lags, Q, np.array(pvals)

lags_wn, Q_wn, pvals_wn = ljung_box_pvalues(wn, 20)
lags_ar, Q_ar, pvals_ar = ljung_box_pvalues(ar1, 20)

fig = plt.figure(figsize=(11, 6))
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.55)

colors = {'wn': '#2196F3', 'ar': '#E53935', 'sig': '#FF9800'}

# --- Row 0: White Noise ---
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(wn[:80], color=colors['wn'], linewidth=0.9)
ax0.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax0.set_title('Zgomot alb $\\epsilon_t$', fontsize=9, fontweight='bold')
ax0.set_xlabel('$t$', fontsize=8); ax0.set_ylabel('$\\epsilon_t$', fontsize=8)
ax0.tick_params(labelsize=7)

ax1 = fig.add_subplot(gs[0, 1])
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(wn, lags=20, ax=ax1, color=colors['wn'], vlines_kwargs={'colors': colors['wn']},
         title='ACF — zgomot alb', alpha=0.05, zero=False, auto_ylims=True)
ax1.set_xlabel('Lag', fontsize=8); ax1.tick_params(labelsize=7)
ax1.title.set_fontsize(9); ax1.title.set_fontweight('bold')

ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(lags_wn, pvals_wn, color=colors['wn'], alpha=0.8, width=0.6)
ax2.axhline(0.05, color=colors['sig'], linewidth=1.5, linestyle='--', label='$\\alpha=0.05$')
ax2.set_ylim(0, 1)
ax2.set_title('Ljung-Box p-valori — WN', fontsize=9, fontweight='bold')
ax2.set_xlabel('Lag $m$', fontsize=8); ax2.set_ylabel('$p$-valoare', fontsize=8)
ax2.tick_params(labelsize=7)
ax2.legend(fontsize=7)
ax2.text(10, 0.5, 'Toate $p > 0.05$\n$\\Rightarrow$ Nu respingem $H_0$\n(WN confirmată)',
         fontsize=7, ha='center', color=colors['wn'],
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', alpha=0.8))

# --- Row 1: AR(1) ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(ar1[:80], color=colors['ar'], linewidth=0.9)
ax3.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax3.set_title('Proces AR(1), $\\phi=0.7$', fontsize=9, fontweight='bold')
ax3.set_xlabel('$t$', fontsize=8); ax3.set_ylabel('$X_t$', fontsize=8)
ax3.tick_params(labelsize=7)

ax4 = fig.add_subplot(gs[1, 1])
plot_acf(ar1, lags=20, ax=ax4, color=colors['ar'], vlines_kwargs={'colors': colors['ar']},
         title='ACF — AR(1)', alpha=0.05, zero=False, auto_ylims=True)
ax4.set_xlabel('Lag', fontsize=8); ax4.tick_params(labelsize=7)
ax4.title.set_fontsize(9); ax4.title.set_fontweight('bold')

ax5 = fig.add_subplot(gs[1, 2])
bar_colors = [colors['ar'] if p < 0.05 else '#90A4AE' for p in pvals_ar]
ax5.bar(lags_ar, pvals_ar, color=bar_colors, alpha=0.85, width=0.6)
ax5.axhline(0.05, color=colors['sig'], linewidth=1.5, linestyle='--', label='$\\alpha=0.05$')
ax5.set_ylim(0, 1)
ax5.set_title('Ljung-Box p-valori — AR(1)', fontsize=9, fontweight='bold')
ax5.set_xlabel('Lag $m$', fontsize=8); ax5.set_ylabel('$p$-valoare', fontsize=8)
ax5.tick_params(labelsize=7)
ax5.legend(fontsize=7)
ax5.text(10, 0.5, '$p < 0.05$ la lag-uri mici\n$\\Rightarrow$ Respingem $H_0$\n(autocorrelație detectată)',
         fontsize=7, ha='center', color=colors['ar'],
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', alpha=0.8))

fig.suptitle('Testul Ljung-Box: $H_0$ — seria este zgomot alb',
             fontsize=10, fontweight='bold', y=1.01)

plt.savefig('charts/ch1_ljung_box.pdf', bbox_inches='tight', dpi=150)
print("Saved charts/ch1_ljung_box.pdf")
plt.close()
