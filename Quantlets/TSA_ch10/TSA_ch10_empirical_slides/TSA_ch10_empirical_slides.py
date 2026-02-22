"""
TSA_ch10_empirical_slides
==========================
Generates real empirical results for ch10 theoretical slides:
  1. Bitcoin: Rolling VaR with GARCH(1,1) Normal vs Student-t + backtesting
  2. Diebold-Mariano test: Normal vs Student-t GARCH
  3. Forecast metrics beyond RMSE (MASE, Directional Accuracy)
  4. Innovation distribution: real kurtosis, QQ comparison
  5. VAR stability: companion matrix eigenvalues
  6. Johansen cointegration test
  7. Structural breaks: Zivot-Andrews on unemployment

Author: Time Series Analysis Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'charts'))

# Chart style - Nature journal quality
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.75

def save_fig(name):
    path = os.path.join(CHARTS_DIR, name)
    plt.savefig(f'{path}.pdf', bbox_inches='tight', transparent=True, dpi=300)
    plt.savefig(f'{path}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.savefig(f'{path}.pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"   Saved: {path}.pdf")

# =============================================================================
# 1. BITCOIN: GARCH VaR + BACKTESTING
# =============================================================================
print("=" * 70)
print("PART 1: BITCOIN GARCH VaR AND BACKTESTING")
print("=" * 70)

import yfinance as yf
from arch import arch_model

# Download data
btc = yf.download('BTC-USD', start='2019-01-01', end='2025-01-01', progress=False)
if isinstance(btc.columns, pd.MultiIndex):
    btc = btc['Close']['BTC-USD'].dropna()
else:
    btc = btc['Close'].dropna()

returns = btc.pct_change().dropna() * 100
n = len(returns)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = returns.iloc[:train_end]
val = returns.iloc[train_end:val_end]
test = returns.iloc[val_end:]
train_val = returns.iloc[:val_end]

print(f"   Data: {len(returns)} obs, Test: {len(test)} obs")
print(f"   Test period: {test.index[0].strftime('%Y-%m-%d')} to {test.index[-1].strftime('%Y-%m-%d')}")

# =============================================================================
# FIT 4 MODELS: GARCH-N, GARCH-t, GJR-GARCH-t, Historical Simulation
# =============================================================================

# --- Model 1: GARCH(1,1) Normal ---
print("\n   Fitting GARCH(1,1) Normal...")
garch_norm = arch_model(train_val, vol='Garch', p=1, q=1, dist='normal', mean='Constant')
fit_norm = garch_norm.fit(disp='off')
omega_n, alpha_n, beta_n = fit_norm.params['omega'], fit_norm.params['alpha[1]'], fit_norm.params['beta[1]']
mu_n = fit_norm.params['mu']
print(f"   Normal: mu={mu_n:.4f}, omega={omega_n:.6f}, alpha={alpha_n:.4f}, beta={beta_n:.4f}")

# --- Model 2: GARCH(1,1) Student-t ---
print("   Fitting GARCH(1,1) Student-t...")
garch_t = arch_model(train_val, vol='Garch', p=1, q=1, dist='t', mean='Constant')
fit_t = garch_t.fit(disp='off')
omega_t, alpha_t, beta_t = fit_t.params['omega'], fit_t.params['alpha[1]'], fit_t.params['beta[1]']
mu_t = fit_t.params['mu']
nu = fit_t.params['nu']
print(f"   Student-t: mu={mu_t:.4f}, omega={omega_t:.6f}, alpha={alpha_t:.4f}, beta={beta_t:.4f}, nu={nu:.2f}")

# --- Model 3: GJR-GARCH(1,1) Student-t (leverage effect) ---
print("   Fitting GJR-GARCH(1,1) Student-t...")
gjr_model = arch_model(train_val, vol='Garch', p=1, o=1, q=1, dist='t', mean='Constant')
fit_gjr = gjr_model.fit(disp='off')
omega_g = fit_gjr.params['omega']
alpha_g = fit_gjr.params['alpha[1]']
gamma_g = fit_gjr.params['gamma[1]']
beta_g = fit_gjr.params['beta[1]']
mu_g = fit_gjr.params['mu']
nu_g = fit_gjr.params['nu']
print(f"   GJR: mu={mu_g:.4f}, omega={omega_g:.6f}, alpha={alpha_g:.4f}, gamma={gamma_g:.4f}, beta={beta_g:.4f}, nu={nu_g:.2f}")

# --- Innovation distribution statistics ---
resid_norm = fit_norm.resid / fit_norm.conditional_volatility
resid_t = fit_t.resid / fit_t.conditional_volatility
kurt_resid = float(stats.kurtosis(resid_norm.dropna(), fisher=False))
skew_resid = float(stats.skew(resid_norm.dropna()))
print(f"\n   Standardized residuals: kurtosis={kurt_resid:.2f}, skew={skew_resid:.2f}")

# =============================================================================
# ROLLING 1-STEP VaR ON TEST SET (all 4 models)
# =============================================================================
print("\n   Computing rolling VaR on test set (4 models)...")

# Quantiles
z_05_norm = stats.norm.ppf(0.05)
z_01_norm = stats.norm.ppf(0.01)
z_05_t = stats.t.ppf(0.05, df=nu) * np.sqrt((nu - 2) / nu)
z_01_t = stats.t.ppf(0.01, df=nu) * np.sqrt((nu - 2) / nu)
z_05_gjr = stats.t.ppf(0.05, df=nu_g) * np.sqrt((nu_g - 2) / nu_g)
z_01_gjr = stats.t.ppf(0.01, df=nu_g) * np.sqrt((nu_g - 2) / nu_g)

# Allocate arrays for all models
var_norm_05 = np.zeros(len(test))
var_norm_01 = np.zeros(len(test))
var_t_05 = np.zeros(len(test))
var_t_01 = np.zeros(len(test))
var_gjr_05 = np.zeros(len(test))
var_gjr_01 = np.zeros(len(test))
var_hs_05 = np.zeros(len(test))
var_hs_01 = np.zeros(len(test))
sigma_norm = np.zeros(len(test))
sigma_t = np.zeros(len(test))
sigma_gjr = np.zeros(len(test))

all_ret = returns.values.astype(float)

# Initialize with last conditional variance
cv_norm = fit_norm.conditional_volatility.values
cv_t = fit_t.conditional_volatility.values
cv_gjr = fit_gjr.conditional_volatility.values
last_sigma2_n = cv_norm[-1] ** 2
last_sigma2_t = cv_t[-1] ** 2
last_sigma2_g = cv_gjr[-1] ** 2

HS_WINDOW = 250  # Historical simulation window

for i in range(len(test)):
    idx = val_end + i
    r_prev = all_ret[idx - 1]
    eps_prev_n = r_prev - mu_n
    eps_prev_t = r_prev - mu_t
    eps_prev_g = r_prev - mu_g

    # Model 1: GARCH(1,1) Normal
    last_sigma2_n = omega_n + alpha_n * eps_prev_n ** 2 + beta_n * last_sigma2_n
    sig_n = np.sqrt(last_sigma2_n)
    sigma_norm[i] = sig_n
    var_norm_05[i] = mu_n + sig_n * z_05_norm
    var_norm_01[i] = mu_n + sig_n * z_01_norm

    # Model 2: GARCH(1,1) Student-t
    last_sigma2_t = omega_t + alpha_t * eps_prev_t ** 2 + beta_t * last_sigma2_t
    sig_t = np.sqrt(last_sigma2_t)
    sigma_t[i] = sig_t
    var_t_05[i] = mu_t + sig_t * z_05_t
    var_t_01[i] = mu_t + sig_t * z_01_t

    # Model 3: GJR-GARCH(1,1) Student-t
    leverage = gamma_g * eps_prev_g ** 2 * (1 if eps_prev_g < 0 else 0)
    last_sigma2_g = omega_g + alpha_g * eps_prev_g ** 2 + leverage + beta_g * last_sigma2_g
    sig_g = np.sqrt(last_sigma2_g)
    sigma_gjr[i] = sig_g
    var_gjr_05[i] = mu_g + sig_g * z_05_gjr
    var_gjr_01[i] = mu_g + sig_g * z_01_gjr

    # Model 4: Historical Simulation (rolling 250-day window)
    start_hs = max(0, idx - HS_WINDOW)
    window_rets = all_ret[start_hs:idx]
    var_hs_05[i] = np.percentile(window_rets, 5)
    var_hs_01[i] = np.percentile(window_rets, 1)

test_vals = test.values.astype(float)

# Count violations for all models
models_data = {
    'GARCH-N': {'var05': var_norm_05, 'var01': var_norm_01, 'sigma': sigma_norm,
                'color': '#DC3545', 'aic': fit_norm.aic, 'bic': fit_norm.bic},
    'GARCH-t': {'var05': var_t_05, 'var01': var_t_01, 'sigma': sigma_t,
                'color': '#1A3A6E', 'aic': fit_t.aic, 'bic': fit_t.bic},
    'GJR-t':   {'var05': var_gjr_05, 'var01': var_gjr_01, 'sigma': sigma_gjr,
                'color': '#2E7D32', 'aic': fit_gjr.aic, 'bic': fit_gjr.bic},
    'HistSim': {'var05': var_hs_05, 'var01': var_hs_01, 'sigma': None,
                'color': '#E67E22', 'aic': None, 'bic': None},
}

for name, md in models_data.items():
    md['viol05'] = test_vals < md['var05']
    md['viol01'] = test_vals < md['var01']
    md['n_viol05'] = int(md['viol05'].sum())
    md['n_viol01'] = int(md['viol01'].sum())

T = len(test)
print(f"\n   VaR Violations (T={T}):")
for name, md in models_data.items():
    pct05 = md['n_viol05'] / T * 100
    pct01 = md['n_viol01'] / T * 100
    print(f"   {name:10s} 5%: {md['n_viol05']:3d} ({pct05:4.1f}%)   1%: {md['n_viol01']:3d} ({pct01:4.1f}%)")

# Keep backward-compatible variables
viol_norm_05 = models_data['GARCH-N']['viol05']
viol_norm_01 = models_data['GARCH-N']['viol01']
viol_t_05 = models_data['GARCH-t']['viol05']
viol_t_01 = models_data['GARCH-t']['viol01']
n_viol_norm_05 = models_data['GARCH-N']['n_viol05']
n_viol_norm_01 = models_data['GARCH-N']['n_viol01']
n_viol_t_05 = models_data['GARCH-t']['n_viol05']
n_viol_t_01 = models_data['GARCH-t']['n_viol01']

# --- Kupiec Test ---
def kupiec_test(violations, T, alpha):
    """Kupiec unconditional coverage test."""
    x = int(np.sum(violations))
    if x == T:
        return np.inf, 0.0
    if x == 0:
        # LR = -2 * ln[(1-alpha)^T / 1^T] = -2 * T * ln(1-alpha)
        lr = -2 * T * np.log(1 - alpha)
        p_val = 1 - stats.chi2.cdf(lr, 1)
        return float(lr), float(p_val)
    p_hat = x / T
    lr = -2 * (np.log((1 - alpha) ** (T - x) * alpha ** x) -
               np.log((1 - p_hat) ** (T - x) * p_hat ** x))
    p_val = 1 - stats.chi2.cdf(lr, 1)
    return float(lr), float(p_val)

# Run Kupiec on all 4 models
print(f"\n   Kupiec Test (H0: correct coverage):")
for name, md in models_data.items():
    lr05, p05 = kupiec_test(md['viol05'], T, 0.05)
    lr01, p01 = kupiec_test(md['viol01'], T, 0.01)
    md['kupiec_lr05'], md['kupiec_p05'] = lr05, p05
    md['kupiec_lr01'], md['kupiec_p01'] = lr01, p01
    print(f"   {name:10s} 5%: LR={lr05:.2f}, p={p05:.4f} {'REJECT' if p05 < 0.05 else 'OK'}   "
          f"1%: LR={lr01:.2f}, p={p01:.4f} {'REJECT' if p01 < 0.05 else 'OK'}")

# Backward-compatible variables
lr_n05, p_n05 = models_data['GARCH-N']['kupiec_lr05'], models_data['GARCH-N']['kupiec_p05']
lr_t05, p_t05 = models_data['GARCH-t']['kupiec_lr05'], models_data['GARCH-t']['kupiec_p05']

# --- Christoffersen Independence Test ---
def christoffersen_test(violations):
    """Christoffersen conditional coverage test (UC + independence)."""
    v = np.array(violations, dtype=int)
    T = len(v)
    x = int(v.sum())
    if x == 0 or x == T:
        return 0.0, 1.0

    # Transition counts
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))

    # Transition probabilities
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p_hat = (n01 + n11) / (T - 1)

    if p01 == 0 or p11 == 0 or p_hat == 0:
        return 0.0, 1.0
    if p01 == 1 or p11 == 1 or p_hat == 1:
        return 0.0, 1.0

    # Independence LR
    lr_ind_num = (1 - p01) ** n00 * p01 ** n01 * (1 - p11) ** n10 * p11 ** n11
    lr_ind_den = (1 - p_hat) ** (n00 + n10) * p_hat ** (n01 + n11)

    if lr_ind_den == 0 or lr_ind_num == 0:
        return 0.0, 1.0

    lr_ind = -2 * np.log(lr_ind_den / lr_ind_num)
    p_val = 1 - stats.chi2.cdf(lr_ind, 1)
    return float(lr_ind), float(p_val)

# Run Christoffersen on all 4 models (5% and 1%)
print(f"\n   Christoffersen Independence Test:")
for name, md in models_data.items():
    lr_cc, p_cc = christoffersen_test(md['viol05'])
    md['chr_lr05'], md['chr_p05'] = lr_cc, p_cc
    lr_cc1, p_cc1 = christoffersen_test(md['viol01'])
    md['chr_lr01'], md['chr_p01'] = lr_cc1, p_cc1
    print(f"   {name:10s} 5%: LR_ind={lr_cc:.2f}, p={p_cc:.4f}   1%: LR_ind={lr_cc1:.2f}, p={p_cc1:.4f}")

lr_cc_n05, p_cc_n05 = models_data['GARCH-N']['chr_lr05'], models_data['GARCH-N']['chr_p05']
lr_cc_t05, p_cc_t05 = models_data['GARCH-t']['chr_lr05'], models_data['GARCH-t']['chr_p05']

# =============================================================================
# CHART 1: Multi-model VaR comparison (2x2 grid)
# =============================================================================
print("\n   Generating multi-model VaR backtest chart (2x2)...")

dates = test.index
model_list = ['GARCH-N', 'GARCH-t', 'GJR-t', 'HistSim']
titles = ['GARCH(1,1) Normal', 'GARCH(1,1) Student-t', 'GJR-GARCH(1,1) Student-t', 'Historical Simulation']

fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
axes = axes.flatten()

for j, (name, title) in enumerate(zip(model_list, titles)):
    ax = axes[j]
    md = models_data[name]
    ax.plot(dates, test_vals, color='#333333', linewidth=0.4, alpha=0.7)
    ax.plot(dates, md['var05'], color=md['color'], linewidth=0.8, linestyle='--')
    ax.fill_between(dates, md['var05'], test_vals.min() - 2,
                    alpha=0.10, color=md['color'])
    # Mark violations
    vdates = dates[md['viol05']]
    vvals = test_vals[md['viol05']]
    ax.scatter(vdates, vvals, color=md['color'], s=10, zorder=5, marker='v')
    ax.axhline(0, color='gray', linewidth=0.3)
    pct = md['n_viol05'] / T * 100
    kupiec_ok = 'Pass' if md['kupiec_p05'] > 0.05 else 'Fail'
    ax.set_title(f'{title}\nViol: {md["n_viol05"]}/{T} ({pct:.1f}%)  Kupiec: {kupiec_ok}',
                 fontweight='bold', fontsize=8)
    if j >= 2:
        ax.set_xlabel('Date')
    if j % 2 == 0:
        ax.set_ylabel('Return (%)')

plt.tight_layout()
save_fig('ch10_btc_var_multimodel')

# =============================================================================
# CHART 2: Original 2-panel chart (backward compatible)
# =============================================================================
print("   Generating original 2-panel VaR chart...")

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

# Top: Normal VaR
axes[0].plot(dates, test_vals, color='#333333', linewidth=0.5, alpha=0.8, label='BTC Returns')
axes[0].fill_between(dates, var_norm_05, min(test_vals.min(), var_norm_01.min()) - 1,
                      alpha=0.15, color='#DC3545', label='VaR 5% zone')
axes[0].plot(dates, var_norm_05, color='#DC3545', linewidth=0.8, linestyle='--', label='VaR 5%')
axes[0].plot(dates, var_norm_01, color='#8B0000', linewidth=0.8, linestyle=':', label='VaR 1%')
viol_dates_05 = dates[viol_norm_05]
viol_vals_05 = test_vals[viol_norm_05]
axes[0].scatter(viol_dates_05, viol_vals_05, color='#DC3545', s=12, zorder=5, marker='v', label=f'Violations ({n_viol_norm_05})')
axes[0].set_ylabel('Return (%)')
axes[0].set_title(f'GARCH(1,1) Normal — VaR Backtest (violations 5%: {n_viol_norm_05}/{T} = {n_viol_norm_05/T*100:.1f}%)',
                   fontweight='bold')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=6, frameon=False)
axes[0].axhline(0, color='gray', linewidth=0.3)

# Bottom: Student-t VaR
axes[1].plot(dates, test_vals, color='#333333', linewidth=0.5, alpha=0.8, label='BTC Returns')
axes[1].fill_between(dates, var_t_05, min(test_vals.min(), var_t_01.min()) - 1,
                      alpha=0.15, color='#1A3A6E', label='VaR 5% zone')
axes[1].plot(dates, var_t_05, color='#1A3A6E', linewidth=0.8, linestyle='--', label='VaR 5%')
axes[1].plot(dates, var_t_01, color='#0D1B3E', linewidth=0.8, linestyle=':', label='VaR 1%')
viol_dates_t05 = dates[viol_t_05]
viol_vals_t05 = test_vals[viol_t_05]
axes[1].scatter(viol_dates_t05, viol_vals_t05, color='#1A3A6E', s=12, zorder=5, marker='v', label=f'Violations ({n_viol_t_05})')
axes[1].set_ylabel('Return (%)')
axes[1].set_xlabel('Date')
axes[1].set_title(f'GARCH(1,1) Student-t (ν={nu:.1f}) — VaR Backtest (violations 5%: {n_viol_t_05}/{T} = {n_viol_t_05/T*100:.1f}%)',
                   fontweight='bold')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=6, frameon=False)
axes[1].axhline(0, color='gray', linewidth=0.3)

plt.tight_layout()
save_fig('ch10_btc_var_backtest')

# =============================================================================
# CHART 3: All VaR lines overlaid on single panel
# =============================================================================
print("   Generating single-panel overlay chart...")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dates, test_vals, color='#333333', linewidth=0.4, alpha=0.6, label='BTC Returns')
for name, title in zip(model_list, titles):
    md = models_data[name]
    ax.plot(dates, md['var05'], color=md['color'], linewidth=0.9, linestyle='--',
            label=f'{title} ({md["n_viol05"]}/{T})')
ax.axhline(0, color='gray', linewidth=0.3)
ax.set_ylabel('Return (%)')
ax.set_xlabel('Date')
ax.set_title('Rolling VaR 5%: Multi-Model Comparison on Bitcoin Test Set', fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=7, frameon=False)
plt.tight_layout()
save_fig('ch10_btc_var_overlay')

# =============================================================================
# CHART 4: Multi-model VaR 1% comparison (2x2 grid)
# =============================================================================
print("\n   Generating multi-model VaR 1% backtest chart (2x2)...")

fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
axes = axes.flatten()

for j, (name, title) in enumerate(zip(model_list, titles)):
    ax = axes[j]
    md = models_data[name]
    ax.plot(dates, test_vals, color='#333333', linewidth=0.4, alpha=0.7)
    ax.plot(dates, md['var01'], color=md['color'], linewidth=0.8, linestyle='--')
    ax.fill_between(dates, md['var01'], test_vals.min() - 2,
                    alpha=0.10, color=md['color'])
    # Mark violations
    vdates = dates[md['viol01']]
    vvals = test_vals[md['viol01']]
    ax.scatter(vdates, vvals, color=md['color'], s=10, zorder=5, marker='v')
    ax.axhline(0, color='gray', linewidth=0.3)
    pct = md['n_viol01'] / T * 100
    kupiec_ok = 'Pass' if md['kupiec_p01'] > 0.05 else 'Fail'
    ax.set_title(f'{title}\nViol: {md["n_viol01"]}/{T} ({pct:.1f}%)  Kupiec: {kupiec_ok}',
                 fontweight='bold', fontsize=8)
    if j >= 2:
        ax.set_xlabel('Date')
    if j % 2 == 0:
        ax.set_ylabel('Return (%)')

plt.tight_layout()
save_fig('ch10_btc_var_multimodel_1pct')

# =============================================================================
# 2. DIEBOLD-MARIANO TEST
# =============================================================================
print("\n" + "=" * 70)
print("PART 2: DIEBOLD-MARIANO TEST")
print("=" * 70)

# Compare squared forecast errors from Normal vs Student-t GARCH volatility
# Volatility forecast: sigma_t from each model
# Loss function: squared error of volatility forecast vs realized |r_t|
realized_vol = np.abs(test_vals)

e_norm = realized_vol - sigma_norm
e_t = realized_vol - sigma_t

# DM test: d_t = L(e_norm) - L(e_t) where L = squared loss
d_t = e_norm ** 2 - e_t ** 2
d_bar = np.mean(d_t)

# Newey-West variance estimate (lag = floor(T^{1/3}))
max_lag = int(np.floor(T ** (1/3)))
gamma_0 = np.var(d_t, ddof=1)
nw_var = gamma_0
for k in range(1, max_lag + 1):
    gamma_k = np.mean((d_t[k:] - d_bar) * (d_t[:-k] - d_bar))
    nw_var += 2 * (1 - k / (max_lag + 1)) * gamma_k

dm_stat = d_bar / np.sqrt(nw_var / T)
dm_pval = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

print(f"   DM statistic: {dm_stat:.4f}")
print(f"   p-value: {dm_pval:.4f}")
print(f"   Conclusion: {'Student-t significantly better' if dm_pval < 0.05 and dm_stat > 0 else 'Student-t better (not significant)' if dm_stat > 0 else 'Normal better' if dm_pval < 0.05 else 'No significant difference'}")

# =============================================================================
# 3. FORECAST METRICS BEYOND RMSE
# =============================================================================
print("\n" + "=" * 70)
print("PART 3: FORECAST METRICS BEYOND RMSE")
print("=" * 70)

# Use GARCH Normal volatility forecasts vs realized volatility
forecast_vol = sigma_norm
actual_vol = realized_vol

# RMSE
rmse_vol = np.sqrt(np.mean((actual_vol - forecast_vol) ** 2))
mae_vol = np.mean(np.abs(actual_vol - forecast_vol))

# MASE: compare to naive forecast (sigma_{t} = |r_{t-1}|)
naive_forecast = np.abs(all_ret[val_end - 1:val_end + T - 1])
mae_naive = np.mean(np.abs(actual_vol - naive_forecast))
mase = mae_vol / mae_naive if mae_naive > 0 else np.inf

# Directional Accuracy: did volatility go up/down correctly?
actual_direction = np.diff(actual_vol) > 0
forecast_direction = np.diff(forecast_vol) > 0
da = np.mean(actual_direction == forecast_direction) * 100

# Quantile Loss at 5% for VaR
ql_05 = np.mean(np.where(test_vals < var_norm_05,
                           0.05 * (test_vals - var_norm_05),
                           (1 - 0.05) * (var_norm_05 - test_vals)))
# Fix sign: QL should be positive
ql_05_vals = np.where(test_vals < var_norm_05,
                       0.05 * np.abs(test_vals - var_norm_05),
                       0.95 * np.abs(var_norm_05 - test_vals))
ql_05 = np.mean(ql_05_vals)

print(f"   Volatility Forecast (GARCH Normal):")
print(f"   RMSE:  {rmse_vol:.4f}")
print(f"   MAE:   {mae_vol:.4f}")
print(f"   MASE:  {mase:.4f} (vs naive |r_{{t-1}}|)")
print(f"   Directional Accuracy: {da:.1f}%")
print(f"   Quantile Loss (5% VaR): {ql_05:.4f}")

# =============================================================================
# 4. INNOVATION DISTRIBUTION
# =============================================================================
print("\n" + "=" * 70)
print("PART 4: INNOVATION DISTRIBUTION ANALYSIS")
print("=" * 70)

std_resid = (fit_norm.resid / fit_norm.conditional_volatility).dropna()
kurt = float(stats.kurtosis(std_resid, fisher=False))  # excess = False -> raw
skew = float(stats.skew(std_resid))
jb_stat, jb_pval = stats.jarque_bera(std_resid)

print(f"   Standardized residuals (GARCH Normal):")
print(f"   Kurtosis: {kurt:.2f} (Normal=3)")
print(f"   Skewness: {skew:.2f}")
print(f"   Jarque-Bera: {jb_stat:.2f}, p={jb_pval:.6f}")
print(f"   Student-t df (nu): {nu:.2f}")

# AIC/BIC comparison
print(f"\n   Model comparison:")
print(f"   Normal:    AIC={fit_norm.aic:.1f}, BIC={fit_norm.bic:.1f}")
print(f"   Student-t: AIC={fit_t.aic:.1f}, BIC={fit_t.bic:.1f}")
print(f"   Better by AIC: {'Student-t' if fit_t.aic < fit_norm.aic else 'Normal'}")

# =============================================================================
# 5. VAR STABILITY - companion matrix eigenvalues
# =============================================================================
print("\n" + "=" * 70)
print("PART 5: VAR STABILITY AND EIGENVALUES")
print("=" * 70)

import pandas_datareader.data as web
from statsmodels.tsa.api import VAR

# Download economic data from FRED
start_date = '2000-01-01'
end_date = '2025-01-01'

print("   Downloading economic data from FRED...")
try:
    gdp = web.DataReader('GDP', 'fred', start_date, end_date).resample('QS').last()
    unemp = web.DataReader('UNRATE', 'fred', start_date, end_date).resample('QS').last()
    fed = web.DataReader('FEDFUNDS', 'fred', start_date, end_date).resample('QS').last()
    inflation = web.DataReader('CPIAUCSL', 'fred', start_date, end_date).resample('QS').last()

    # Compute growth rates
    gdp_growth = gdp.pct_change().dropna() * 100
    inflation_rate = inflation.pct_change().dropna() * 100

    econ = pd.concat([gdp_growth, unemp, fed, inflation_rate], axis=1).dropna()
    econ.columns = ['GDP_growth', 'Unemployment', 'FedRate', 'Inflation']
    print(f"   Economic data: {len(econ)} quarterly obs")

    # Split
    train_econ_end = int(len(econ) * 0.70)
    val_econ_end = int(len(econ) * 0.85)
    train_econ = econ.iloc[:val_econ_end]

    # Fit VAR(2) on train+val
    var_model = VAR(train_econ)
    var_fit = var_model.fit(2)

    is_stable = var_fit.is_stable()

    # Build companion matrix for VAR(p)
    k = var_fit.neqs  # number of variables
    p = var_fit.k_ar  # number of lags
    coefs = np.array(var_fit.coefs)  # shape: (p, k, k)

    companion_matrix = np.zeros((k * p, k * p))
    for i in range(p):
        companion_matrix[:k, i * k:(i + 1) * k] = coefs[i]
    if p > 1:
        companion_matrix[k:k + k * (p - 1), :k * (p - 1)] = np.eye(k * (p - 1))

    eigenvalues = np.linalg.eigvals(companion_matrix)
    moduli = np.abs(eigenvalues)

    print(f"\n   VAR(2) Companion Matrix Eigenvalues:")
    for j, (ev, mod) in enumerate(zip(eigenvalues, moduli)):
        print(f"   λ_{j+1} = {ev:.4f}, |λ_{j+1}| = {mod:.4f}")
    print(f"\n   Max modulus: {max(moduli):.4f}")
    print(f"   Stable: {is_stable} (all |λ| < 1)")

    # --- Eigenvalue chart (unit circle) ---
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), color='#999999', linewidth=0.8, linestyle='--', label='Unit circle')
    ax.scatter(eigenvalues.real, eigenvalues.imag, color='#DC3545', s=60, zorder=5, edgecolors='white', linewidth=0.5)
    for j, ev in enumerate(eigenvalues):
        ax.annotate(f'|λ|={abs(ev):.3f}', (ev.real, ev.imag),
                     textcoords="offset points", xytext=(8, 5), fontsize=7, color='#333333')
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axvline(0, color='gray', linewidth=0.3)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('VAR(2) Companion Matrix Eigenvalues', fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    plt.tight_layout()
    save_fig('ch10_var_eigenvalues')

    HAS_ECON = True
except Exception as e:
    print(f"   ERROR downloading economic data: {e}")
    HAS_ECON = False

# =============================================================================
# 6. JOHANSEN COINTEGRATION TEST
# =============================================================================
if HAS_ECON:
    print("\n" + "=" * 70)
    print("PART 6: JOHANSEN COINTEGRATION TEST")
    print("=" * 70)

    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    # Test on levels (GDP, unemployment, fed rate, CPI level)
    levels = pd.concat([
        web.DataReader('GDP', 'fred', start_date, end_date).resample('QS').last(),
        web.DataReader('UNRATE', 'fred', start_date, end_date).resample('QS').last(),
        web.DataReader('FEDFUNDS', 'fred', start_date, end_date).resample('QS').last(),
        web.DataReader('CPIAUCSL', 'fred', start_date, end_date).resample('QS').last()
    ], axis=1).dropna()
    levels.columns = ['GDP', 'Unemployment', 'FedRate', 'CPI']

    johansen_result = coint_johansen(levels.iloc[:val_econ_end], det_order=0, k_ar_diff=2)

    print(f"\n   Johansen Trace Test (det_order=0, k_ar_diff=2):")
    print(f"   {'r':>3s}  {'Trace Stat':>12s}  {'5% CV':>10s}  {'Reject?':>8s}")
    for i in range(4):
        trace_stat = johansen_result.lr1[i]
        cv_5 = johansen_result.cvt[i, 1]  # 5% critical value
        reject = "Yes" if trace_stat > cv_5 else "No"
        print(f"   {i:>3d}  {trace_stat:>12.2f}  {cv_5:>10.2f}  {reject:>8s}")

    # Number of cointegrating relations
    n_coint = sum(1 for i in range(4) if johansen_result.lr1[i] > johansen_result.cvt[i, 1])
    print(f"\n   Cointegrating relations found: {n_coint}")

# =============================================================================
# 7. STRUCTURAL BREAKS (Unemployment)
# =============================================================================
print("\n" + "=" * 70)
print("PART 7: STRUCTURAL BREAKS - UNEMPLOYMENT")
print("=" * 70)

unemp_monthly = web.DataReader('UNRATE', 'fred', '2000-01-01', '2025-01-01')

# Chow test at COVID (March 2020)
break_date = '2020-03-01'
pre = unemp_monthly.loc[:break_date].values.flatten()
post = unemp_monthly.loc[break_date:].values.flatten()

# Simple Chow test: F-test comparing restricted vs unrestricted RSS
from numpy.linalg import lstsq

y_all = unemp_monthly.values.flatten()
n_all = len(y_all)
X_all = np.column_stack([np.ones(n_all), np.arange(n_all)])

# Restricted: single regression
beta_r, rss_r_arr, _, _ = lstsq(X_all, y_all, rcond=None)
rss_r = np.sum((y_all - X_all @ beta_r) ** 2)

# Unrestricted: separate regressions
n_pre = len(pre)
n_post = len(post)
X_pre = np.column_stack([np.ones(n_pre), np.arange(n_pre)])
X_post = np.column_stack([np.ones(n_post), np.arange(n_post)])
beta_pre, _, _, _ = lstsq(X_pre, pre, rcond=None)
beta_post, _, _, _ = lstsq(X_post, post, rcond=None)
rss_u = np.sum((pre - X_pre @ beta_pre) ** 2) + np.sum((post - X_post @ beta_post) ** 2)

k = 2  # number of parameters
F_chow = ((rss_r - rss_u) / k) / (rss_u / (n_all - 2 * k))
p_chow = 1 - stats.f.cdf(F_chow, k, n_all - 2 * k)

print(f"   Chow Test at {break_date} (COVID):")
print(f"   F-statistic: {F_chow:.2f}")
print(f"   p-value: {p_chow:.6f}")
print(f"   Conclusion: {'Structural break confirmed' if p_chow < 0.05 else 'No break'}")

# CUSUM-like: rolling mean shift detection
rolling_mean = pd.Series(y_all).rolling(24).mean().dropna().values
overall_mean = np.mean(y_all)
cusum = np.cumsum(y_all - overall_mean) / np.std(y_all)
print(f"   CUSUM max deviation: {np.max(np.abs(cusum)):.2f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: VALUES FOR SLIDE UPDATES")
print("=" * 70)

print("\n--- Multi-Model VaR Comparison (5%) ---")
print(f"   {'Model':15s} {'Viol':>5s} {'Rate':>7s} {'Kupiec p':>10s} {'Chr. p':>10s} {'AIC':>10s} {'Basel':>8s}")
for name in model_list:
    md = models_data[name]
    pct = md['n_viol05'] / T * 100
    aic_str = f"{md['aic']:.1f}" if md['aic'] is not None else "---"
    # Basel zone (scaled to 250 days)
    viol_250 = md['n_viol05'] / T * 250
    if viol_250 <= 4:
        basel = 'Green'
    elif viol_250 <= 9:
        basel = 'Yellow'
    else:
        basel = 'Red'
    print(f"   {name:15s} {md['n_viol05']:5d} {pct:6.1f}% {md['kupiec_p05']:10.4f} {md['chr_p05']:10.4f} {aic_str:>10s} {basel:>8s}")

print(f"""
--- VaR Slide ---
VaR formula: VaR_{{t+1}}^α = μ + σ_{{t+1}} · z_α
Normal z_0.05 = {z_05_norm:.4f}, Student-t z_0.05 = {z_05_t:.4f} (ν={nu:.1f})

--- DM Test Slide ---
Normal vs Student-t GARCH (volatility MSE):
  DM = {dm_stat:.2f}, p = {dm_pval:.4f}

--- Metrics Slide ---
MASE = {mase:.2f}, DA = {da:.1f}%, QL(5%) = {ql_05:.4f}

--- Innovation Distribution Slide ---
Kurtosis = {kurt:.2f}, Skewness = {skew:.2f}
Student-t ν = {nu:.2f}
AIC: Normal={fit_norm.aic:.1f}, Student-t={fit_t.aic:.1f}, GJR-t={fit_gjr.aic:.1f}

--- Chow Test ---
F = {F_chow:.2f}, p = {p_chow:.6f}
""")

if HAS_ECON:
    print(f"--- VAR Eigenvalues ---")
    for j, ev in enumerate(eigenvalues):
        print(f"  λ_{j+1}: |{abs(ev):.4f}|")
    print(f"  Max: {max(moduli):.4f}, Stable: {is_stable}")
    print(f"\n--- Johansen ---")
    print(f"  Cointegrating relations: {n_coint}")

print("\n" + "=" * 70)
print("DONE - Charts saved to charts/ directory")
print("=" * 70)
