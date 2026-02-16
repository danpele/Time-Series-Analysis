#!/usr/bin/env python3
"""
Quantlet:     TSA_ch2_case_study
Description:  Complete ARMA modeling of annual sunspot data using Box-Jenkins methodology
Keywords:     ARMA, AR, MA, Box-Jenkins, ACF, PACF, Ljung-Box, AIC, BIC, forecasting, sunspots
Author:       Daniel Traian PELE
Date:         2025
Data:         Annual sunspot numbers (1700-2008) from statsmodels
Output:       Figures (ch2_case_raw_data, ch2_case_acf_pacf, ch2_case_model_comparison,
              ch2_case_diagnostics, ch2_case_forecast), model summary statistics
See also:     TSA_ch3_arima_case, TSA_ch4_sarima_case
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.datasets import sunspots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
COLORS = ['#1A3A6E', '#CD0000', '#2E7D32', '#B5853F', '#E67E22', '#8E44AD']
BLUE, RED, GREEN, BROWN, ORANGE, PURPLE = COLORS

chart_dir = '../../charts/'
np.random.seed(42)

plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
    'axes.grid': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.facecolor': 'none',
    'legend.framealpha': 0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.2,
})


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def save_fig(fig, name):
    """Save figure as both PNG and PDF with transparent background."""
    fig.savefig(chart_dir + name + '.png', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(chart_dir + name + '.pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.png + {name}.pdf")


def style_ax(ax):
    """Apply consistent styling to an axes object."""
    ax.patch.set_alpha(0)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_legend_bottom(ax, ncol=3):
    """Place legend outside bottom center."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=ncol, frameon=False)


def stem_plot(ax, lags, values, color=None, conf=None):
    """Custom stem plot for ACF/PACF with optional confidence bands."""
    if color is None:
        color = BLUE
    markerline, stemlines, baseline = ax.stem(lags, values, linefmt='-',
                                               markerfmt='o', basefmt='-')
    plt.setp(stemlines, color=color, linewidth=1.2)
    plt.setp(markerline, color=color, markersize=4)
    plt.setp(baseline, color='black', linewidth=0.5)
    if conf is not None:
        ax.axhline(conf, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.axhline(-conf, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.axhspan(-conf, conf, alpha=0.06, color='gray')


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 72)
print("  ARMA CASE STUDY: Annual Sunspot Numbers (Box-Jenkins Methodology)")
print("=" * 72)

data = sunspots.load_pandas().data
y = data['SUNACTIVITY'].values
years = data['YEAR'].values.astype(int)
n = len(y)

print(f"\n  Dataset: Annual sunspot numbers")
print(f"  Period:  {int(years[0])} -- {int(years[-1])}")
print(f"  Observations: {n}")
print(f"  Mean:    {np.mean(y):.2f}")
print(f"  Std:     {np.std(y):.2f}")
print(f"  Min:     {np.min(y):.1f}   Max: {np.max(y):.1f}")

# =============================================================================
# 2. FIGURE 1 -- RAW TIME SERIES
# =============================================================================
print("\n--- Step 1: Visual inspection of the series ---")

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

ax.plot(years, y, color=BLUE, linewidth=0.9, label='Annual sunspot number')
ax.fill_between(years, 0, y, alpha=0.10, color=BLUE)
ax.set_xlabel('Year')
ax.set_ylabel('Number of sunspots')
ax.set_title('Sunspot Activity (1700--2008)', fontweight='bold')
style_ax(ax)
add_legend_bottom(ax, ncol=1)

plt.tight_layout()
save_fig(fig, 'ch2_case_raw_data')

# =============================================================================
# 3. FIGURE 2 -- ACF AND PACF
# =============================================================================
print("\n--- Step 2: Identification via ACF/PACF ---")

nlags = 30
acf_vals = acf(y, nlags=nlags, fft=True)
pacf_vals = pacf(y, nlags=nlags, method='ywm')
conf_bound = 1.96 / np.sqrt(n)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
fig.patch.set_alpha(0)

# ACF
stem_plot(axes[0], np.arange(nlags + 1), acf_vals, color=BLUE, conf=conf_bound)
axes[0].set_title('Sample ACF: damped sinusoidal decay', fontweight='bold')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('Autocorrelation')
style_ax(axes[0])

# PACF
stem_plot(axes[1], np.arange(nlags + 1), pacf_vals, color=GREEN, conf=conf_bound)
axes[1].set_title('Sample PACF: significant spikes at lags 1, 2, 9', fontweight='bold')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Partial autocorrelation')
style_ax(axes[1])

fig.suptitle('ACF/PACF Analysis of Sunspot Data', fontweight='bold', y=1.02)
plt.tight_layout()
save_fig(fig, 'ch2_case_acf_pacf')

print(f"  ACF at lag 1:  {acf_vals[1]:.4f}")
print(f"  ACF at lag 2:  {acf_vals[2]:.4f}")
print(f"  ACF at lag 11: {acf_vals[11]:.4f}")
print(f"  PACF at lag 1: {pacf_vals[1]:.4f}")
print(f"  PACF at lag 2: {pacf_vals[2]:.4f}")
print(f"  PACF at lag 9: {pacf_vals[9]:.4f}")
print("  Interpretation: sinusoidal ACF decay + PACF spikes at lags 1,2,9")
print("                  suggest AR(2), AR(9), or ARMA(2,q) candidates.")

# =============================================================================
# 4. FIT CANDIDATE MODELS AND COMPARE AIC/BIC
# =============================================================================
print("\n--- Step 3: Model estimation and selection ---")

candidate_models = {
    'AR(2)':     (2, 0, 0),
    'AR(9)':     (9, 0, 0),
    'ARMA(2,1)': (2, 0, 1),
    'ARMA(2,2)': (2, 0, 2),
}

results = {}
for name, order in candidate_models.items():
    try:
        model = ARIMA(y, order=order)
        fit = model.fit()
        results[name] = {
            'fit': fit,
            'AIC': fit.aic,
            'BIC': fit.bic,
            'LogL': fit.llf,
            'nparams': fit.df_model + 1,  # +1 for sigma^2
        }
    except Exception as e:
        print(f"  WARNING: {name} failed to converge -- {e}")

# Build comparison DataFrame
comparison_rows = []
for name in candidate_models:
    if name in results:
        r = results[name]
        comparison_rows.append({
            'Model': name,
            'AIC': r['AIC'],
            'BIC': r['BIC'],
            'Log-Lik': r['LogL'],
            'Params': r['nparams'],
        })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df = comparison_df.sort_values('AIC').reset_index(drop=True)

print("\n  Model Comparison Table:")
print("  " + "-" * 62)
print(f"  {'Model':<14} {'AIC':>10} {'BIC':>10} {'Log-Lik':>12} {'Params':>8}")
print("  " + "-" * 62)
for _, row in comparison_df.iterrows():
    marker = " <-- best AIC" if row['AIC'] == comparison_df['AIC'].min() else ""
    print(f"  {row['Model']:<14} {row['AIC']:>10.2f} {row['BIC']:>10.2f} "
          f"{row['Log-Lik']:>12.2f} {int(row['Params']):>8}{marker}")
print("  " + "-" * 62)

best_name = comparison_df.iloc[0]['Model']
best_fit = results[best_name]['fit']
print(f"\n  Selected model (lowest AIC): {best_name}")

# FIGURE 3 -- MODEL COMPARISON BAR CHART
fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

model_names = comparison_df['Model'].tolist()
aic_values = comparison_df['AIC'].tolist()
bic_values = comparison_df['BIC'].tolist()

x = np.arange(len(model_names))
width = 0.35

bars_aic = ax.bar(x - width / 2, aic_values, width, color=BLUE, alpha=0.85, label='AIC')
bars_bic = ax.bar(x + width / 2, bic_values, width, color=RED, alpha=0.85, label='BIC')

# Highlight the best model (lowest AIC)
best_idx = 0  # already sorted by AIC ascending
bars_aic[best_idx].set_edgecolor(GREEN)
bars_aic[best_idx].set_linewidth(2.5)

ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=20, ha='right')
ax.set_ylabel('Information criterion value')
ax.set_title('Model Comparison: AIC and BIC', fontweight='bold')

ax.annotate(f'Best: {model_names[best_idx]}',
            xy=(best_idx - width / 2, aic_values[best_idx]),
            xytext=(best_idx + 1.2, aic_values[best_idx] - 15),
            arrowprops=dict(arrowstyle='->', color=GREEN, linewidth=1.5),
            fontsize=10, color=GREEN, fontweight='bold')

style_ax(ax)
add_legend_bottom(ax, ncol=2)

plt.tight_layout()
save_fig(fig, 'ch2_case_model_comparison')

# =============================================================================
# 5. BEST MODEL SUMMARY
# =============================================================================
print("\n--- Step 4: Best model summary ---")
print(best_fit.summary())

# =============================================================================
# 6. FIGURE 4 -- RESIDUAL DIAGNOSTICS
# =============================================================================
print("\n--- Step 5: Residual diagnostics ---")

residuals = best_fit.resid

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
fig.patch.set_alpha(0)

# Panel (a): Residual time series
ax = axes[0, 0]
ax.plot(years, residuals, color=BLUE, linewidth=0.7, label='Residuals')
ax.axhline(0, color='black', linewidth=0.5)
sigma_r = np.std(residuals)
ax.axhline(2 * sigma_r, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
ax.axhline(-2 * sigma_r, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
ax.set_title('(a) Standardized residuals', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Residual')
style_ax(ax)
add_legend_bottom(ax, ncol=1)

# Panel (b): ACF of residuals
ax = axes[0, 1]
resid_acf = acf(residuals, nlags=20, fft=True)
stem_plot(ax, np.arange(21), resid_acf, color=BLUE, conf=conf_bound)
ax.set_title('(b) ACF of residuals', fontweight='bold')
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
style_ax(ax)

# Panel (c): Histogram + Normal density
ax = axes[1, 0]
ax.hist(residuals, bins=30, density=True, color=BLUE, alpha=0.65,
        edgecolor='white', label='Residuals')
xr = np.linspace(residuals.min(), residuals.max(), 200)
ax.plot(xr, stats.norm.pdf(xr, np.mean(residuals), np.std(residuals)),
        color=RED, linewidth=1.8, label='Normal density')
ax.set_title('(c) Residual distribution', fontweight='bold')
ax.set_xlabel('Residual value')
ax.set_ylabel('Density')
style_ax(ax)
add_legend_bottom(ax, ncol=2)

# Panel (d): Q-Q plot
ax = axes[1, 1]
(osm, osr), (slope, intercept, r_val) = stats.probplot(residuals, dist="norm")
ax.plot(osm, osr, 'o', color=BLUE, markersize=3, alpha=0.5, label='Sample quantiles')
ax.plot(osm, slope * osm + intercept, '-', color=RED, linewidth=1.5, label='Reference line')
ax.set_title('(d) Normal Q-Q plot', fontweight='bold')
ax.set_xlabel('Theoretical quantiles')
ax.set_ylabel('Sample quantiles')
style_ax(ax)
add_legend_bottom(ax, ncol=2)

fig.suptitle(f'Residual Diagnostics: {best_name}', fontweight='bold', y=1.02)
plt.tight_layout()
save_fig(fig, 'ch2_case_diagnostics')

# Ljung-Box test
lb_lags = [5, 10, 15, 20]
lb_test = acorr_ljungbox(residuals, lags=lb_lags, return_df=True)

print("\n  Ljung-Box Test for Residual Autocorrelation:")
print("  " + "-" * 48)
print(f"  {'Lag':>6}  {'Q-stat':>10}  {'p-value':>10}  {'Decision':>14}")
print("  " + "-" * 48)
for lag_val in lb_lags:
    q_stat = lb_test.loc[lag_val, 'lb_stat']
    p_val = lb_test.loc[lag_val, 'lb_pvalue']
    decision = "Adequate" if p_val > 0.05 else "Reject H0"
    print(f"  {lag_val:>6}  {q_stat:>10.2f}  {p_val:>10.4f}  {decision:>14}")
print("  " + "-" * 48)

# Normality test
jb_stat, jb_pval = stats.jarque_bera(residuals)
print(f"\n  Jarque-Bera normality test: stat = {jb_stat:.2f}, p-value = {jb_pval:.4f}")
if jb_pval > 0.05:
    print("  -> Cannot reject normality at 5% level.")
else:
    print("  -> Residuals deviate from normality (common for sunspot data).")

# =============================================================================
# 7. FIGURE 5 -- FORECAST WITH CONFIDENCE INTERVALS
# =============================================================================
print("\n--- Step 6: Forecasting ---")

h = 30  # forecast horizon
forecast_result = best_fit.get_forecast(steps=h)
forecast_mean = forecast_result.predicted_mean
forecast_ci_raw = forecast_result.conf_int(alpha=0.05)

# Handle both DataFrame and ndarray returns from conf_int()
if hasattr(forecast_ci_raw, 'iloc'):
    forecast_ci_lower = forecast_ci_raw.iloc[:, 0].values
    forecast_ci_upper = forecast_ci_raw.iloc[:, 1].values
else:
    forecast_ci_lower = forecast_ci_raw[:, 0]
    forecast_ci_upper = forecast_ci_raw[:, 1]

forecast_years = np.arange(int(years[-1]) + 1, int(years[-1]) + 1 + h)

# Also refit on training sample (exclude last 30 obs) for out-of-sample plot
train_end = n - 30
y_train = y[:train_end]
y_test = y[train_end:]
years_train = years[:train_end]
years_test = years[train_end:]

order_best = candidate_models[best_name]
model_oos = ARIMA(y_train, order=order_best).fit()
oos_forecast = model_oos.get_forecast(steps=30)
oos_mean = oos_forecast.predicted_mean
oos_ci_raw = oos_forecast.conf_int(alpha=0.05)

if hasattr(oos_ci_raw, 'iloc'):
    oos_ci_lower = oos_ci_raw.iloc[:, 0].values
    oos_ci_upper = oos_ci_raw.iloc[:, 1].values
else:
    oos_ci_lower = oos_ci_raw[:, 0]
    oos_ci_upper = oos_ci_raw[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
fig.patch.set_alpha(0)

# Left: In-sample fit + out-of-sample forecast from full model
ax = axes[0]
ax.patch.set_alpha(0)
tail = 80
ax.plot(years[-tail:], y[-tail:], color=BLUE, linewidth=1.0, label='Historical data')
ax.plot(forecast_years, forecast_mean, color=RED, linewidth=1.5,
        linestyle='--', label=f'{best_name} forecast')
ax.fill_between(forecast_years,
                np.maximum(forecast_ci_lower, 0),
                forecast_ci_upper,
                color=RED, alpha=0.12, label='95% confidence interval')
ax.axvline(years[-1], color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_title(f'(a) {h}-year ahead forecast ({best_name})', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Sunspot number')
style_ax(ax)
add_legend_bottom(ax, ncol=3)

# Right: Out-of-sample evaluation (train/test split)
ax = axes[1]
ax.patch.set_alpha(0)
ax.plot(years_train[-50:], y_train[-50:], color=BLUE, linewidth=1.0, label='Training data')
ax.plot(years_test, y_test, color='gray', linewidth=1.0, alpha=0.8, label='Actual (test)')
ax.plot(years_test, oos_mean, color=RED, linewidth=1.5, linestyle='--', label='Forecast')
ax.fill_between(years_test,
                np.maximum(oos_ci_lower, 0),
                oos_ci_upper,
                color=RED, alpha=0.12, label='95% CI')
ax.axvline(years_train[-1], color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_title('(b) Out-of-sample evaluation (last 30 years held out)', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Sunspot number')
style_ax(ax)
add_legend_bottom(ax, ncol=4)

plt.tight_layout()
save_fig(fig, 'ch2_case_forecast')

# Forecast accuracy on test set
mae = np.mean(np.abs(y_test - oos_mean))
rmse = np.sqrt(np.mean((y_test - oos_mean) ** 2))
mape = np.mean(np.abs((y_test - oos_mean) / np.where(y_test == 0, 1, y_test))) * 100

print(f"\n  Out-of-sample forecast accuracy ({best_name}, h=1..30):")
print(f"    MAE  = {mae:.2f}")
print(f"    RMSE = {rmse:.2f}")
print(f"    MAPE = {mape:.1f}%")

# =============================================================================
# 8. ROLLING WINDOW OUT-OF-SAMPLE EVALUATION
# =============================================================================
print("\n--- Step 7: Rolling window out-of-sample evaluation ---")

window_size = 200
n_forecasts = n - window_size
rolling_forecasts = np.full(n_forecasts, np.nan)
rolling_actual = y[window_size:]
n_failures = 0

for i in range(n_forecasts):
    y_window = y[i:i + window_size]
    try:
        model_roll = ARIMA(y_window, order=order_best)
        fit_roll = model_roll.fit()
        fcast = fit_roll.get_forecast(steps=1)
        pred_mean = fcast.predicted_mean
        # Handle both Series/array returns
        if hasattr(pred_mean, 'iloc'):
            rolling_forecasts[i] = pred_mean.iloc[0]
        elif hasattr(pred_mean, '__len__'):
            rolling_forecasts[i] = pred_mean[0]
        else:
            rolling_forecasts[i] = float(pred_mean)
    except Exception as e:
        n_failures += 1
        # If fitting fails for a window, carry forward last known forecast
        if i > 0 and not np.isnan(rolling_forecasts[i - 1]):
            rolling_forecasts[i] = rolling_forecasts[i - 1]

# Remove any remaining NaN values for metrics
valid = ~np.isnan(rolling_forecasts)
rolling_errors = rolling_actual[valid] - rolling_forecasts[valid]
rolling_mae = np.mean(np.abs(rolling_errors))
rolling_rmse = np.sqrt(np.mean(rolling_errors ** 2))

print(f"\n  Rolling 1-step-ahead forecast ({best_name}, window={window_size}):")
print(f"    Number of forecasts: {np.sum(valid)}  (failures: {n_failures})")
print(f"    MAE  = {rolling_mae:.2f}")
print(f"    RMSE = {rolling_rmse:.2f}")

# =============================================================================
# 9. COMPREHENSIVE COMPARISON TABLE
# =============================================================================
print("\n--- Step 8: Final model comparison summary ---")

print("\n  " + "=" * 80)
print(f"  {'Model':<14} {'AIC':>9} {'BIC':>9} {'Log-Lik':>11} "
      f"{'Params':>7} {'LB(10) p':>10} {'LB(20) p':>10}")
print("  " + "=" * 80)

for name in candidate_models:
    if name not in results:
        continue
    r = results[name]
    fit_i = r['fit']
    resid_i = fit_i.resid
    lb_i = acorr_ljungbox(resid_i, lags=[10, 20], return_df=True)
    p10 = lb_i.loc[10, 'lb_pvalue']
    p20 = lb_i.loc[20, 'lb_pvalue']
    best_marker = "  ***" if name == best_name else ""
    print(f"  {name:<14} {r['AIC']:>9.2f} {r['BIC']:>9.2f} {r['LogL']:>11.2f} "
          f"{r['nparams']:>7d} {p10:>10.4f} {p20:>10.4f}{best_marker}")

print("  " + "=" * 80)
print(f"  *** = selected model ({best_name})\n")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 72)
print("  ANALYSIS COMPLETE")
print("=" * 72)
print(f"""
  Box-Jenkins workflow applied to annual sunspot data (1700--2008):

    1. Identification:  ACF shows damped sinusoidal decay; PACF shows
                        significant spikes at lags 1, 2, and 9.
    2. Candidates:      AR(2), AR(9), ARMA(2,1), ARMA(2,2)
    3. Selection:       {best_name} chosen by AIC = {results[best_name]['AIC']:.2f}
    4. Diagnostics:     Ljung-Box tests performed at lags 5, 10, 15, 20.
    5. Forecasting:     {h}-year horizon with 95% confidence intervals.
    6. Rolling eval:    1-step-ahead RMSE = {rolling_rmse:.2f} (window = {window_size})

  Output figures saved to {chart_dir}:
    - ch2_case_raw_data.png / .pdf
    - ch2_case_acf_pacf.png / .pdf
    - ch2_case_model_comparison.png / .pdf
    - ch2_case_diagnostics.png / .pdf
    - ch2_case_forecast.png / .pdf
""")
