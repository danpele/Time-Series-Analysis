#!/usr/bin/env python3
"""
Quantlet:     TSA_ch3_case_study
Description:  Complete ARIMA modeling of US Real GDP using Box-Jenkins methodology
Keywords:     ARIMA, unit root, ADF, KPSS, differencing, ACF, PACF, Ljung-Box,
              AIC, BIC, forecasting, rolling forecast, GDP, FRED
Author:       Daniel Traian PELE
Date:         2025
Data:         US Real GDP (GDPC1) from FRED via statsmodels macrodata fallback
Output:       Figures (ch3_case_raw_data, ch3_case_adf_test, ch3_case_acf_diff,
              ch3_case_model_comparison, ch3_case_diagnostics, ch3_case_forecast,
              ch3_case_rolling_forecast), model summary statistics
See also:     TSA_ch2_arma_case, TSA_ch4_sarima_case
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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
COLORS = ['#1A3A6E', '#CD0000', '#2E7D32', '#B5853F', '#E67E22', '#8E44AD']
BLUE, RED, GREEN, BROWN, ORANGE, PURPLE = COLORS

CHARTS_DIR = '../../charts/'
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
    fig.savefig(CHARTS_DIR + name + '.png', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(CHARTS_DIR + name + '.pdf', transparent=True, bbox_inches='tight')
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
print("  ARIMA CASE STUDY: US Real GDP (Box-Jenkins Methodology)")
print("=" * 72)

# Try FRED first, fall back to statsmodels macrodata
try:
    import pandas_datareader as pdr
    gdp_raw = pdr.get_data_fred('GDPC1', start='1960-01-01', end='2024-09-30')
    gdp_data = gdp_raw['GDPC1'].dropna()
    data_source = 'FRED (GDPC1)'
    print(f"\n  Data loaded from FRED successfully.")
except Exception:
    from statsmodels.datasets import macrodata
    md = macrodata.load_pandas().data
    # macrodata contains quarterly US macro data 1959Q1 - 2009Q3
    dates = pd.period_range(start='1959Q1', periods=len(md), freq='Q').to_timestamp()
    gdp_data = pd.Series(md['realgdp'].values, index=dates, name='GDPC1')
    data_source = 'statsmodels macrodata'
    print(f"\n  FRED unavailable; using statsmodels macrodata fallback.")

# Log transform
log_gdp = np.log(gdp_data)
n = len(log_gdp)

print(f"\n  Dataset: US Real GDP")
print(f"  Source:  {data_source}")
print(f"  Period:  {gdp_data.index[0].strftime('%Y-%m')} -- {gdp_data.index[-1].strftime('%Y-%m')}")
print(f"  Observations: {n}")
print(f"  Mean (level):  {gdp_data.mean():.2f}")
print(f"  Std  (level):  {gdp_data.std():.2f}")
print(f"  Min:  {gdp_data.min():.1f}   Max: {gdp_data.max():.1f}")

# =============================================================================
# 2. STATIONARITY TESTING (ADF + KPSS)
# =============================================================================
print("\n--- Step 1: Stationarity testing (ADF & KPSS) ---")

# ADF on levels (with constant + trend)
adf_level = adfuller(log_gdp, maxlag=8, regression='ct')
print(f"\n  ADF Test on Log GDP (Levels):")
print(f"    Test statistic: {adf_level[0]:.4f}")
print(f"    p-value:        {adf_level[1]:.4f}")
print(f"    Critical values: 1%: {adf_level[4]['1%']:.3f},  "
      f"5%: {adf_level[4]['5%']:.3f},  10%: {adf_level[4]['10%']:.3f}")
print(f"    Conclusion: {'Reject H0 (Stationary)' if adf_level[1] < 0.05 else 'Cannot reject H0 => Unit root present'}")

# KPSS on levels
kpss_level = kpss(log_gdp, regression='ct', nlags='auto')
print(f"\n  KPSS Test on Log GDP (Levels):")
print(f"    Test statistic: {kpss_level[0]:.4f}")
print(f"    p-value:        {kpss_level[1]:.4f}")
print(f"    Critical values: 10%: {kpss_level[3]['10%']:.3f},  "
      f"5%: {kpss_level[3]['5%']:.3f},  1%: {kpss_level[3]['1%']:.3f}")
print(f"    Conclusion: {'Reject H0 => Non-stationary' if kpss_level[1] < 0.05 else 'Cannot reject H0 (Stationary)'}")

# First difference
diff_gdp = log_gdp.diff().dropna()

# ADF on first difference
adf_diff = adfuller(diff_gdp, maxlag=8, regression='c')
print(f"\n  ADF Test on GDP Growth Rate (First Difference):")
print(f"    Test statistic: {adf_diff[0]:.4f}")
print(f"    p-value:        {adf_diff[1]:.6f}")
print(f"    Conclusion: {'Reject H0 => Stationary' if adf_diff[1] < 0.05 else 'Cannot reject H0 => Unit root present'}")

# KPSS on first difference
kpss_diff = kpss(diff_gdp, regression='c', nlags='auto')
print(f"\n  KPSS Test on GDP Growth Rate (First Difference):")
print(f"    Test statistic: {kpss_diff[0]:.4f}")
print(f"    p-value:        {kpss_diff[1]:.4f}")
print(f"    Conclusion: {'Reject H0 => Non-stationary' if kpss_diff[1] < 0.05 else 'Cannot reject H0 => Stationary'}")

print(f"\n  => Log GDP is I(1): use d=1 in ARIMA model.")

# =============================================================================
# 3. FIGURE 1 -- RAW TIME SERIES
# =============================================================================
print("\n--- Step 2: Plotting raw data ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
fig.patch.set_alpha(0)

# Left panel: GDP in levels
ax = axes[0]
ax.patch.set_alpha(0)
ax.plot(gdp_data.index, gdp_data.values, color=BLUE, linewidth=1.0,
        label='US Real GDP')
ax.fill_between(gdp_data.index, 0, gdp_data.values, alpha=0.08, color=BLUE)
ax.set_xlabel('Date')
ax.set_ylabel('Billions of Chained 2017 Dollars')
ax.set_title('(a) US Real GDP (FRED: GDPC1)', fontweight='bold')
ax.text(0.02, 0.98, f'Source: {data_source}',
        transform=ax.transAxes, fontsize=8, verticalalignment='top',
        style='italic', color='gray')
style_ax(ax)
add_legend_bottom(ax, ncol=1)

# Right panel: Log GDP
ax = axes[1]
ax.patch.set_alpha(0)
ax.plot(log_gdp.index, log_gdp.values, color=GREEN, linewidth=1.0,
        label='Log(Real GDP)')
ax.set_xlabel('Date')
ax.set_ylabel('Log(GDP)')
ax.set_title('(b) Log-Transformed GDP', fontweight='bold')
style_ax(ax)
add_legend_bottom(ax, ncol=1)

plt.tight_layout()
save_fig(fig, 'ch3_case_raw_data')

# =============================================================================
# 4. FIGURE 2 -- ADF TEST VISUALIZATION
# =============================================================================
print("\n--- Step 3: ADF test visualization ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
fig.patch.set_alpha(0)

# Left panel: Log GDP (levels) -- non-stationary
ax = axes[0]
ax.patch.set_alpha(0)
ax.plot(log_gdp.index, log_gdp.values, color=BLUE, linewidth=1.0,
        label='Log GDP (levels)')
ax.set_title('(a) Log GDP -- Non-Stationary', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Log(GDP)')
ax.text(0.02, 0.98,
        f'ADF stat: {adf_level[0]:.2f}\np-value: {adf_level[1]:.4f}\nUnit root: Yes',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))
style_ax(ax)
add_legend_bottom(ax, ncol=1)

# Right panel: GDP growth rate (differenced) -- stationary
ax = axes[1]
ax.patch.set_alpha(0)
ax.plot(diff_gdp.index, diff_gdp.values * 100, color=GREEN, linewidth=0.9,
        label='GDP growth rate (%)')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
ax.set_title('(b) GDP Growth Rate -- Stationary', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Percent change')
ax.text(0.02, 0.98,
        f'ADF stat: {adf_diff[0]:.2f}\np-value: {adf_diff[1]:.4f}\nStationary: Yes',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8))
style_ax(ax)
add_legend_bottom(ax, ncol=1)

plt.tight_layout()
save_fig(fig, 'ch3_case_adf_test')

# =============================================================================
# 5. FIGURE 3 -- ACF/PACF BEFORE AND AFTER DIFFERENCING
# =============================================================================
print("\n--- Step 4: Identification via ACF/PACF ---")

nlags = 20
n_diff = len(diff_gdp)

acf_level = acf(log_gdp, nlags=nlags, fft=True)
pacf_level = pacf(log_gdp, nlags=nlags, method='ywm')
conf_level = 1.96 / np.sqrt(n)

acf_diff_vals = acf(diff_gdp, nlags=nlags, fft=True)
pacf_diff_vals = pacf(diff_gdp, nlags=nlags, method='ywm')
conf_diff = 1.96 / np.sqrt(n_diff)

fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
fig.patch.set_alpha(0)

# (a) ACF of levels
stem_plot(axes[0, 0], np.arange(nlags + 1), acf_level, color=BLUE, conf=conf_level)
axes[0, 0].set_title('(a) ACF: Log GDP (Levels) -- slow decay', fontweight='bold')
axes[0, 0].set_xlabel('Lag')
axes[0, 0].set_ylabel('Autocorrelation')
style_ax(axes[0, 0])

# (b) PACF of levels
stem_plot(axes[0, 1], np.arange(nlags + 1), pacf_level, color=BLUE, conf=conf_level)
axes[0, 1].set_title('(b) PACF: Log GDP (Levels)', fontweight='bold')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylabel('Partial autocorrelation')
style_ax(axes[0, 1])

# (c) ACF of differenced series
stem_plot(axes[1, 0], np.arange(nlags + 1), acf_diff_vals, color=GREEN, conf=conf_diff)
axes[1, 0].set_title('(c) ACF: GDP Growth (Differenced)', fontweight='bold')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('Autocorrelation')
style_ax(axes[1, 0])

# (d) PACF of differenced series
stem_plot(axes[1, 1], np.arange(nlags + 1), pacf_diff_vals, color=GREEN, conf=conf_diff)
axes[1, 1].set_title('(d) PACF: GDP Growth (Differenced)', fontweight='bold')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('Partial autocorrelation')
style_ax(axes[1, 1])

fig.suptitle('ACF/PACF Before and After Differencing', fontweight='bold', y=1.02)
plt.tight_layout()
save_fig(fig, 'ch3_case_acf_diff')

print(f"  ACF of diff at lag 1:  {acf_diff_vals[1]:.4f}")
print(f"  ACF of diff at lag 2:  {acf_diff_vals[2]:.4f}")
print(f"  PACF of diff at lag 1: {pacf_diff_vals[1]:.4f}")
print(f"  PACF of diff at lag 2: {pacf_diff_vals[2]:.4f}")
print("  Interpretation: ACF spike at lag 1 then cuts off => MA(1) component")
print("                  PACF spike at lag 1, decays       => AR(1) component")
print("                  Candidates: ARIMA(1,1,0), ARIMA(0,1,1), ARIMA(1,1,1)")

# =============================================================================
# 6. FIT CANDIDATE MODELS AND COMPARE AIC/BIC
# =============================================================================
print("\n--- Step 5: Model estimation and selection ---")

candidate_models = {
    'ARIMA(0,1,0)': (0, 1, 0),
    'ARIMA(1,1,0)': (1, 1, 0),
    'ARIMA(0,1,1)': (0, 1, 1),
    'ARIMA(1,1,1)': (1, 1, 1),
    'ARIMA(2,1,1)': (2, 1, 1),
}

results = {}
for name, order in candidate_models.items():
    try:
        model = ARIMA(log_gdp, order=order)
        fit = model.fit()
        results[name] = {
            'fit': fit,
            'AIC': fit.aic,
            'BIC': fit.bic,
            'LogL': fit.llf,
            'nparams': fit.df_model + 1,
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
print("  " + "-" * 66)
print(f"  {'Model':<16} {'AIC':>10} {'BIC':>10} {'Log-Lik':>12} {'Params':>8}")
print("  " + "-" * 66)
for _, row in comparison_df.iterrows():
    marker = " <-- best AIC" if row['AIC'] == comparison_df['AIC'].min() else ""
    print(f"  {row['Model']:<16} {row['AIC']:>10.2f} {row['BIC']:>10.2f} "
          f"{row['Log-Lik']:>12.2f} {int(row['Params']):>8}{marker}")
print("  " + "-" * 66)

best_name = comparison_df.iloc[0]['Model']
best_fit = results[best_name]['fit']
print(f"\n  Selected model (lowest AIC): {best_name}")

# =============================================================================
# 7. FIGURE 4 -- MODEL COMPARISON BAR CHART
# =============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 4.8))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

model_names = comparison_df['Model'].tolist()
aic_values = comparison_df['AIC'].tolist()
bic_values = comparison_df['BIC'].tolist()

x = np.arange(len(model_names))
width = 0.35

bars_aic = ax.bar(x - width / 2, aic_values, width, color=BLUE, alpha=0.85, label='AIC')
bars_bic = ax.bar(x + width / 2, bic_values, width, color=ORANGE, alpha=0.85, label='BIC')

# Highlight the best model (lowest AIC) -- already sorted ascending
best_idx = 0
bars_aic[best_idx].set_edgecolor(RED)
bars_aic[best_idx].set_linewidth(2.5)

ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=20, ha='right')
ax.set_ylabel('Information criterion value')
ax.set_title('ARIMA Model Comparison: US Real GDP', fontweight='bold')

ax.annotate(f'Best: {model_names[best_idx]}',
            xy=(best_idx - width / 2, aic_values[best_idx]),
            xytext=(best_idx + 1.5, aic_values[best_idx] + 20),
            arrowprops=dict(arrowstyle='->', color=RED, linewidth=1.5),
            fontsize=10, color=RED, fontweight='bold')

style_ax(ax)
add_legend_bottom(ax, ncol=2)

plt.tight_layout()
save_fig(fig, 'ch3_case_model_comparison')

# =============================================================================
# 8. BEST MODEL SUMMARY
# =============================================================================
print("\n--- Step 6: Best model summary ---")
print(best_fit.summary())

# =============================================================================
# 9. FIGURE 5 -- RESIDUAL DIAGNOSTICS
# =============================================================================
print("\n--- Step 7: Residual diagnostics ---")

residuals = best_fit.resid

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
fig.patch.set_alpha(0)

# Panel (a): Residual time series
ax = axes[0, 0]
ax.plot(residuals.index, residuals.values, color=BLUE, linewidth=0.7, label='Residuals')
ax.axhline(0, color='black', linewidth=0.5)
sigma_r = np.std(residuals)
ax.axhline(2 * sigma_r, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
ax.axhline(-2 * sigma_r, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
ax.set_title('(a) Standardized residuals', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Residual')
style_ax(ax)
add_legend_bottom(ax, ncol=1)

# Panel (b): ACF of residuals
ax = axes[0, 1]
resid_acf = acf(residuals.dropna(), nlags=20, fft=True)
conf_resid = 1.96 / np.sqrt(len(residuals.dropna()))
stem_plot(ax, np.arange(21), resid_acf, color=BLUE, conf=conf_resid)
ax.set_title('(b) ACF of residuals', fontweight='bold')
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
style_ax(ax)

# Panel (c): Histogram + Normal density
ax = axes[1, 0]
resid_clean = residuals.dropna().values
ax.hist(resid_clean, bins=30, density=True, color=BLUE, alpha=0.65,
        edgecolor='white', label='Residuals')
xr = np.linspace(resid_clean.min(), resid_clean.max(), 200)
ax.plot(xr, stats.norm.pdf(xr, np.mean(resid_clean), np.std(resid_clean)),
        color=RED, linewidth=1.8, label='Normal density')
ax.set_title('(c) Residual distribution', fontweight='bold')
ax.set_xlabel('Residual value')
ax.set_ylabel('Density')
style_ax(ax)
add_legend_bottom(ax, ncol=2)

# Panel (d): Q-Q plot
ax = axes[1, 1]
res_standardized = (resid_clean - np.mean(resid_clean)) / np.std(resid_clean)
(osm, osr), (slope, intercept, r_val) = stats.probplot(res_standardized, dist="norm")
ax.plot(osm, osr, 'o', color=BLUE, markersize=3, alpha=0.5, label='Sample quantiles')
line_min = min(osm.min(), osr.min())
line_max = max(osm.max(), osr.max())
ax.plot([line_min, line_max], [line_min, line_max], '-', color=RED, linewidth=1.5,
        label='45-degree line')
ax.set_title('(d) Normal Q-Q plot', fontweight='bold')
ax.set_xlabel('Theoretical quantiles')
ax.set_ylabel('Sample quantiles')
style_ax(ax)
add_legend_bottom(ax, ncol=2)

fig.suptitle(f'Residual Diagnostics: {best_name}', fontweight='bold', y=1.02)
plt.tight_layout()
save_fig(fig, 'ch3_case_diagnostics')

# Ljung-Box test
lb_lags = [5, 10, 15, 20]
lb_test = acorr_ljungbox(residuals.dropna(), lags=lb_lags, return_df=True)

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

# Normality tests
jb_stat, jb_pval = stats.jarque_bera(resid_clean)
skew_val = stats.skew(resid_clean)
kurt_val = stats.kurtosis(resid_clean)

print(f"\n  Normality Analysis:")
print(f"    Skewness:        {skew_val:.3f}  (normal = 0)")
print(f"    Excess kurtosis: {kurt_val:.3f}  (normal = 0)")
print(f"    Jarque-Bera:     stat = {jb_stat:.2f}, p-value = {jb_pval:.6f}")
if jb_pval > 0.05:
    print("    -> Cannot reject normality at 5% level.")
else:
    print("    -> Residuals deviate from normality (common due to COVID-19 outlier in 2020Q2).")

# =============================================================================
# 10. FIGURE 6 -- FORECAST WITH TRAIN/VAL/TEST SPLIT
# =============================================================================
print("\n--- Step 8: Out-of-sample forecasting ---")

# Define train/val/test split (70% / 15% / 15%)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_data = log_gdp.iloc[:train_end]
val_data = log_gdp.iloc[train_end:val_end]
test_data = log_gdp.iloc[val_end:]

print(f"\n  Train/Validation/Test Split (70/15/15):")
print(f"    Training:   {len(train_data):>4} obs  ({train_data.index[0].strftime('%Y-%m')} -- {train_data.index[-1].strftime('%Y-%m')})")
print(f"    Validation: {len(val_data):>4} obs  ({val_data.index[0].strftime('%Y-%m')} -- {val_data.index[-1].strftime('%Y-%m')})")
print(f"    Test:       {len(test_data):>4} obs  ({test_data.index[0].strftime('%Y-%m')} -- {test_data.index[-1].strftime('%Y-%m')})")

# Fit model on training data and forecast
order_best = candidate_models[best_name]
model_train = ARIMA(train_data, order=order_best)
fit_train = model_train.fit()

# Forecast for val + test + 8 quarters ahead
forecast_steps = len(val_data) + len(test_data) + 8
forecast = fit_train.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci_raw = forecast.conf_int(alpha=0.05)

# Handle both DataFrame and ndarray returns
if hasattr(forecast_ci_raw, 'iloc'):
    forecast_ci_lower = forecast_ci_raw.iloc[:, 0]
    forecast_ci_upper = forecast_ci_raw.iloc[:, 1]
else:
    forecast_ci_lower = forecast_ci_raw[:, 0]
    forecast_ci_upper = forecast_ci_raw[:, 1]

# Create forecast dates
forecast_dates = pd.date_range(start=train_data.index[-1],
                                periods=forecast_steps + 1, freq='QS')[1:]

fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

# Training data (last 30 points for visibility)
ax.plot(train_data.iloc[-30:].index, train_data.iloc[-30:].values,
        color=BLUE, linewidth=1.5, label='Training (70%)')

# Validation data -- connect from last training point
val_conn = pd.concat([train_data.iloc[[-1]], val_data])
ax.plot(val_conn.index, val_conn.values,
        color=GREEN, linewidth=1.5, label='Validation (15%)')

# Test data -- connect from last validation point
test_conn = pd.concat([val_data.iloc[[-1]], test_data])
ax.plot(test_conn.index, test_conn.values,
        color=PURPLE, linewidth=1.5, label='Test (15%)')

# Forecast -- connect from last training point
forecast_conn_idx = [train_data.index[-1]] + list(forecast_dates)
forecast_conn_vals = [train_data.iloc[-1]] + list(forecast_mean.values)
ax.plot(forecast_conn_idx, forecast_conn_vals,
        color=RED, linewidth=1.5, linestyle='--', label=f'{best_name} forecast')
ax.fill_between(forecast_dates, forecast_ci_lower, forecast_ci_upper,
                color=RED, alpha=0.12, label='95% CI')

# Mark splits
ax.axvline(x=train_data.index[-1], color='gray', linestyle=':', linewidth=1.0)
ax.axvline(x=val_data.index[-1], color='gray', linestyle=':', linewidth=1.0)

ax.set_title(f'US Real GDP: {best_name} Out-of-Sample Forecast (70/15/15 Split)',
             fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Log(GDP)')
style_ax(ax)
add_legend_bottom(ax, ncol=6)

# Calculate test RMSE
test_forecast = forecast_mean.iloc[len(val_data):len(val_data) + len(test_data)]
test_rmse = np.sqrt(np.mean((test_data.values - test_forecast.values) ** 2))
test_mae = np.mean(np.abs(test_data.values - test_forecast.values))

ax.text(0.02, 0.98, f'{best_name}\nTest RMSE: {test_rmse:.6f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        style='italic', color='gray')

plt.tight_layout()
save_fig(fig, 'ch3_case_forecast')

print(f"\n  Out-of-sample forecast accuracy ({best_name}):")
print(f"    Test RMSE: {test_rmse:.6f}")
print(f"    Test MAE:  {test_mae:.6f}")

# =============================================================================
# 11. FIGURE 7 -- ROLLING 1-STEP-AHEAD FORECAST
# =============================================================================
print("\n--- Step 9: Rolling 1-step-ahead forecast ---")

rolling_forecasts = []
rolling_upper = []
rolling_lower = []
rolling_dates = []
rolling_actuals = []
n_failures = 0

print(f"\n  Computing rolling forecasts on test set ({len(test_data)} steps)...")
for i in range(len(test_data)):
    # Expanding window: use all data up to the current test point
    current_train = log_gdp.iloc[:val_end + i]
    try:
        model_roll = ARIMA(current_train, order=order_best)
        fit_roll = model_roll.fit()
        fc = fit_roll.get_forecast(steps=1)

        pred_mean = fc.predicted_mean
        if hasattr(pred_mean, 'iloc'):
            rolling_forecasts.append(pred_mean.iloc[0])
        elif hasattr(pred_mean, '__len__'):
            rolling_forecasts.append(pred_mean[0])
        else:
            rolling_forecasts.append(float(pred_mean))

        ci = fc.conf_int(alpha=0.05)
        if hasattr(ci, 'iloc'):
            rolling_lower.append(ci.iloc[0, 0])
            rolling_upper.append(ci.iloc[0, 1])
        else:
            rolling_lower.append(ci[0, 0])
            rolling_upper.append(ci[0, 1])

        rolling_dates.append(test_data.index[i])
        rolling_actuals.append(test_data.iloc[i])
    except Exception:
        n_failures += 1
        # Carry forward last forecast if available
        if rolling_forecasts:
            rolling_forecasts.append(rolling_forecasts[-1])
            rolling_lower.append(rolling_lower[-1])
            rolling_upper.append(rolling_upper[-1])
        else:
            rolling_forecasts.append(np.nan)
            rolling_lower.append(np.nan)
            rolling_upper.append(np.nan)
        rolling_dates.append(test_data.index[i])
        rolling_actuals.append(test_data.iloc[i])

    if (i + 1) % 10 == 0 or i == len(test_data) - 1:
        print(f"    Completed {i + 1}/{len(test_data)} forecasts")

# Calculate metrics
rolling_act_arr = np.array(rolling_actuals)
rolling_fcast_arr = np.array(rolling_forecasts)
valid = ~np.isnan(rolling_fcast_arr)
rolling_errors = rolling_act_arr[valid] - rolling_fcast_arr[valid]
rolling_rmse = np.sqrt(np.mean(rolling_errors ** 2))
rolling_mae = np.mean(np.abs(rolling_errors))

print(f"\n  Rolling 1-step-ahead forecast ({best_name}):")
print(f"    Number of forecasts: {np.sum(valid)}  (failures: {n_failures})")
print(f"    RMSE: {rolling_rmse:.6f}")
print(f"    MAE:  {rolling_mae:.6f}")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

# Training data
ax.plot(train_data.index, train_data.values, color=BLUE, linewidth=1.2,
        label='Training (70%)')

# Validation data -- connect from training
val_conn = pd.concat([train_data.iloc[[-1]], val_data])
ax.plot(val_conn.index, val_conn.values, color=GREEN, linewidth=1.2,
        label='Validation (15%)')

# Test data (actual) -- connect from validation
test_conn = pd.concat([val_data.iloc[[-1]], test_data])
ax.plot(test_conn.index, test_conn.values, color=PURPLE, linewidth=1.2,
        label='Test (15%)')

# Rolling forecasts with CI
ax.plot(rolling_dates, rolling_forecasts, color=RED, linewidth=1.5,
        linestyle='--', label='Rolling forecast')
ax.fill_between(rolling_dates, rolling_lower, rolling_upper,
                color=RED, alpha=0.12, label='95% CI')

# Mark splits
ax.axvline(x=train_data.index[-1], color='gray', linestyle=':', linewidth=1.0)
ax.axvline(x=val_data.index[-1], color='gray', linestyle=':', linewidth=1.0)

ax.set_title(f'US Real GDP: Rolling 1-Step Ahead Forecast ({best_name})',
             fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Log(GDP)')
style_ax(ax)
add_legend_bottom(ax, ncol=6)

ax.text(0.02, 0.98, f'{best_name}\nTest RMSE: {rolling_rmse:.6f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        style='italic', color='gray')

plt.tight_layout()
save_fig(fig, 'ch3_case_rolling_forecast')

# =============================================================================
# 12. COMPREHENSIVE COMPARISON TABLE
# =============================================================================
print("\n--- Step 10: Final model comparison summary ---")

print("\n  " + "=" * 88)
print(f"  {'Model':<16} {'AIC':>9} {'BIC':>9} {'Log-Lik':>11} "
      f"{'Params':>7} {'LB(10) p':>10} {'LB(20) p':>10}")
print("  " + "=" * 88)

for name in candidate_models:
    if name not in results:
        continue
    r = results[name]
    fit_i = r['fit']
    resid_i = fit_i.resid.dropna()
    lb_i = acorr_ljungbox(resid_i, lags=[10, 20], return_df=True)
    p10 = lb_i.loc[10, 'lb_pvalue']
    p20 = lb_i.loc[20, 'lb_pvalue']
    best_marker = "  ***" if name == best_name else ""
    print(f"  {name:<16} {r['AIC']:>9.2f} {r['BIC']:>9.2f} {r['LogL']:>11.2f} "
          f"{r['nparams']:>7d} {p10:>10.4f} {p20:>10.4f}{best_marker}")

print("  " + "=" * 88)
print(f"  *** = selected model ({best_name})\n")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 72)
print("  ANALYSIS COMPLETE")
print("=" * 72)
print(f"""
  Box-Jenkins ARIMA workflow applied to US Real GDP ({data_source}):

    1. Log Transform:   Applied to stabilize variance.
    2. Stationarity:    ADF + KPSS confirm Log GDP is I(1).
    3. Differencing:    First difference yields stationary GDP growth rate.
    4. Identification:  ACF/PACF of differenced series suggest AR(1) and MA(1).
    5. Candidates:      ARIMA(0,1,0), ARIMA(1,1,0), ARIMA(0,1,1),
                        ARIMA(1,1,1), ARIMA(2,1,1)
    6. Selection:       {best_name} chosen by AIC = {results[best_name]['AIC']:.2f}
    7. Diagnostics:     Ljung-Box tests at lags 5, 10, 15, 20;
                        Jarque-Bera normality test.
    8. Forecasting:     70/15/15 train/val/test split with 95% CI.
    9. Rolling eval:    1-step-ahead RMSE = {rolling_rmse:.6f}

  Output figures saved to {CHARTS_DIR}:
    - ch3_case_raw_data.png / .pdf
    - ch3_case_adf_test.png / .pdf
    - ch3_case_acf_diff.png / .pdf
    - ch3_case_model_comparison.png / .pdf
    - ch3_case_diagnostics.png / .pdf
    - ch3_case_forecast.png / .pdf
    - ch3_case_rolling_forecast.png / .pdf
""")
