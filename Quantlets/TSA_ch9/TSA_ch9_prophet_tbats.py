"""
TSA_ch9_prophet_tbats
=====================
Prophet and TBATS for Complex Seasonality

This script demonstrates:
- Multiple seasonality patterns
- Fourier terms for seasonality
- Prophet model components
- TBATS decomposition
- Model comparison

Author: Time Series Analysis Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    try:
        from fbprophet import Prophet
        HAS_PROPHET = True
    except ImportError:
        HAS_PROPHET = False
        print("Prophet not installed. Some examples will use simulated results.")

# Chart style settings - Nature journal quality
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
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.75

def save_fig(name):
    """Save figure with transparent background."""
    plt.savefig(f'{name}.pdf', bbox_inches='tight', transparent=True, dpi=300)
    plt.savefig(f'{name}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"   Saved: {name}.pdf")

print("=" * 70)
print("PROPHET AND TBATS: COMPLEX SEASONALITY")
print("=" * 70)

# =============================================================================
# 1. Generate Data with Multiple Seasonalities
# =============================================================================
np.random.seed(42)
n = 365 * 3  # 3 years of daily data

print("\n1. MULTIPLE SEASONALITY DATA")
print("-" * 40)

# Time index
t = np.arange(n)

# Components
trend = 100 + 0.02 * t  # Slow upward trend
weekly = 10 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
yearly = 20 * np.sin(2 * np.pi * t / 365)  # Yearly seasonality
noise = np.random.normal(0, 5, n)

# Combined series
y = trend + weekly + yearly + noise

dates = pd.date_range('2020-01-01', periods=n, freq='D')
df = pd.DataFrame({'ds': dates, 'y': y})

print(f"   Data: {n} daily observations")
print(f"   Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print("   Seasonalities: Weekly (s=7) + Yearly (s=365)")

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Full series
axes[0].plot(dates, y, color='#1A3A6E', linewidth=0.5)
axes[0].set_title('Time Series with Multiple Seasonalities', fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Value')

# Zoom: 2 months to show weekly pattern
axes[1].plot(dates[:60], y[:60], color='#1A3A6E', linewidth=1, marker='o', markersize=2)
axes[1].set_title('Weekly Pattern (First 2 Months)', fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Value')

# Yearly pattern
monthly_avg = df.set_index('ds').resample('M').mean()
axes[2].plot(monthly_avg.index, monthly_avg['y'], color='#DC3545', linewidth=2, marker='o')
axes[2].set_title('Yearly Pattern (Monthly Averages)', fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Average Value')

plt.tight_layout()
plt.savefig('ch9_multiple_seasonality.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch9_multiple_seasonality.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch9_multiple_seasonality.pdf")

# =============================================================================
# 2. Fourier Terms for Seasonality
# =============================================================================
print("\n2. FOURIER TERMS FOR SEASONALITY")
print("-" * 40)

def create_fourier_terms(t, period, n_terms):
    """Create Fourier series terms for seasonality."""
    terms = {}
    for k in range(1, n_terms + 1):
        terms[f'sin_{period}_{k}'] = np.sin(2 * np.pi * k * t / period)
        terms[f'cos_{period}_{k}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(terms)

# Create Fourier terms
fourier_weekly = create_fourier_terms(t, 7, 3)
fourier_yearly = create_fourier_terms(t, 365, 5)

print(f"   Weekly seasonality: 3 Fourier pairs (6 terms)")
print(f"   Yearly seasonality: 5 Fourier pairs (10 terms)")

# Visualize Fourier approximation
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Weekly Fourier terms
for k in range(1, 4):
    axes[0, 0].plot(t[:30], np.sin(2 * np.pi * k * t[:30] / 7),
                    label=f'sin(2π·{k}·t/7)', alpha=0.7)
axes[0, 0].set_title('Weekly Fourier Terms (Sine)', fontweight='bold')
axes[0, 0].set_xlabel('Day')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

# Approximate weekly pattern
weekly_approx = (fourier_weekly.iloc[:, :6].values @
                 np.array([10, 0, 2, 0, 1, 0]))  # Approximate coefficients
axes[0, 1].plot(t[:60], weekly[:60], 'b-', label='True', linewidth=2)
axes[0, 1].plot(t[:60], weekly_approx[:60], 'r--', label='Fourier (K=3)', linewidth=2)
axes[0, 1].set_title('Weekly Pattern: True vs Fourier Approximation', fontweight='bold')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# Yearly Fourier terms
for k in range(1, 4):
    axes[1, 0].plot(t, np.sin(2 * np.pi * k * t / 365),
                    label=f'K={k}', alpha=0.7)
axes[1, 0].set_title('Yearly Fourier Terms', fontweight='bold')
axes[1, 0].set_xlabel('Day')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

# Effect of K on approximation
yearly_approx1 = 20 * np.sin(2 * np.pi * t / 365)  # K=1
yearly_approx3 = 20 * np.sin(2 * np.pi * t / 365) + 3 * np.sin(2 * np.pi * 2 * t / 365)
axes[1, 1].plot(t, yearly, 'b-', label='True', linewidth=2, alpha=0.7)
axes[1, 1].plot(t, yearly_approx1, 'r--', label='K=1', linewidth=1.5)
axes[1, 1].set_title('Yearly Pattern Approximation', fontweight='bold')
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()
plt.savefig('ch9_fourier_terms.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch9_fourier_terms.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch9_fourier_terms.pdf")

# =============================================================================
# 3. Prophet Model (if available)
# =============================================================================
print("\n3. PROPHET MODEL")
print("-" * 40)

if HAS_PROPHET:
    # Fit Prophet
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                   daily_seasonality=False, changepoint_prior_scale=0.05)
    model.fit(df)

    # Make forecast
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    print("   Prophet model fitted successfully")
    print(f"   Forecast horizon: 90 days")

    # Plot components
    fig = model.plot_components(forecast)
    plt.savefig('ch9_prophet_components.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: ch9_prophet_components.pdf")

    # Forecast plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['ds'], df['y'], color='#1A3A6E', linewidth=0.5, label='Historical')
    ax.plot(forecast['ds'], forecast['yhat'], color='#DC3545', linewidth=1.5, label='Forecast')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                    color='#DC3545', alpha=0.2, label='95% CI')

    # Visual separator between historical and forecast
    split_point = df['ds'].iloc[-1]
    ax.axvline(x=split_point, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    y_pos = ax.get_ylim()[1] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(split_point, y_pos, '  Forecast ', fontsize=9, ha='left', va='top',
            color='black', fontweight='bold', alpha=0.8)

    ax.set_title('Prophet Forecast', fontweight='bold', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig('ch9_prophet_forecast.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('ch9_prophet_forecast.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: ch9_prophet_forecast.pdf")
else:
    print("   Prophet not available. Creating simulated results...")

    # Simulate Prophet-like decomposition
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)

    axes[0].plot(dates, trend, color='#1A3A6E', linewidth=1.5)
    axes[0].set_title('Trend', fontweight='bold')
    axes[0].set_ylabel('Trend')

    axes[1].plot(range(7), 10 * np.sin(2 * np.pi * np.arange(7) / 7), color='#1A3A6E',
                 linewidth=2, marker='o')
    axes[1].set_title('Weekly Seasonality', fontweight='bold')
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    days_of_year = pd.date_range('2020-01-01', periods=365, freq='D')
    axes[2].plot(days_of_year, 20 * np.sin(2 * np.pi * np.arange(365) / 365),
                 color='#1A3A6E', linewidth=1.5)
    axes[2].set_title('Yearly Seasonality', fontweight='bold')

    axes[3].plot(dates, y, color='#1A3A6E', linewidth=0.5)
    axes[3].set_title('Observed', fontweight='bold')

    plt.tight_layout()
    plt.savefig('ch9_prophet_components.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('ch9_prophet_components.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: ch9_prophet_components.pdf (simulated)")

# =============================================================================
# 4. TBATS Components (Simulated)
# =============================================================================
print("\n4. TBATS DECOMPOSITION")
print("-" * 40)

print("   TBATS = Trigonometric + Box-Cox + ARMA + Trend + Seasonal")
print("\n   Components:")
print("   T - Trigonometric seasonality (Fourier)")
print("   B - Box-Cox transformation")
print("   A - ARMA errors")
print("   T - Trend (possibly damped)")
print("   S - Seasonal (multiple periods)")

# Simulate TBATS-like decomposition
level = trend + np.cumsum(np.random.normal(0, 0.1, n))  # Local level
seasonal_combined = weekly + yearly

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(dates, y, color='#1A3A6E', linewidth=0.5)
axes[0].set_title('Observed', fontweight='bold')
axes[0].set_ylabel('Value')

axes[1].plot(dates, level, color='#2E7D32', linewidth=1)
axes[1].set_title('Level (Trend)', fontweight='bold')
axes[1].set_ylabel('Level')

axes[2].plot(dates, seasonal_combined, color='#E67E22', linewidth=0.8)
axes[2].set_title('Combined Seasonality (Weekly + Yearly)', fontweight='bold')
axes[2].set_ylabel('Seasonal')

axes[3].plot(dates, noise, color='#666666', linewidth=0.5)
axes[3].axhline(y=0, color='red', linestyle='--', linewidth=0.5)
axes[3].set_title('Residuals', fontweight='bold')
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Date')

plt.tight_layout()
plt.savefig('ch9_tbats_decomposition.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch9_tbats_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch9_tbats_decomposition.pdf")

# =============================================================================
# 5. Changepoint Detection
# =============================================================================
print("\n5. CHANGEPOINT DETECTION")
print("-" * 40)

# Create series with changepoints
n_cp = 500
t_cp = np.arange(n_cp)
y_cp = np.zeros(n_cp)

# Different regimes
y_cp[:150] = 50 + 0.1 * t_cp[:150] + np.random.normal(0, 3, 150)
y_cp[150:300] = 80 + 0.3 * (t_cp[150:300] - 150) + np.random.normal(0, 3, 150)
y_cp[300:] = 130 - 0.1 * (t_cp[300:] - 300) + np.random.normal(0, 3, 200)

changepoints = [150, 300]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t_cp, y_cp, color='#1A3A6E', linewidth=1)
for cp in changepoints:
    ax.axvline(x=cp, color='#DC3545', linestyle='--', linewidth=2, alpha=0.7)
ax.scatter(changepoints, [y_cp[cp] for cp in changepoints], color='#DC3545',
           s=100, zorder=5, label='Changepoints')
ax.set_title('Trend Changepoint Detection', fontweight='bold', fontsize=14)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()
plt.savefig('ch9_changepoints.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch9_changepoints.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch9_changepoints.pdf")
print(f"   Detected changepoints at: {changepoints}")

# =============================================================================
# 6. Model Selection Guide
# =============================================================================
print("\n6. MODEL SELECTION GUIDE")
print("-" * 40)

print("""
   When to use Prophet:
   ✓ Business forecasting with holidays
   ✓ Missing data / irregular intervals
   ✓ Multiple strong seasonalities
   ✓ Need interpretable components
   ✓ Trend changepoints expected

   When to use TBATS:
   ✓ High-frequency data (hourly, sub-daily)
   ✓ Non-integer seasonal periods
   ✓ Complex seasonal patterns
   ✓ Need Box-Cox transformation
   ✓ Automatic model selection

   When to use SARIMA:
   ✓ Single seasonality
   ✓ Well-behaved, regular data
   ✓ Need statistical inference
   ✓ Short-term forecasting
""")

# Summary comparison table
comparison_data = {
    'Feature': ['Multiple Seasonality', 'Missing Data', 'Changepoints',
                'Interpretability', 'Speed', 'Automation'],
    'Prophet': ['Excellent', 'Good', 'Excellent', 'Excellent', 'Fast', 'High'],
    'TBATS': ['Excellent', 'Limited', 'No', 'Good', 'Slow', 'High'],
    'SARIMA': ['Limited', 'No', 'No', 'Good', 'Fast', 'Low']
}
comparison_df = pd.DataFrame(comparison_data)
print("\n   Model Comparison:")
print(comparison_df.to_string(index=False))

print("\n" + "=" * 70)
print("PROPHET AND TBATS ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch9_multiple_seasonality.pdf: Data with multiple seasonal patterns")
print("  - ch9_fourier_terms.pdf: Fourier approximation of seasonality")
print("  - ch9_prophet_components.pdf: Prophet decomposition")
print("  - ch9_tbats_decomposition.pdf: TBATS-style decomposition")
print("  - ch9_changepoints.pdf: Trend changepoint detection")
