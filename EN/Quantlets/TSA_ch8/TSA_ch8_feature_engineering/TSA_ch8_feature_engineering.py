"""
TSA_ch8_feature_engineering

Feature engineering for time series ML models.
Demonstrates lag features, rolling statistics, and calendar features
using Germany daily electricity consumption data (OPSD, 2012-2017).

Author: Daniel Traian Pele
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Chart style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['axes.grid'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

COLORS = {
    'blue': '#2E86AB',
    'red': '#A23B72',
    'green': '#28A745',
    'orange': '#E67E22',
    'purple': '#8E44AD',
}

# =============================================================================
# Load real data: Germany daily electricity consumption (OPSD)
# =============================================================================
url = ('https://raw.githubusercontent.com/jenfly/opsd/master/'
       'opsd_germany_daily.csv')
try:
    df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
    y_full = df['Consumption'].dropna()
    print(f"Loaded OPSD data: {len(y_full)} observations")
except Exception as e:
    print(f"Download failed ({e}), using fallback")
    np.random.seed(789)
    n = 365 * 5
    dates = pd.date_range('2012-01-01', periods=n, freq='D')
    trend = np.linspace(1300, 1250, n)
    weekly = 100 * np.array([1.1, 1.05, 1.0, 0.95, 0.9, 0.7, 0.6]
                            * (n // 7 + 1))[:n]
    annual = 80 * np.cos(2 * np.pi * np.arange(n) / 365.25)
    noise = np.random.randn(n) * 40
    y_full = pd.Series(trend + weekly + annual + noise, index=dates,
                       name='Consumption')

# Use a 200-day window for clearer visualization
y = y_full.iloc[-200:].copy()
n = len(y)

# =============================================================================
# Create features
# =============================================================================
lag1 = y.shift(1)
lag2 = y.shift(2)
rolling_mean7 = y.rolling(7).mean()
rolling_std7 = y.rolling(7).std()

# Feature importance from a quick Random Forest fit
from sklearn.ensemble import RandomForestRegressor

feat_df = pd.DataFrame({
    'y_{t-1}': y.shift(1),
    'y_{t-2}': y.shift(2),
    'y_{t-7}': y.shift(7),
    'MA_7': y.rolling(7).mean(),
    'STD_7': y.rolling(7).std(),
}).dropna()
target = y.loc[feat_df.index]

rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(feat_df, target)
importances = rf.feature_importances_
feature_names = feat_df.columns.tolist()

# =============================================================================
# Generate chart
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# (a) Original series
axes[0, 0].plot(y.index, y.values, color=COLORS['blue'], linewidth=1.5,
                label='$y_t$ (GWh)')
axes[0, 0].set_title('Original Series $y_t$', fontweight='bold')
axes[0, 0].set_ylabel('Consumption (GWh)')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  frameon=False)

# (b) Lag features
axes[0, 1].plot(y.index, y.values, color=COLORS['blue'], linewidth=1.5,
                alpha=0.5, label='$y_t$')
axes[0, 1].plot(y.index, lag1.values, color=COLORS['red'], linewidth=1.5,
                label='$y_{t-1}$ (lag 1)')
axes[0, 1].plot(y.index, lag2.values, color=COLORS['orange'], linewidth=1.5,
                label='$y_{t-2}$ (lag 2)')
axes[0, 1].set_title('Lag Features', fontweight='bold')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  frameon=False, ncol=3)

# (c) Rolling statistics
axes[1, 0].plot(y.index, y.values, color=COLORS['blue'], linewidth=1,
                alpha=0.5, label='$y_t$')
axes[1, 0].plot(y.index, rolling_mean7.values, color=COLORS['green'],
                linewidth=2, label='Rolling Mean (7)')
axes[1, 0].fill_between(y.index,
                        (rolling_mean7 - rolling_std7).values,
                        (rolling_mean7 + rolling_std7).values,
                        color=COLORS['green'], alpha=0.15,
                        label='$\\pm$ 1 Std (7)')
axes[1, 0].set_title('Rolling Statistics', fontweight='bold')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  frameon=False, ncol=3)

# (d) Feature importance from Random Forest
sorted_idx = np.argsort(importances)
bars = axes[1, 1].barh(
    [feature_names[i] for i in sorted_idx],
    importances[sorted_idx],
    color=[COLORS['blue'] if 'y_' in feature_names[i]
           else COLORS['green'] if 'MA' in feature_names[i]
           else COLORS['orange']
           for i in sorted_idx],
    alpha=0.8
)
axes[1, 1].set_title('Random Forest Feature Importance', fontweight='bold')
axes[1, 1].set_xlabel('Importance')

for bar, val in zip(bars, importances[sorted_idx]):
    axes[1, 1].text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{val:.0%}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('ch8_feature_engineering.pdf', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('ch8_feature_engineering.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('ch8_feature_engineering.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Created: ch8_feature_engineering.pdf")
print("Created: ch8_feature_engineering.png")
