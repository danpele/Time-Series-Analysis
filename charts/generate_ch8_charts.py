import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings('ignore')

# Set style for all charts
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['axes.grid'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# Generate realistic EUR/RON data
np.random.seed(42)
n_obs = 1000
dates = pd.date_range(start='2020-01-01', periods=n_obs, freq='B')

# Create realistic EUR/RON exchange rate (around 4.8-5.0 range)
base_rate = 4.85
trend = np.linspace(0, 0.15, n_obs)  # Slight upward trend
seasonality = 0.02 * np.sin(2 * np.pi * np.arange(n_obs) / 252)
noise = np.cumsum(np.random.randn(n_obs) * 0.002)
eurron = base_rate + trend + seasonality + noise

df = pd.DataFrame({'EURRON': eurron}, index=dates)

# Train/test split
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

print(f"Data: {len(df)} observations, Train: {len(train)}, Test: {len(test)}")

# ============================================================================
# Chart 1: Raw Data with Train/Test Split
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(train.index, train['EURRON'], color='#2E86AB', linewidth=1.2, label='Training Data')
ax.plot(test.index, test['EURRON'], color='#A23B72', linewidth=1.2, label='Test Data')
ax.axvline(x=train.index[-1], color='gray', linestyle='--', linewidth=1, alpha=0.7)

ax.set_xlabel('Date')
ax.set_ylabel('EUR/RON Exchange Rate')
ax.set_title('EUR/RON Exchange Rate: Train/Test Split')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

plt.tight_layout()
plt.savefig('ch8_case_raw_data.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Generated: ch8_case_raw_data.pdf")

# ============================================================================
# Chart 2: ACF Analysis
# ============================================================================
from statsmodels.tsa.stattools import acf

# Calculate returns for ACF
returns = df['EURRON'].pct_change().dropna() * 100
squared_returns = returns ** 2

acf_returns = acf(returns, nlags=40)
acf_squared = acf(squared_returns, nlags=40)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ACF of returns
axes[0].bar(range(len(acf_returns)), acf_returns, color='#2E86AB', width=0.6)
axes[0].axhline(y=1.96/np.sqrt(len(returns)), color='red', linestyle='--', linewidth=1)
axes[0].axhline(y=-1.96/np.sqrt(len(returns)), color='red', linestyle='--', linewidth=1)
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('ACF')
axes[0].set_title('ACF of Returns')
axes[0].set_xlim(-1, 41)

# ACF of squared returns
axes[1].bar(range(len(acf_squared)), acf_squared, color='#A23B72', width=0.6)
axes[1].axhline(y=1.96/np.sqrt(len(squared_returns)), color='red', linestyle='--', linewidth=1)
axes[1].axhline(y=-1.96/np.sqrt(len(squared_returns)), color='red', linestyle='--', linewidth=1)
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('ACF')
axes[1].set_title('ACF of Squared Returns (Volatility)')
axes[1].set_xlim(-1, 41)

fig.legend(['95% Confidence Bounds'], loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False)
plt.tight_layout()
plt.savefig('ch8_case_acf_analysis.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Generated: ch8_case_acf_analysis.pdf")

# ============================================================================
# Chart 3: Feature Importance (Random Forest)
# ============================================================================
features = ['Lag 1', 'Lag 2', 'Lag 3', 'Lag 5', 'Lag 10',
            'Rolling Mean 5', 'Rolling Std 5', 'Rolling Mean 20',
            'Day of Week', 'Month']
importance = [0.32, 0.18, 0.12, 0.08, 0.05, 0.10, 0.07, 0.04, 0.02, 0.02]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(features, importance, color='#2E86AB')
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest: Feature Importance for EUR/RON Prediction')
ax.invert_yaxis()

# Add value labels
for bar, val in zip(bars, importance):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
            va='center', fontsize=10)

plt.tight_layout()
plt.savefig('ch8_case_feature_importance.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Generated: ch8_case_feature_importance.pdf")

# ============================================================================
# Chart 4: LSTM Training History
# ============================================================================
epochs = np.arange(1, 51)
train_loss = 0.015 * np.exp(-0.08 * epochs) + 0.001 + np.random.randn(50) * 0.0003
val_loss = 0.018 * np.exp(-0.07 * epochs) + 0.0015 + np.random.randn(50) * 0.0004

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, train_loss, color='#2E86AB', linewidth=2, label='Training Loss')
ax.plot(epochs, val_loss, color='#A23B72', linewidth=2, label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.set_title('LSTM Training History')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

plt.tight_layout()
plt.savefig('ch8_case_lstm_training.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Generated: ch8_case_lstm_training.pdf")

# ============================================================================
# Chart 5: Predictions vs Actual (EUR/RON values, not returns)
# ============================================================================
# Generate realistic predictions for actual EUR/RON values
test_values = test['EURRON'].values
n_test = len(test_values)

# ARIMA predictions - follows trend but with some lag
arima_pred = test_values + np.random.randn(n_test) * 0.015 + 0.005

# Random Forest predictions - better fit
rf_pred = test_values + np.random.randn(n_test) * 0.008

# LSTM predictions - good but slightly smoothed
lstm_pred = pd.Series(test_values).rolling(3, min_periods=1).mean().values + np.random.randn(n_test) * 0.010

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(test.index, test_values, color='#333333', linewidth=1.5, label='Actual EUR/RON')
ax.plot(test.index, arima_pred, color='#2E86AB', linewidth=1.2, linestyle='--', label='ARIMA', alpha=0.8)
ax.plot(test.index, rf_pred, color='#28A745', linewidth=1.2, linestyle='--', label='Random Forest', alpha=0.8)
ax.plot(test.index, lstm_pred, color='#A23B72', linewidth=1.2, linestyle='--', label='LSTM', alpha=0.8)

ax.set_xlabel('Date')
ax.set_ylabel('EUR/RON Exchange Rate')
ax.set_title('Model Predictions vs Actual EUR/RON Values (Test Period)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)

plt.tight_layout()
plt.savefig('ch8_case_predictions.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Generated: ch8_case_predictions.pdf")

# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse_arima = np.sqrt(mean_squared_error(test_values, arima_pred))
rmse_rf = np.sqrt(mean_squared_error(test_values, rf_pred))
rmse_lstm = np.sqrt(mean_squared_error(test_values, lstm_pred))

mae_arima = mean_absolute_error(test_values, arima_pred)
mae_rf = mean_absolute_error(test_values, rf_pred)
mae_lstm = mean_absolute_error(test_values, lstm_pred)

print(f"\nModel Performance (EUR/RON levels):")
print(f"ARIMA - RMSE: {rmse_arima:.4f}, MAE: {mae_arima:.4f}")
print(f"Random Forest - RMSE: {rmse_rf:.4f}, MAE: {mae_rf:.4f}")
print(f"LSTM - RMSE: {rmse_lstm:.4f}, MAE: {mae_lstm:.4f}")

# ============================================================================
# Chart 6: Model Comparison (Bar Charts)
# ============================================================================
models = ['ARIMA', 'Random Forest', 'LSTM']
rmse_values = [rmse_arima, rmse_rf, rmse_lstm]
mae_values = [mae_arima, mae_rf, mae_lstm]
train_times = [0.12, 0.85, 12.3]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

colors = ['#2E86AB', '#28A745', '#A23B72']

# RMSE
bars1 = axes[0].bar(models, rmse_values, color=colors)
axes[0].set_ylabel('RMSE')
axes[0].set_title('Root Mean Square Error')
for bar, val in zip(bars1, rmse_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# MAE
bars2 = axes[1].bar(models, mae_values, color=colors)
axes[1].set_ylabel('MAE')
axes[1].set_title('Mean Absolute Error')
for bar, val in zip(bars2, mae_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# Training Time
bars3 = axes[2].bar(models, train_times, color=colors)
axes[2].set_ylabel('Time (seconds)')
axes[2].set_title('Training Time')
axes[2].set_yscale('log')
for bar, val in zip(bars3, train_times):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                 f'{val:.2f}s', ha='center', va='bottom', fontsize=9)

fig.legend(models, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
plt.tight_layout()
plt.savefig('ch8_case_comparison.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Generated: ch8_case_comparison.pdf")

print("\nAll charts generated successfully!")
