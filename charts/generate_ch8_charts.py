import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yfinance as yf
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
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Download real EUR/RON data
try:
    eurron_raw = yf.download('EURRON=X', start='2019-01-01', end='2025-01-01', progress=False)['Close'].squeeze()
    eurron_raw = eurron_raw.dropna()
    if len(eurron_raw) < 200:
        raise ValueError("Insufficient data")
    df = pd.DataFrame({'EURRON': eurron_raw.values.flatten()}, index=eurron_raw.index)
    print("  Using real EUR/RON data from Yahoo Finance")
except Exception as e:
    print(f"  EUR/RON download failed ({e}), using fallback")
    np.random.seed(42)
    n_obs = 1000
    dates = pd.date_range(start='2020-01-01', periods=n_obs, freq='B')
    base_rate = 4.85
    trend = np.linspace(0, 0.15, n_obs)
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
# Build features and train real models for Charts 3-6
# ============================================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import time

# Create features from EUR/RON data
feature_df = pd.DataFrame({
    'Lag 1': df['EURRON'].shift(1),
    'Lag 2': df['EURRON'].shift(2),
    'Lag 3': df['EURRON'].shift(3),
    'Lag 5': df['EURRON'].shift(5),
    'Lag 10': df['EURRON'].shift(10),
    'Rolling Mean 5': df['EURRON'].rolling(5).mean(),
    'Rolling Std 5': df['EURRON'].rolling(5).std(),
    'Rolling Mean 20': df['EURRON'].rolling(20).mean(),
    'Day of Week': df.index.dayofweek,
    'Month': df.index.month,
})
feature_df = feature_df.dropna()
target_all = df['EURRON'].loc[feature_df.index]

# Split at same time boundary as train/test
split_date = train.index[-1]
X_train_ml = feature_df.loc[feature_df.index <= split_date]
X_test_ml = feature_df.loc[feature_df.index > split_date]
y_train_ml = target_all.loc[X_train_ml.index]
y_test_ml = target_all.loc[X_test_ml.index]

print(f"ML features: Train={len(X_train_ml)}, Test={len(X_test_ml)}")

# --- Train ARIMA (1-step-ahead rolling forecast) ---
print("  Training ARIMA(1,1,1)...")
t0 = time.time()
arima_model = ARIMA(train['EURRON'], order=(1, 1, 1))
arima_fit = arima_model.fit()
# Multi-step forecast for test period
arima_forecast = arima_fit.forecast(steps=len(X_test_ml))
# Align to ML test index
arima_pred = arima_forecast.values[:len(X_test_ml)]
arima_time = time.time() - t0
print(f"  ARIMA trained in {arima_time:.2f}s")

# --- Train Random Forest ---
print("  Training Random Forest...")
t0 = time.time()
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15,
                                  min_samples_leaf=5, random_state=42)
rf_model.fit(X_train_ml, y_train_ml)
rf_time = time.time() - t0
rf_pred = rf_model.predict(X_test_ml)
rf_importance = rf_model.feature_importances_
print(f"  RF trained in {rf_time:.2f}s")

# --- Train MLP (Neural Network as LSTM proxy) ---
print("  Training MLP Neural Network (100 epochs)...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train_ml)
X_test_sc = scaler_X.transform(X_test_ml)
y_train_sc = scaler_y.fit_transform(y_train_ml.values.reshape(-1, 1)).ravel()

# Track training history epoch-by-epoch
train_losses = []
val_losses = []
t0 = time.time()
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                   solver='adam', max_iter=1, warm_start=True,
                   random_state=42, learning_rate_init=0.001, tol=1e-8)
for epoch in range(100):
    mlp.max_iter = epoch + 1
    mlp.fit(X_train_sc, y_train_sc)
    train_losses.append(mlp.loss_)
    val_pred_sc = mlp.predict(X_test_sc)
    val_pred_inv = scaler_y.inverse_transform(val_pred_sc.reshape(-1, 1)).ravel()
    val_losses.append(mean_squared_error(y_test_ml, val_pred_inv))
mlp_time = time.time() - t0
mlp_pred_sc = mlp.predict(X_test_sc)
mlp_pred = scaler_y.inverse_transform(mlp_pred_sc.reshape(-1, 1)).ravel()
print(f"  MLP trained in {mlp_time:.2f}s")

# Compute metrics
rmse_arima = np.sqrt(mean_squared_error(y_test_ml, arima_pred))
rmse_rf = np.sqrt(mean_squared_error(y_test_ml, rf_pred))
rmse_mlp = np.sqrt(mean_squared_error(y_test_ml, mlp_pred))

mae_arima = mean_absolute_error(y_test_ml, arima_pred)
mae_rf = mean_absolute_error(y_test_ml, rf_pred)
mae_mlp = mean_absolute_error(y_test_ml, mlp_pred)

print(f"\nModel Performance (EUR/RON levels):")
print(f"ARIMA     - RMSE: {rmse_arima:.4f}, MAE: {mae_arima:.4f}, Time: {arima_time:.2f}s")
print(f"RF        - RMSE: {rmse_rf:.4f}, MAE: {mae_rf:.4f}, Time: {rf_time:.2f}s")
print(f"MLP/LSTM  - RMSE: {rmse_mlp:.4f}, MAE: {mae_mlp:.4f}, Time: {mlp_time:.2f}s")

# ============================================================================
# Chart 3: Feature Importance (Real Random Forest)
# ============================================================================
feature_names = feature_df.columns.tolist()
sorted_idx = np.argsort(rf_importance)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh([feature_names[i] for i in sorted_idx],
               rf_importance[sorted_idx], color='#2E86AB')
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest: Feature Importance for EUR/RON Prediction')

# Add value labels
for bar, val in zip(bars, rf_importance[sorted_idx]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
            va='center', fontsize=10)

plt.tight_layout()
plt.savefig('ch8_case_feature_importance.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Generated: ch8_case_feature_importance.pdf")

# ============================================================================
# Chart 4: MLP/LSTM Training History (Real)
# ============================================================================
epochs = np.arange(1, len(train_losses) + 1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, train_losses, color='#2E86AB', linewidth=2, label='Training Loss')
ax.plot(epochs, val_losses, color='#A23B72', linewidth=2, label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.set_title('Neural Network Training History (EUR/RON)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

plt.tight_layout()
plt.savefig('ch8_case_lstm_training.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Generated: ch8_case_lstm_training.pdf")

# ============================================================================
# Chart 5: Predictions vs Actual (Real Models)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(y_test_ml.index, y_test_ml.values, color='#333333', linewidth=1.5,
        label='Actual EUR/RON')
ax.plot(y_test_ml.index, arima_pred, color='#2E86AB', linewidth=1.2,
        linestyle='--', label='ARIMA', alpha=0.8)
ax.plot(y_test_ml.index, rf_pred, color='#28A745', linewidth=1.2,
        linestyle='--', label='Random Forest', alpha=0.8)
ax.plot(y_test_ml.index, mlp_pred, color='#A23B72', linewidth=1.2,
        linestyle='--', label='MLP/LSTM', alpha=0.8)

ax.set_xlabel('Date')
ax.set_ylabel('EUR/RON Exchange Rate')
ax.set_title('Model Predictions vs Actual EUR/RON Values (Test Period)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)

plt.tight_layout()
plt.savefig('ch8_case_predictions.pdf', bbox_inches='tight', transparent=True)
plt.close()
print("Generated: ch8_case_predictions.pdf")

# ============================================================================
# Chart 6: Model Comparison (Bar Charts - Real Metrics)
# ============================================================================
models = ['ARIMA', 'Random Forest', 'MLP/LSTM']
rmse_values = [rmse_arima, rmse_rf, rmse_mlp]
mae_values = [mae_arima, mae_rf, mae_mlp]
train_times = [arima_time, rf_time, mlp_time]

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
