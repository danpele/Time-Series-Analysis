#!/usr/bin/env python3
"""
Generate charts for Chapter 8 lecture quiz answers
Modern Extensions: ARFIMA, Random Forest, LSTM
Time Series Analysis Course
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Set style for transparent backgrounds
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Output directory
output_dir = 'charts'
os.makedirs(output_dir, exist_ok=True)

def save_fig(name):
    plt.savefig(f'{output_dir}/{name}.pdf', format='pdf', bbox_inches='tight',
                transparent=True, dpi=150)
    plt.close()
    print(f"  Created {name}.pdf")

# =============================================================================
# CHAPTER 8: Modern Extensions Quiz Charts
# =============================================================================
print("Chapter 8: Modern Extensions Quiz Charts")

# Quiz 1: Long Memory - ACF decay comparison
def ch8_quiz1_long_memory():
    """Compare ACF decay: short memory vs long memory"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    lags = np.arange(0, 31)

    # Short memory (AR(1)) - exponential decay
    phi = 0.7
    acf_short = phi ** lags

    axes[0].bar(lags, acf_short, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0].axhline(y=1.96/np.sqrt(100), color='red', linestyle='--', alpha=0.7)
    axes[0].axhline(y=-1.96/np.sqrt(100), color='red', linestyle='--', alpha=0.7)
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    axes[0].set_title('Short Memory (AR(1))\nExponential Decay: $\\rho_k = \\phi^k$', fontsize=10, color='steelblue')
    axes[0].set_ylim(-0.2, 1.1)

    # Long memory (ARFIMA) - hyperbolic decay
    d = 0.35
    # Approximate ACF for ARFIMA: rho_k ~ C * k^(2d-1)
    acf_long = np.ones(31)
    for k in range(1, 31):
        acf_long[k] = np.prod([(j - 1 + d) / (j - d) for j in range(1, k + 1)])

    axes[1].bar(lags, acf_long, color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1].axhline(y=1.96/np.sqrt(100), color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(y=-1.96/np.sqrt(100), color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('ACF')
    axes[1].set_title('Long Memory (ARFIMA, d=0.35)\nHyperbolic Decay: $\\rho_k \\sim k^{2d-1}$', fontsize=10, color='green')
    axes[1].set_ylim(-0.2, 1.1)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch8_quiz1_long_memory')

ch8_quiz1_long_memory()


# Quiz 2: ARFIMA parameter d interpretation
def ch8_quiz2_arfima_d():
    """Show effect of different d values on ACF"""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    lags = np.arange(0, 26)

    d_values = [0.0, 0.25, 0.45]
    titles = ['d = 0 (ARMA)\nShort Memory', 'd = 0.25\nModerate Long Memory', 'd = 0.45\nStrong Long Memory']
    colors = ['blue', 'orange', 'red']

    for ax, d, title, color in zip(axes, d_values, titles, colors):
        if d == 0:
            # AR(1) with phi = 0.5
            acf = 0.5 ** lags
        else:
            # ARFIMA ACF approximation
            acf = np.ones(26)
            for k in range(1, 26):
                acf[k] = np.prod([(j - 1 + d) / (j - d) for j in range(1, k + 1)])

        ax.bar(lags, acf, color=color, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0.196, color='gray', linestyle='--', alpha=0.7, label='95% CI')
        ax.axhline(y=-0.196, color='gray', linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Lag')
        ax.set_title(title, fontsize=10, color=color)
        ax.set_ylim(-0.2, 1.1)

    axes[0].set_ylabel('ACF')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch8_quiz2_arfima_d')

ch8_quiz2_arfima_d()


# Quiz 3: Hurst exponent interpretation
def ch8_quiz3_hurst():
    """Visualize Hurst exponent and its interpretation"""
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    n = 200

    # H < 0.5: Anti-persistent (mean reverting)
    # Simulate by alternating signs
    eps = np.random.normal(0, 1, n)
    anti_persistent = np.zeros(n)
    for t in range(1, n):
        anti_persistent[t] = -0.3 * anti_persistent[t-1] + eps[t]

    axes[0].plot(anti_persistent, 'b-', linewidth=1, alpha=0.8)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].set_title('H < 0.5: Anti-Persistent\n(Mean-Reverting)', fontsize=10, color='blue')
    axes[0].text(100, max(anti_persistent)*0.8, 'd < 0', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), ha='center')

    # H = 0.5: Random Walk
    random_walk = np.cumsum(np.random.normal(0, 1, n))

    axes[1].plot(random_walk, 'gray', linewidth=1.5)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Time')
    axes[1].set_title('H = 0.5: Random Walk\n(No Memory)', fontsize=10, color='gray')
    axes[1].text(100, max(random_walk)*0.8, 'd = 0', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8), ha='center')

    # H > 0.5: Persistent (trending)
    # Simulate with positive autocorrelation
    persistent = np.zeros(n)
    eps = np.random.normal(0, 1, n)
    for t in range(1, n):
        persistent[t] = 0.8 * persistent[t-1] + eps[t]
    persistent = np.cumsum(persistent * 0.1)  # Make it trend

    axes[2].plot(persistent, 'green', linewidth=1.5)
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time')
    axes[2].set_title('H > 0.5: Persistent\n(Trend-Following)', fontsize=10, color='green')
    axes[2].text(100, min(persistent) + (max(persistent)-min(persistent))*0.8, 'd > 0', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), ha='center')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch8_quiz3_hurst')

ch8_quiz3_hurst()


# Quiz 4: Random Forest feature importance
def ch8_quiz4_rf_features():
    """Visualize typical feature importance for time series RF"""
    fig, ax = plt.subplots(figsize=(8, 5))

    features = ['lag_1', 'lag_2', 'rolling_std_5', 'rolling_mean_5',
                'lag_3', 'lag_5', 'dayofweek', 'month', 'lag_10', 'rolling_mean_20']
    importance = [0.35, 0.18, 0.12, 0.10, 0.08, 0.05, 0.04, 0.03, 0.03, 0.02]

    colors = ['darkblue', 'blue', 'steelblue', 'steelblue',
              'lightblue', 'lightblue', 'orange', 'orange', 'gray', 'gray']

    y_pos = np.arange(len(features))

    bars = ax.barh(y_pos, importance, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest: Typical Feature Importance\nfor Time Series Forecasting', fontsize=11)
    ax.invert_yaxis()

    # Add value labels
    for bar, imp in zip(bars, importance):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{imp:.0%}', va='center', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkblue', alpha=0.7, label='Lag features (most important)'),
        Patch(facecolor='steelblue', alpha=0.7, label='Rolling statistics'),
        Patch(facecolor='orange', alpha=0.7, label='Calendar features'),
        Patch(facecolor='gray', alpha=0.7, label='Other')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=8, ncol=2, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch8_quiz4_rf_features')

ch8_quiz4_rf_features()


# Quiz 5: Time Series CV vs K-Fold
def ch8_quiz5_cv_comparison():
    """Illustrate why standard k-fold is wrong for time series"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    n_samples = 20
    n_folds = 5

    # Standard K-Fold (WRONG for time series)
    ax = axes[0]
    for fold in range(n_folds):
        test_start = fold * (n_samples // n_folds)
        test_end = (fold + 1) * (n_samples // n_folds)

        for i in range(n_samples):
            if test_start <= i < test_end:
                color = 'red'
                label = 'Test' if i == test_start and fold == 0 else None
            else:
                color = 'blue'
                label = 'Train' if i == 0 and fold == 0 else None
            ax.barh(fold, 1, left=i, color=color, edgecolor='white', alpha=0.7, label=label)

    ax.set_yticks(range(n_folds))
    ax.set_yticklabels([f'Fold {i+1}' for i in range(n_folds)])
    ax.set_xlabel('Time')
    ax.set_title('Standard K-Fold CV (WRONG for Time Series)\nFuture data used to predict past!',
                 fontsize=10, color='red')
    ax.set_xlim(-0.5, n_samples + 0.5)

    # Add arrows showing data leakage
    ax.annotate('', xy=(8, 0.3), xytext=(12, 0.3),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(10, 0.6, 'Data\nLeakage!', ha='center', fontsize=8, color='red', fontweight='bold')

    # Time Series CV (CORRECT)
    ax = axes[1]
    initial_train = 8
    test_size = 2

    fold = 0
    for train_end in range(initial_train, n_samples - test_size + 1, test_size):
        for i in range(n_samples):
            if i < train_end:
                color = 'blue'
            elif i < train_end + test_size:
                color = 'red'
            else:
                color = 'lightgray'
            ax.barh(fold, 1, left=i, color=color, edgecolor='white', alpha=0.7)
        fold += 1
        if fold >= 6:
            break

    ax.set_yticks(range(fold))
    ax.set_yticklabels([f'Fold {i+1}' for i in range(fold)])
    ax.set_xlabel('Time')
    ax.set_title('Time Series CV (CORRECT)\nTrain always before test, expanding window',
                 fontsize=10, color='green')
    ax.set_xlim(-0.5, n_samples + 0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Train'),
        Patch(facecolor='red', alpha=0.7, label='Test'),
        Patch(facecolor='lightgray', alpha=0.7, label='Future (not used)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=8, ncol=3, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch8_quiz5_cv_comparison')

ch8_quiz5_cv_comparison()


# Quiz 6: LSTM vs RNN - Vanishing Gradient
def ch8_quiz6_lstm_advantage():
    """Illustrate LSTM's advantage over standard RNN"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # RNN gradient flow
    ax = axes[0]
    timesteps = 10
    gradient_rnn = [1.0]
    decay_factor = 0.7
    for t in range(1, timesteps):
        gradient_rnn.append(gradient_rnn[-1] * decay_factor)

    ax.bar(range(timesteps), gradient_rnn, color='red', alpha=0.7, edgecolor='black')
    ax.axhline(y=0.1, color='blue', linestyle='--', alpha=0.7, label='Useful threshold')
    ax.set_xlabel('Timesteps back')
    ax.set_ylabel('Gradient magnitude')
    ax.set_title('Standard RNN\nVanishing Gradient Problem', fontsize=10, color='red')
    ax.set_ylim(0, 1.2)
    ax.text(5, 0.8, 'Gradients vanish\n= Cannot learn\nlong dependencies',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # LSTM gradient flow
    ax = axes[1]
    gradient_lstm = [1.0]
    decay_factor_lstm = 0.95  # Much better gradient flow
    for t in range(1, timesteps):
        gradient_lstm.append(gradient_lstm[-1] * decay_factor_lstm)

    ax.bar(range(timesteps), gradient_lstm, color='green', alpha=0.7, edgecolor='black')
    ax.axhline(y=0.1, color='blue', linestyle='--', alpha=0.7, label='Useful threshold')
    ax.set_xlabel('Timesteps back')
    ax.set_ylabel('Gradient magnitude')
    ax.set_title('LSTM\nGradients Preserved', fontsize=10, color='green')
    ax.set_ylim(0, 1.2)
    ax.text(5, 0.5, 'Cell state allows\ngradients to flow\n= Learns long\ndependencies',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch8_quiz6_lstm_advantage')

ch8_quiz6_lstm_advantage()


# Quiz 7: Model Selection Decision
def ch8_quiz7_model_selection():
    """Visualization of model selection criteria"""
    fig, ax = plt.subplots(figsize=(9, 5))

    models = ['ARIMA', 'ARFIMA', 'Random\nForest', 'LSTM']

    # Scores on different criteria (0-10 scale)
    criteria = {
        'Small Data': [9, 8, 4, 2],
        'Nonlinearity': [2, 2, 9, 9],
        'Interpretability': [9, 7, 6, 2],
        'Computation Speed': [9, 8, 6, 3],
        'Long Memory': [3, 9, 5, 7]
    }

    x = np.arange(len(models))
    width = 0.15
    multiplier = 0

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for (criterion, scores), color in zip(criteria.items(), colors):
        offset = width * multiplier
        bars = ax.bar(x + offset, scores, width, label=criterion, color=color, alpha=0.8)
        multiplier += 1

    ax.set_ylabel('Score (0-10)')
    ax.set_title('Model Selection: Strengths by Criteria', fontsize=11)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)
    ax.set_ylim(0, 11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch8_quiz7_model_selection')

ch8_quiz7_model_selection()


# Quiz 8: Data Leakage Example
def ch8_quiz8_data_leakage():
    """Illustrate data leakage in time series feature engineering"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    np.random.seed(42)
    n = 15

    # Sample data
    dates = [f't={i}' for i in range(n)]
    values = np.cumsum(np.random.normal(0.5, 1, n)) + 50

    # Correct rolling mean (only past data)
    ax = axes[0]
    window = 3
    rolling_correct = np.full(n, np.nan)
    for i in range(window, n):
        rolling_correct[i] = np.mean(values[i-window:i])  # Only past data

    ax.plot(values, 'b-o', markersize=6, label='Original', zorder=2)
    ax.plot(rolling_correct, 'g--s', markersize=5, label='Rolling Mean (correct)', zorder=3)

    # Highlight the calculation for t=5
    ax.annotate('', xy=(5, values[5]), xytext=(5, rolling_correct[5]+3),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(5, rolling_correct[5]+4, 'Uses t=2,3,4\n(past only)',
            ha='center', fontsize=8, color='green')

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('CORRECT: Rolling Mean Uses Past Only', fontsize=10, color='green')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False)

    # Wrong: centered rolling mean (uses future data)
    ax = axes[1]
    rolling_wrong = np.full(n, np.nan)
    for i in range(1, n-1):
        rolling_wrong[i] = np.mean(values[i-1:i+2])  # Uses future!

    ax.plot(values, 'b-o', markersize=6, label='Original', zorder=2)
    ax.plot(rolling_wrong, 'r--^', markersize=5, label='Centered Mean (WRONG)', zorder=3)

    # Highlight the problem for t=5
    ax.annotate('', xy=(5, values[5]), xytext=(5, rolling_wrong[5]-3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(5, rolling_wrong[5]-5, 'Uses t=4,5,6\n(INCLUDES FUTURE!)',
            ha='center', fontsize=8, color='red', fontweight='bold')

    # Highlight future data point
    ax.scatter([6], [values[6]], s=150, c='red', marker='X', zorder=4, label='Future data used!')

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('WRONG: Centered Mean Uses Future Data!', fontsize=10, color='red')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch8_quiz8_data_leakage')

ch8_quiz8_data_leakage()


# Quiz 9: Model Complexity vs Data Size
def ch8_quiz9_complexity_data():
    """Show relationship between model complexity and required data"""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ['ARIMA\n(~5 params)', 'ARFIMA\n(~6 params)', 'Random Forest\n(~1000 params)',
              'LSTM\n(~10,000+ params)']
    min_data = [50, 100, 500, 5000]
    recommended_data = [200, 500, 2000, 50000]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, min_data, width, label='Minimum Data', color='orange', alpha=0.7)
    bars2 = ax.bar(x + width/2, recommended_data, width, label='Recommended Data', color='steelblue', alpha=0.7)

    ax.set_ylabel('Number of Observations (log scale)')
    ax.set_title('Model Complexity vs Data Requirements', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=9, frameon=False)
    ax.set_yscale('log')
    ax.set_ylim(10, 100000)

    # Add text annotations
    for bar1, bar2, min_d, rec_d in zip(bars1, bars2, min_data, recommended_data):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() * 1.2,
                f'{min_d}', ha='center', fontsize=8, color='orange')
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() * 1.2,
                f'{rec_d}', ha='center', fontsize=8, color='steelblue')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch8_quiz9_complexity_data')

ch8_quiz9_complexity_data()


# Quiz 10: Evaluation metrics comparison
def ch8_quiz10_metrics():
    """Compare RMSE, MAE, and MAPE sensitivity"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    np.random.seed(42)

    # Scenario 1: Normal errors
    actual = np.array([100, 105, 98, 102, 110, 95, 108, 103, 99, 107])
    forecast_good = actual + np.random.normal(0, 2, 10)
    errors_good = actual - forecast_good

    ax = axes[0]
    ax.bar(range(10), np.abs(errors_good), color='steelblue', alpha=0.7, label='|Error|')
    ax.axhline(y=np.mean(np.abs(errors_good)), color='red', linestyle='--', linewidth=2, label=f'MAE = {np.mean(np.abs(errors_good)):.2f}')
    rmse = np.sqrt(np.mean(errors_good**2))
    ax.axhline(y=rmse, color='green', linestyle=':', linewidth=2, label=f'RMSE = {rmse:.2f}')
    ax.set_xlabel('Observation')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Normal Errors\nMAE $\\approx$ RMSE', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False)
    ax.set_ylim(0, 15)

    # Scenario 2: With outliers
    forecast_outlier = actual + np.random.normal(0, 2, 10)
    forecast_outlier[5] = actual[5] - 15  # Add outlier
    errors_outlier = actual - forecast_outlier

    ax = axes[1]
    colors = ['steelblue' if i != 5 else 'red' for i in range(10)]
    ax.bar(range(10), np.abs(errors_outlier), color=colors, alpha=0.7)
    ax.axhline(y=np.mean(np.abs(errors_outlier)), color='red', linestyle='--', linewidth=2,
               label=f'MAE = {np.mean(np.abs(errors_outlier)):.2f}')
    rmse_out = np.sqrt(np.mean(errors_outlier**2))
    ax.axhline(y=rmse_out, color='green', linestyle=':', linewidth=2,
               label=f'RMSE = {rmse_out:.2f}')
    ax.set_xlabel('Observation')
    ax.set_ylabel('Absolute Error')
    ax.set_title('With Outlier\nRMSE >> MAE (RMSE penalizes outliers)', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False)
    ax.set_ylim(0, 20)

    # Annotate outlier
    ax.annotate('Outlier!', xy=(5, np.abs(errors_outlier[5])), xytext=(6.5, 17),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch8_quiz10_metrics')

ch8_quiz10_metrics()


print("\nAll Chapter 8 quiz charts created successfully!")
