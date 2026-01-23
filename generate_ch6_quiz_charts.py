#!/usr/bin/env python3
"""
Generate charts for Chapter 6 seminar quiz answers
Cointegration and VECM
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
# CHAPTER 6: Cointegration and VECM Quiz Charts
# =============================================================================
print("Chapter 6: Cointegration & VECM Quiz Charts")

# Quiz 1: Cointegration concept
def ch6_quiz1_cointegration():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    n = 150
    # Common stochastic trend
    eps = np.random.normal(0, 1, n)
    tau = np.cumsum(eps)  # Random walk (common trend)

    # Two cointegrated series
    y1 = tau + np.random.normal(0, 0.5, n)
    y2 = 0.8 * tau + np.random.normal(0, 0.5, n)

    # Spread (cointegrating relationship)
    spread = y1 - 1.25 * y2  # Should be stationary

    axes[0].plot(y1, 'b-', linewidth=1, label='$Y_t$')
    axes[0].plot(y2, 'r-', linewidth=1, label='$X_t$')
    axes[0].set_title('Both Series: I(1)', fontsize=10)
    axes[0].set_xlabel('Time')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    axes[1].plot(y1, y2, 'g.', alpha=0.5, markersize=3)
    z = np.polyfit(y2, y1, 1)
    p = np.poly1d(z)
    axes[1].plot(y2, p(y2), 'r-', linewidth=2, label=f'$Y = {z[0]:.2f}X$')
    axes[1].set_xlabel('$X_t$')
    axes[1].set_ylabel('$Y_t$')
    axes[1].set_title('Long-run Relationship', fontsize=10)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

    axes[2].plot(spread, 'g-', linewidth=1)
    axes[2].axhline(y=np.mean(spread), color='red', linestyle='--', alpha=0.7)
    axes[2].fill_between(range(n), np.mean(spread) - 1.96*np.std(spread),
                         np.mean(spread) + 1.96*np.std(spread), alpha=0.2, color='green')
    axes[2].set_title('Spread $Y_t - \\beta X_t$: I(0)', fontsize=10, color='green')
    axes[2].set_xlabel('Time')

    plt.tight_layout()
    save_fig('ch6_quiz1_cointegration')

ch6_quiz1_cointegration()

# Quiz 2: Spurious regression
def ch6_quiz2_spurious():
    np.random.seed(123)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    n = 100
    # Two INDEPENDENT random walks
    y1 = np.cumsum(np.random.normal(0, 1, n))
    y2 = np.cumsum(np.random.normal(0, 1, n))

    # Regression
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(y2, y1)

    axes[0].plot(y1, 'b-', linewidth=1.5, label='$Y_t$ (Random Walk 1)')
    axes[0].plot(y2, 'r-', linewidth=1.5, label='$X_t$ (Random Walk 2)')
    axes[0].set_xlabel('Time')
    axes[0].set_title('Two INDEPENDENT Random Walks', fontsize=10)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    axes[1].scatter(y2, y1, alpha=0.5, s=30)
    axes[1].plot(y2, intercept + slope * y2, 'r-', linewidth=2)
    axes[1].set_xlabel('$X_t$')
    axes[1].set_ylabel('$Y_t$')
    axes[1].set_title(f'Spurious Regression\n$R^2 = {r_value**2:.3f}$, but variables are unrelated!',
                      fontsize=10, color='red')

    # Add text box with warning
    textstr = f'Slope: {slope:.2f}\np-value: {p_value:.4f}\nDW $\\approx$ 0.1'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes, fontsize=9,
                 verticalalignment='top', bbox=props)

    plt.tight_layout()
    save_fig('ch6_quiz2_spurious')

ch6_quiz2_spurious()

# Quiz 4: Johansen test - multiple cointegrating vectors
def ch6_quiz4_johansen():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Eigenvalues visualization
    eigenvalues = [0.45, 0.18, 0.05]
    colors = ['green', 'green', 'gray']
    labels = ['Significant\n(r=1)', 'Significant\n(r=2)', 'Not sig.\n(r<3)']

    x = np.arange(3)
    bars = axes[0].bar(x, eigenvalues, color=colors, alpha=0.7, edgecolor='black')
    axes[0].axhline(y=0.12, color='red', linestyle='--', linewidth=2, label='5% Threshold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['$\\lambda_1$', '$\\lambda_2$', '$\\lambda_3$'])
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_title('Johansen Eigenvalues', fontsize=10)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

    for i, (bar, label) in enumerate(zip(bars, labels)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     label, ha='center', va='bottom', fontsize=8)

    # Comparison table visualization
    axes[1].axis('off')
    table_data = [
        ['Method', 'Engle-Granger', 'Johansen'],
        ['# of CI vectors', '1 only', 'Multiple'],
        ['Dep. variable', 'Required', 'Not needed'],
        ['Estimation', 'Two-step', 'MLE'],
        ['Efficiency', 'Lower', 'Higher'],
    ]

    table = axes[1].table(cellText=table_data, loc='center', cellLoc='center',
                          colWidths=[0.35, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color header row
    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    axes[1].set_title('Johansen vs Engle-Granger', fontsize=11, pad=20)

    plt.tight_layout()
    save_fig('ch6_quiz4_johansen')

ch6_quiz4_johansen()

# Quiz 6: Adjustment coefficients
def ch6_quiz6_adjustment():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    np.random.seed(42)

    n = 100
    # Simulate VECM with different adjustment speeds
    alpha1, alpha2 = -0.3, 0.1
    beta = 1.0

    y1 = np.zeros(n)
    y2 = np.zeros(n)
    y1[0], y2[0] = 2, 0  # Start with disequilibrium

    for t in range(1, n):
        ec = y1[t-1] - beta * y2[t-1]  # Error correction term
        y1[t] = y1[t-1] + alpha1 * ec + np.random.normal(0, 0.3)
        y2[t] = y2[t-1] + alpha2 * ec + np.random.normal(0, 0.3)

    axes[0].plot(y1, 'b-', linewidth=1.5, label='$Y_1$ ($\\alpha_1 = -0.3$)')
    axes[0].plot(y2, 'r-', linewidth=1.5, label='$Y_2$ ($\\alpha_2 = +0.1$)')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Time')
    axes[0].set_title('VECM: Both Variables Adjust', fontsize=10)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    # Adjustment diagram
    axes[1].axis('off')
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 8)

    # Equilibrium line
    axes[1].plot([1, 9], [4, 4], 'g-', linewidth=3, label='Equilibrium')
    axes[1].text(5, 4.5, 'Long-run: $Y_1 = \\beta Y_2$', ha='center', fontsize=10, color='green')

    # Adjustment arrows
    # Y1 above equilibrium
    axes[1].annotate('', xy=(3, 4.2), xytext=(3, 6),
                     arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    axes[1].text(3, 6.3, '$Y_1$ adjusts down\n($\\alpha_1 < 0$)', ha='center', fontsize=9, color='blue')

    # Y2 adjusts up
    axes[1].annotate('', xy=(7, 3.8), xytext=(7, 2),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))
    axes[1].text(7, 1.5, '$Y_2$ adjusts up\n($\\alpha_2 > 0$)', ha='center', fontsize=9, color='red')

    axes[1].set_title('Error Correction Mechanism', fontsize=11)

    plt.tight_layout()
    save_fig('ch6_quiz6_adjustment')

ch6_quiz6_adjustment()

# Additional quiz charts

# Engle-Granger critical values visualization
def ch6_quiz3_eg_critical():
    fig, ax = plt.subplots(figsize=(8, 4))

    # Critical values
    cv_1 = -3.90
    cv_5 = -3.34
    cv_10 = -3.04
    test_stat = -3.92  # From the problem

    # Color regions
    ax.axvspan(-5, cv_1, alpha=0.3, color='green', label='Reject at 1%')
    ax.axvspan(cv_1, cv_5, alpha=0.3, color='lightgreen', label='Reject at 5%')
    ax.axvspan(cv_5, cv_10, alpha=0.3, color='yellow', label='Reject at 10%')
    ax.axvspan(cv_10, 0, alpha=0.3, color='red', label='Cannot Reject')

    # Critical values lines
    ax.axvline(x=cv_1, color='darkgreen', linestyle='--', linewidth=2)
    ax.axvline(x=cv_5, color='green', linestyle='--', linewidth=2)
    ax.axvline(x=cv_10, color='olive', linestyle='--', linewidth=2)

    # Test statistic
    ax.axvline(x=test_stat, color='blue', linestyle='-', linewidth=3, label=f'Test Stat = {test_stat}')
    ax.scatter([test_stat], [0.5], s=200, c='blue', zorder=5, marker='v')

    # Labels
    ax.text(cv_1, 0.85, '1%\n-3.90', ha='center', fontsize=9)
    ax.text(cv_5, 0.85, '5%\n-3.34', ha='center', fontsize=9)
    ax.text(cv_10, 0.85, '10%\n-3.04', ha='center', fontsize=9)
    ax.text(test_stat + 0.1, 0.6, f'ADF = {test_stat}', ha='left', fontsize=10, color='blue')

    ax.set_xlim(-5, 0)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Engle-Granger Test Statistic')
    ax.set_title('Cointegration Test: Reject $H_0$ (Cointegrated!)', color='green')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=False)
    ax.set_yticks([])

    plt.tight_layout()
    save_fig('ch6_quiz3_eg_critical')

ch6_quiz3_eg_critical()

# VECM impulse response
def ch6_quiz_vecm_irf():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Simulate VECM IRF
    h = 20
    t = np.arange(h)

    # Response in cointegrated system - permanent effect but convergence to equilibrium
    irf_y1_shock1 = 1 * np.exp(-0.1 * t) + 0.5 * (1 - np.exp(-0.1 * t))
    irf_y2_shock1 = 0.3 * (1 - np.exp(-0.15 * t))

    axes[0].plot(t, irf_y1_shock1, 'b-', linewidth=2, label='$Y_1$ response')
    axes[0].plot(t, irf_y2_shock1, 'r-', linewidth=2, label='$Y_2$ response')
    axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[0].fill_between(t, irf_y1_shock1 - 0.15, irf_y1_shock1 + 0.15, alpha=0.2, color='blue')
    axes[0].fill_between(t, irf_y2_shock1 - 0.1, irf_y2_shock1 + 0.1, alpha=0.2, color='red')
    axes[0].set_xlabel('Horizon')
    axes[0].set_ylabel('Response')
    axes[0].set_title('IRF: Shock to $Y_1$', fontsize=10)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    # Key difference from stationary VAR
    # Stationary
    irf_stationary = np.exp(-0.2 * t)
    # Cointegrated (permanent effect)
    irf_cointegrated = 0.3 + 0.7 * np.exp(-0.15 * t)

    axes[1].plot(t, irf_stationary, 'g--', linewidth=2, label='Stationary VAR: decays to 0')
    axes[1].plot(t, irf_cointegrated, 'b-', linewidth=2, label='VECM: permanent effect')
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[1].axhline(y=0.3, color='blue', linestyle=':', alpha=0.7)
    axes[1].set_xlabel('Horizon')
    axes[1].set_ylabel('Response')
    axes[1].set_title('VECM vs VAR: Permanent vs Transitory', fontsize=10)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

    plt.tight_layout()
    save_fig('ch6_quiz_vecm_irf')

ch6_quiz_vecm_irf()

# Weak exogeneity visualization
def ch6_quiz_weak_exog():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    np.random.seed(42)

    n = 100

    # Case 1: Both adjust
    alpha1, alpha2 = -0.3, 0.2
    y1_both = np.zeros(n)
    y2_both = np.zeros(n)
    y1_both[0], y2_both[0] = 2, 0

    for t in range(1, n):
        ec = y1_both[t-1] - y2_both[t-1]
        y1_both[t] = y1_both[t-1] + alpha1 * ec + np.random.normal(0, 0.2)
        y2_both[t] = y2_both[t-1] + alpha2 * ec + np.random.normal(0, 0.2)

    spread_both = y1_both - y2_both

    # Case 2: Y2 weakly exogenous (alpha2 = 0)
    alpha1, alpha2 = -0.3, 0.0
    y1_exog = np.zeros(n)
    y2_exog = np.zeros(n)
    y1_exog[0], y2_exog[0] = 2, 0

    for t in range(1, n):
        ec = y1_exog[t-1] - y2_exog[t-1]
        y1_exog[t] = y1_exog[t-1] + alpha1 * ec + np.random.normal(0, 0.2)
        y2_exog[t] = y2_exog[t-1] + 0 * ec + np.random.normal(0, 0.2)  # Random walk

    spread_exog = y1_exog - y2_exog

    axes[0].plot(spread_both, 'b-', linewidth=1.5)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Spread')
    axes[0].set_title('Both Adjust ($\\alpha_1=-0.3$, $\\alpha_2=0.2$)', fontsize=10)

    axes[1].plot(spread_exog, 'g-', linewidth=1.5)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Spread')
    axes[1].set_title('$Y_2$ Weakly Exogenous ($\\alpha_2=0$)', fontsize=10)

    plt.tight_layout()
    save_fig('ch6_quiz_weak_exog')

ch6_quiz_weak_exog()

# Trace test sequential procedure
def ch6_quiz_trace_test():
    fig, ax = plt.subplots(figsize=(9, 5))

    # Test statistics and critical values for k=3 variables
    r_values = [0, 1, 2]
    trace_stats = [45.2, 18.1, 3.2]
    critical_values = [29.8, 15.5, 3.8]

    x = np.arange(3)
    width = 0.35

    bars1 = ax.bar(x - width/2, trace_stats, width, label='Trace Statistic', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, critical_values, width, label='5% Critical Value', color='coral', alpha=0.8)

    # Add result annotations
    results = ['Reject', 'Reject', 'Fail to Reject']
    colors = ['green', 'green', 'red']
    for i, (bar, result, color) in enumerate(zip(bars1, results, colors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                result, ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

    ax.set_xlabel('Null Hypothesis')
    ax.set_ylabel('Test Statistic')
    ax.set_title('Johansen Trace Test: Sequential Procedure')
    ax.set_xticks(x)
    ax.set_xticklabels(['$H_0: r=0$', '$H_0: r \\leq 1$', '$H_0: r \\leq 2$'])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

    # Add conclusion box
    textstr = 'Conclusion: r = 2\n(Two cointegrating vectors)'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    save_fig('ch6_quiz_trace_test')

ch6_quiz_trace_test()

print("\nAll Chapter 6 quiz charts created successfully!")
print(f"Total charts created in {output_dir}/")
