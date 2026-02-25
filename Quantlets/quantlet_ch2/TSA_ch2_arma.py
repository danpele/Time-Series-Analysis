"""
TSA_ch2_arma
============
ARMA(p,q) Process: Combining AR and MA

This script demonstrates:
- ARMA(1,1) model: X_t = φX_{t-1} + ε_t + θε_{t-1}
- Stationarity and invertibility
- ACF/PACF patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf, pacf

# --- Standard chart style ---
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# Set random seed
np.random.seed(42)

n = 500

print("=" * 60)
print("ARMA(p,q) PROCESS")
print("=" * 60)

print("""
ARMA(p,q) Model:
  X_t = c + φ₁X_{t-1} + ... + φₚX_{t-p} + ε_t + θ₁ε_{t-1} + ... + θqε_{t-q}

Lag Notation:
  φ(L)X_t = θ(L)ε_t

Stationarity: Roots of φ(z) = 0 outside unit circle
Invertibility: Roots of θ(z) = 0 outside unit circle

Special Cases:
  ARMA(p,0) = AR(p)
  ARMA(0,q) = MA(q)
""")

# Simulate ARMA(1,1) with different parameters
params = [
    (0.7, 0.4, 'ARMA(1,1): φ=0.7, θ=0.4'),
    (0.7, -0.4, 'ARMA(1,1): φ=0.7, θ=-0.4'),
    (-0.5, 0.6, 'ARMA(1,1): φ=-0.5, θ=0.6'),
    (0.9, 0.3, 'ARMA(1,1): φ=0.9, θ=0.3 (persistent)')
]

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for i, (phi, theta, title) in enumerate(params):
    # Create ARMA process
    ar = np.array([1, -phi])  # AR coefficients: 1 - φL
    ma = np.array([1, theta])  # MA coefficients: 1 + θL
    arma_process = ArmaProcess(ar, ma)

    # Simulate
    x = arma_process.generate_sample(nsample=n)

    # Time series plot
    axes[0, i].plot(x[:200], 'b-', linewidth=0.8, alpha=0.8)
    axes[0, i].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, i].set_title(title, fontsize=10)
    axes[0, i].set_xlabel('Time')
    axes[0, i].set_ylabel('X_t')
    axes[0, i].grid(False)

    # ACF plot
    acf_values = acf(x, nlags=15)
    axes[1, i].bar(range(16), acf_values, color='blue', alpha=0.7)
    axes[1, i].axhline(y=1.96/np.sqrt(n), color='red', linestyle='--')
    axes[1, i].axhline(y=-1.96/np.sqrt(n), color='red', linestyle='--')
    axes[1, i].set_title('ACF: decays (mixed pattern)', fontsize=10)
    axes[1, i].set_xlabel('Lag')
    axes[1, i].set_ylabel('ACF')
    axes[1, i].grid(False)

    # PACF plot
    pacf_values = pacf(x, nlags=15)
    axes[2, i].bar(range(16), pacf_values, color='green', alpha=0.7)
    axes[2, i].axhline(y=1.96/np.sqrt(n), color='red', linestyle='--')
    axes[2, i].axhline(y=-1.96/np.sqrt(n), color='red', linestyle='--')
    axes[2, i].set_title('PACF: decays (mixed pattern)', fontsize=10)
    axes[2, i].set_xlabel('Lag')
    axes[2, i].set_ylabel('PACF')
    axes[2, i].grid(False)

plt.tight_layout()
plt.savefig('../../charts/ch2_arma_properties.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch2_arma_properties.pdf', bbox_inches='tight', transparent=True)
plt.show()

# Model identification guide
print("\n" + "=" * 60)
print("MODEL IDENTIFICATION FROM ACF/PACF PATTERNS")
print("=" * 60)
print("""
┌─────────────┬──────────────────────┬──────────────────────┐
│   Model     │        ACF           │        PACF          │
├─────────────┼──────────────────────┼──────────────────────┤
│   AR(p)     │  Decays (exp/osc)    │  Cuts off at lag p   │
│   MA(q)     │  Cuts off at lag q   │  Decays (exp/osc)    │
│  ARMA(p,q)  │  Decays after lag q  │  Decays after lag p  │
└─────────────┴──────────────────────┴──────────────────────┘

ARMA Identification is HARD:
  - Both ACF and PACF decay
  - No clean cutoff
  - Use information criteria (AIC, BIC) to compare models
""")

# ARMA(1,1) formulas
print("\n" + "=" * 60)
print("ARMA(1,1) FORMULAS")
print("=" * 60)

phi, theta = 0.7, 0.4
sigma_sq = 1

print(f"""
Model: X_t = φX_{{t-1}} + ε_t + θε_{{t-1}}
       X_t = {phi}X_{{t-1}} + ε_t + {theta}ε_{{t-1}}

Mean: E[X_t] = 0 (assuming c = 0)

Variance:
  γ(0) = σ² × (1 + θ² + 2φθ) / (1 - φ²)
       = {sigma_sq} × (1 + {theta**2} + 2×{phi}×{theta}) / (1 - {phi**2})
       = {sigma_sq * (1 + theta**2 + 2*phi*theta) / (1 - phi**2):.4f}

ACF at lag 1:
  ρ(1) = (φ + θ)(1 + φθ) / (1 + θ² + 2φθ)
       = ({phi} + {theta})(1 + {phi}×{theta}) / (1 + {theta**2} + 2×{phi}×{theta})
       = {(phi + theta)*(1 + phi*theta) / (1 + theta**2 + 2*phi*theta):.4f}

ACF at lag h > 1:
  ρ(h) = φ × ρ(h-1)  (decays like AR)
""")
