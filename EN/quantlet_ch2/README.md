# TSA Chapter 2: ARMA Models - Lecture Quantlets

This folder contains Python Quantlets for **Chapter 2: ARMA Models** lecture demonstrations.

## Quantlets

| Quantlet | Description |
|----------|-------------|
| `TSA_ch2_lag_operator.py` | Lag operator L, difference operator Δ, lag polynomial notation |
| `TSA_ch2_ar1.py` | AR(1) properties: mean, variance, ACF geometric decay ρ(h) = φ^h |
| `TSA_ch2_ar2.py` | AR(2) stationarity, characteristic roots, stationarity triangle |
| `TSA_ch2_ma1.py` | MA(1) properties, ACF cutoff, invertibility condition |
| `TSA_ch2_arma.py` | ARMA(1,1) simulation, combined ACF/PACF patterns |
| `TSA_ch2_model_selection.py` | AIC/BIC comparison, model selection workflow |
| `TSA_ch2_diagnostics.py` | Ljung-Box test, residual ACF, Q-Q plots |
| `TSA_ch2_forecasting.py` | Point forecasts, confidence intervals, mean reversion |

## Key Concepts Covered

### Lag Operator
- L operator: LX_t = X_{t-1}
- Difference operator: ΔX_t = (1-L)X_t
- Lag polynomials: φ(L) = 1 - φ₁L - φ₂L² - ...

### AR(p) Models
- X_t = c + φ₁X_{t-1} + ... + φ_pX_{t-p} + ε_t
- Mean: μ = c / (1 - φ₁ - ... - φ_p)
- Stationarity: roots of φ(z) = 0 outside unit circle
- ACF: exponential/oscillating decay
- PACF: cuts off after lag p

### MA(q) Models
- X_t = μ + ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}
- ACF: cuts off after lag q
- PACF: exponential/oscillating decay
- Invertibility: roots of θ(z) = 0 outside unit circle

### ARMA(p,q) Models
- Combines AR and MA components
- ACF and PACF both decay (no sharp cutoff)
- Model selection via AIC/BIC

### Diagnostics
- Ljung-Box test: Q = n(n+2) Σ ρ̂(h)²/(n-h)
- H₀: residuals are white noise
- Q-Q plot for normality check

## Requirements

```python
numpy
matplotlib
scipy
statsmodels
pandas
```

## Usage

```bash
cd quantlet_ch2
python TSA_ch2_ar1.py
```

Charts are saved to `../../charts/` directory.

## Author

Time Series Analysis Course
Bucharest University of Economic Studies
