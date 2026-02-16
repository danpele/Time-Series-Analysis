# TSA Chapter 2: ARMA Models - Seminar Quantlets

This folder contains Python Quantlets for **Chapter 2: ARMA Models** seminar exercise solutions.

## Exercise Solutions

| Quantlet | Exercise | Problem |
|----------|----------|---------|
| `TSA_ch2_ex1_ar1.py` | Exercise 1 | AR(1): X_t = 2 + 0.7X_{t-1} + ε_t, σ² = 9. Find μ, γ(0), γ(1), γ(2), ρ(1), ρ(2) |
| `TSA_ch2_ex2_ma1.py` | Exercise 2 | MA(1): X_t = 5 + ε_t - 0.4ε_{t-1}, σ² = 4. Find μ, γ(0), γ(1), ρ(1), invertibility |
| `TSA_ch2_ex3_roots.py` | Exercise 3 | AR(2): X_t = 0.5X_{t-1} + 0.3X_{t-2} + ε_t. Find roots, check stationarity |
| `TSA_ch2_ex4_forecast.py` | Exercise 4 | AR(1): c=3, φ=0.8, σ²=4, X_100=20. Find 1-step, 2-step, long-run forecasts, 95% CI |

## Python Exercises

| Quantlet | Topic |
|----------|-------|
| `TSA_ch2_python_simulate.py` | Simulate AR(1), fit model, compare estimates with true values |
| `TSA_ch2_python_selection.py` | Model selection using AIC/BIC with multiple candidate models |
| `TSA_ch2_python_diagnostics.py` | Comprehensive residual diagnostics (Ljung-Box, normality, Q-Q) |

## Key Formulas

### AR(1) Properties
```
Mean:           μ = c / (1 - φ)
Variance:       γ(0) = σ² / (1 - φ²)
Autocovariance: γ(h) = φ^h × γ(0)
Autocorrelation: ρ(h) = φ^h
```

### MA(1) Properties
```
Mean:           E[X_t] = μ (the constant)
Variance:       γ(0) = σ²(1 + θ²)
Autocovariance: γ(1) = θσ², γ(h) = 0 for h > 1
Autocorrelation: ρ(1) = θ/(1 + θ²)
Invertibility:  |θ| < 1
```

### AR(2) Stationarity
```
Characteristic equation: 1 - φ₁z - φ₂z² = 0
Stationarity: all roots |z| > 1
Triangle conditions:
  |φ₂| < 1
  φ₁ + φ₂ < 1
  φ₂ - φ₁ < 1
```

### Forecasting
```
Point forecast: X̂_{n+h|n} = μ + φ^h(X_n - μ)
Long-run:       lim X̂_{n+h|n} = μ as h → ∞
MSFE(1):        σ²
95% CI:         X̂ ± 1.96 × √MSFE
```

## Output

Each script produces:
1. **Step-by-step solution** printed to console
2. **Visualization** with multiple panels
3. **Summary box** with all key results
4. **Charts** saved to `../../charts/`

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
cd quantlet_ch2_seminar
python TSA_ch2_ex1_ar1.py
```

## Author

Time Series Analysis Course
Bucharest University of Economic Studies
