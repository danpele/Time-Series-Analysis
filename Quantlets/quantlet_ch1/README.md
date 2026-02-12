# TSA Chapter 1: Stationarity and Basic Concepts

This folder contains Python Quantlets for Chapter 1 - Stationarity.

## Quantlets

| File | Topic | Description |
|------|-------|-------------|
| `TSA_ch1_ts_basics.py` | Time Series Patterns | Trend, seasonality, cycles, noise |
| `TSA_ch1_white_noise.py` | White Noise | Properties and Ljung-Box test |
| `TSA_ch1_random_walk.py` | Random Walk | Comparison with white noise |
| `TSA_ch1_stationarity.py` | Stationarity Types | Strict vs weak (covariance) stationarity |
| `TSA_ch1_acf_patterns.py` | ACF Patterns | ACF for different process types |
| `TSA_ch1_unit_root_tests.py` | Unit Root Tests | ADF and KPSS tests |
| `TSA_ch1_trend_types.py` | Trend Types | Deterministic vs stochastic trends |
| `TSA_ch1_differencing.py` | Differencing | Regular and seasonal differencing |
| `TSA_ch2_ma1.py` | MA(1) Process | Properties, ACF, PACF patterns |

## Requirements

```bash
pip install numpy pandas matplotlib statsmodels scipy
```

## Usage

Each script can be run independently:

```bash
python TSA_ch1_stationarity.py
```

Charts are saved to `../../charts/` directory.

## Key Concepts Covered

1. **Stationarity**: E[X_t]=μ, Var(X_t)=σ², Cov(X_t,X_{t+k})=γ(k)
2. **White Noise**: Zero autocorrelation at all lags
3. **Random Walk**: X_t = X_{t-1} + e_t (unit root, non-stationary)
4. **Unit Root Tests**:
   - ADF: H₀ = unit root (non-stationary)
   - KPSS: H₀ = stationary
5. **Differencing**: ΔX_t = X_t - X_{t-1} removes unit root
6. **ACF Patterns**: Key to identifying ARMA processes
