# TSA Chapter 0: Fundamentals of Time Series Analysis

This folder contains Python Quantlets for Chapter 0 - Fundamentals.

## Quantlets

| File | Topic | Description |
|------|-------|-------------|
| `TSA_ch0_definition.py` | Time Series Basics | Definition, types, and autocorrelation |
| `TSA_ch0_additive_mult.py` | Decomposition Types | Additive vs multiplicative decomposition |
| `TSA_ch0_decomposition.py` | Classical Decomposition | Using statsmodels seasonal_decompose |
| `TSA_ch0_ma.py` | Moving Averages | Centered MA for trend extraction |
| `TSA_ch0_ses.py` | Exponential Smoothing | Simple Exponential Smoothing (SES) |
| `TSA_ch0_holt_winters.py` | Holt-Winters | Level, trend, and seasonal smoothing |
| `TSA_ch0_seasonal.py` | Seasonal Adjustment | Deseasonalization methods |
| `TSA_ch0_forecast_eval.py` | Forecast Evaluation | MAE, MSE, RMSE, MAPE metrics |
| `TSA_ch0_cv.py` | Cross-Validation | Time series CV (expanding/rolling window) |

## Requirements

```bash
pip install numpy pandas matplotlib statsmodels scipy
```

## Usage

Each script can be run independently:

```bash
python TSA_ch0_decomposition.py
```

Charts are saved to `../../charts/` directory.

## Key Concepts Covered

1. **Time Series Definition**: Ordered observations over time
2. **Decomposition**: X_t = T_t + S_t + e_t (additive) or X_t = T_t × S_t × e_t (multiplicative)
3. **Smoothing Methods**: Moving averages, exponential smoothing
4. **Forecast Evaluation**: Scale-dependent (MAE, RMSE) vs scale-independent (MAPE)
5. **Cross-Validation**: Never use standard k-fold CV for time series!
