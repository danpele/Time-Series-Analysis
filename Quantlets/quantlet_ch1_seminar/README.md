# TSA Chapter 1 Seminar: Exercise Solutions

This folder contains Python solutions for Seminar 1 exercises.

## Exercise Solutions

| File | Exercise | Topic |
|------|----------|-------|
| `TSA_ch1_ex1_autocovariance.py` | Exercise 1 | Autocovariance and Autocorrelation |
| `TSA_ch1_ex2_random_walk.py` | Exercise 2 | Random Walk Properties |
| `TSA_ch1_ex3_ar1.py` | Exercise 3 | AR(1) Process |
| `TSA_ch1_ex4_ma1.py` | Exercise 4 | MA(1) Process |
| `TSA_ch1_python_viz.py` | Python Ex 1 | Data Loading and Visualization |
| `TSA_ch1_python_adf.py` | Python Ex 2 | ADF and KPSS Stationarity Tests |

## Features

Each solution file includes:
- **Step-by-step calculations** with explanations
- **Formulas** clearly shown
- **Visualizations** saved to `../../charts/`
- **Final answers** summary
- **Interpretation** of results

## Requirements

```bash
pip install numpy pandas matplotlib statsmodels scipy yfinance
```

## Usage

Run any exercise solution:

```bash
python TSA_ch1_ex1_autocovariance.py
python TSA_ch1_python_adf.py
```

## Learning Objectives

1. **Exercise 1**: Calculate ACF from autocovariances, conditional expectations for AR(1)
2. **Exercise 2**: Understand random walk properties (E[X_t], Var(X_t) grows with t)
3. **Exercise 3**: AR(1) stationarity condition, variance formula, ACF decay
4. **Exercise 4**: MA(1) properties, ACF cuts off after lag 1
5. **Python Ex 1**: Load real financial data, create visualizations
6. **Python Ex 2**: Apply formal stationarity tests, interpret results
