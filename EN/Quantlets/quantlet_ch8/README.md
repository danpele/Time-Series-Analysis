# Chapter 8: Modern Extensions - Quantlet Codes

Jupyter notebooks with **real data** for Chapter 8.

## Contents

| Notebook | Description | Data Source |
|----------|-------------|-------------|
| `ch8_acf_comparison.ipynb` | ACF: Short vs Long Memory | EUR/RON (Yahoo Finance) |
| `ch8_hurst_estimation.ipynb` | Hurst exponent estimation | EUR/RON, S&P 500, Bitcoin, Gold |
| `ch8_random_forest.ipynb` | Random Forest forecasting | Germany electricity (OPSD) |
| `ch8_model_comparison.ipynb` | ARIMA vs RF vs LSTM comparison | Germany electricity (OPSD) |

## Data Sources

- **EUR/RON Exchange Rate**: Yahoo Finance (`EURRON=X`)
- **S&P 500 Index**: Yahoo Finance (`^GSPC`)
- **Bitcoin**: Yahoo Finance (`BTC-USD`)
- **Germany Electricity**: Open Power System Data (ENTSO-E)

## Requirements

```
pip install numpy pandas matplotlib scipy statsmodels scikit-learn yfinance tensorflow
```

## Author

Daniel Traian PELE
Bucharest University of Economic Studies
