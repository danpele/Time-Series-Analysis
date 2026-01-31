# TSA Chapter 3: ARIMA Models - Quantlets

This folder contains reproducible code and charts for Chapter 3: ARIMA Models.

## Quantlets

| Quantlet | Description |
|----------|-------------|
| TSA_ch3_gdp_levels | US Real GDP visualization (non-stationary series) |
| TSA_ch3_differencing | Differencing demonstration |
| TSA_ch3_acf_pacf | ACF/PACF analysis |
| TSA_ch3_arima_forecast | ARIMA forecasting example |
| TSA_ch3_adf_test | ADF unit root test visualization |
| TSA_ch3_diagnostics | Model diagnostic plots |
| TSA_ch3_case_raw_data | Case Study: US Real GDP raw data |
| TSA_ch3_case_adf_test | Case Study: ADF test results |
| TSA_ch3_case_acf_diff | Case Study: ACF before/after differencing |
| TSA_ch3_case_model_comparison | Case Study: ARIMA model selection |
| TSA_ch3_case_diagnostics | Case Study: Residual diagnostics |
| TSA_ch3_case_forecast | Case Study: 8-quarter GDP forecast |

## Data Source

All case study quantlets use real data from:
- **FRED (Federal Reserve Economic Data)**
- Series: GDPC1 (Real Gross Domestic Product)
- Frequency: Quarterly, Seasonally Adjusted
- Units: Billions of Chained 2017 Dollars

## Requirements

```python
pip install numpy pandas matplotlib statsmodels pandas-datareader scipy
```

## Usage

Each quantlet folder contains:
- Python script (`.py`) - Reproducible code
- PDF chart (`.pdf`) - Generated visualization

Run any script to reproduce the chart:
```bash
cd TSA_ch3_gdp_levels
python TSA_ch3_gdp_levels.py
```

## Author

Time Series Analysis Course
