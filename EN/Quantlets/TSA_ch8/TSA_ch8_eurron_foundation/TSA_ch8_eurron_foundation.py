#!/usr/bin/env python3
"""
TSA_ch8_eurron_foundation
=========================
EUR/RON: Complete Model Comparison including Foundation Models

This is the quantlet version. The main script is at:
    generate_ch8_eurron_foundation.py (repository root)

Run from repository root:
    python generate_ch8_eurron_foundation.py

Models: ARIMA(1,1,1), ARFIMA(1,d,1), Random Forest, MLP/LSTM,
        Chronos (zero-shot), TimesFM (zero-shot)

Data: EUR/RON daily exchange rate (Yahoo Finance, 2019-2025)
Split: 70/15/15 temporal (train/validation/test)

Output charts (to charts/):
  - ch8_eurron_series.pdf       : Price + returns overview
  - ch8_case_raw_data.pdf       : Train/val/test split
  - ch8_case_acf_analysis.pdf   : ACF returns vs squared returns
  - ch8_case_feature_importance.pdf : RF feature importance
  - ch8_case_lstm_training.pdf  : MLP training/validation curves
  - ch8_case_predictions.pdf    : All classical models vs actual
  - ch8_case_comparison.pdf     : Bar chart RMSE/MAE/time
  - ch8_foundation_comparison.pdf : Classical vs Foundation bar chart
  - ch8_foundation_predictions.pdf : Chronos/TimesFM predictions + CI

Author: Daniel Traian Pele
"""

import os, sys

# Run the main script from repo root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
main_script = os.path.join(repo_root, 'generate_ch8_eurron_foundation.py')

if os.path.exists(main_script):
    exec(open(main_script).read())
else:
    print(f"Main script not found: {main_script}")
    print("Run from repository root: python generate_ch8_eurron_foundation.py")
