#!/usr/bin/env python3
"""
TSA_ch11_eurron_foundation
===========================
Chapter 11: LLMs and Foundation Models for Time Series

This is the quantlet version. The main script is at:
    generate_ch11_charts.py (repository root)

Run from repository root:
    python generate_ch11_charts.py

Models: ARIMA, Chronos, TimesFM, Lag-Llama, Moirai, TimeGPT (+ classical baselines)

Data: EUR/RON daily exchange rate (Yahoo Finance, 2019-2025),
      synthetic energy data, S&P 500 realized volatility

Output: 27 charts (to charts/) covering theory diagrams, use cases, and quizzes.

Author: Daniel Traian Pele
"""

import os, sys

# Run the main script from repo root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
main_script = os.path.join(repo_root, 'generate_ch11_charts.py')

if os.path.exists(main_script):
    exec(open(main_script).read())
else:
    print(f"Main script not found: {main_script}")
    print("Run from repository root: python generate_ch11_charts.py")
