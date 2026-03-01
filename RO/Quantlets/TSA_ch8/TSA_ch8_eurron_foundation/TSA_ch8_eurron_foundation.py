#!/usr/bin/env python3
"""
TSA_ch8_eurron_foundation
=========================
EUR/RON: Complete Model Comparison including Foundation Models

This is the quantlet version. The main script is at:
    generate_ch8_eurron_foundation.py (repository root)

Run from repository root:
    python generate_ch8_eurron_foundation.py

Author: Daniel Traian Pele
"""

import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
main_script = os.path.join(repo_root, 'generate_ch8_eurron_foundation.py')

if os.path.exists(main_script):
    exec(open(main_script).read())
else:
    print(f"Main script not found: {main_script}")
    print("Run from repository root: python generate_ch8_eurron_foundation.py")
