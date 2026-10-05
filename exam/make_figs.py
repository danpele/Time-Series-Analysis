#!/usr/bin/env python3
"""
make_figs.py -- every chart and every software output of the TSA exam materials, regenerated from the course data
(Quantlets/common/tsa_data.py: data/market, last day 18.09.2026; FRED, Eurostat and the statsmodels data sets online)
in the course style (Quantlets/common/tsa_style.py: transparent background, legend below the plot, no grey).

  exam/practice/figs/*_{ro,en}.pdf   practice set (public): charts, made by exam/practice/figs_practice.py
  exam/practice/out/*.txt            practice set: statsmodels / arch outputs printed as text (\\softout{...})
  exam/practice/practice_numbers.json   the numbers quoted in the practice set and its answer key
  exam/bank/figs/*.pdf               problem bank (instructor only, git-ignored): every exam/bank/figs_chNN.py
  exam/bank/out/*.txt                module defines make(fig_dir, out_dir) -> dict of numbers
  exam/bank/bank_numbers.json        the numbers quoted in the bank problems and solutions

Shared helpers: exam/exam_common.py.  Run from anywhere:  python3 exam/make_figs.py  [--only practice|bank]
                                                         python3 exam/make_figs.py --only bank --ch 2,5
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exam_common as ec   # noqa: E402

DIR_PRACTICE = os.path.join(HERE, 'practice')
DIR_BANK = os.path.join(HERE, 'bank')


def _load(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return name, mod


def _merge_json(path, new):
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(new)
    json.dump(old, open(path, 'w'), indent=1, default=float, ensure_ascii=False)
    print('   numbers ->', os.path.relpath(path, HERE))


def make_practice():
    m = os.path.join(DIR_PRACTICE, 'figs_practice.py')
    if not os.path.exists(m):
        print('practice set: no figs_practice.py')
        return {}
    print('practice set')
    _, mod = _load(m)
    num = mod.make(os.path.join(DIR_PRACTICE, 'figs'), os.path.join(DIR_PRACTICE, 'out')) or {}
    out = os.path.join(DIR_PRACTICE, 'practice_numbers.json')
    json.dump(num, open(out, 'w'), indent=1, default=float, ensure_ascii=False)
    print('   numbers ->', os.path.relpath(out, HERE))
    return num


def make_bank(chapters=None):
    mods = sorted(glob.glob(os.path.join(DIR_BANK, 'figs_ch*.py')))
    if chapters:
        mods = [m for m in mods if int(os.path.basename(m)[7:9]) in chapters]
    if not mods:
        print('problem bank: no figure modules (exam/bank is instructor-only and may be absent)')
        return {}
    print('problem bank')
    num = {}
    for m in mods:
        name, mod = _load(m)
        print(' ', name)
        num[name] = mod.make(os.path.join(DIR_BANK, 'figs'), os.path.join(DIR_BANK, 'out')) or {}
    _merge_json(os.path.join(DIR_BANK, 'bank_numbers.json'), num)
    return num


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['practice', 'bank'])
    ap.add_argument('--ch', default='', help='bank chapters only, e.g. 2,5')
    a = ap.parse_args()
    ec.style()
    if a.only in (None, 'practice'):
        make_practice()
    if a.only in (None, 'bank'):
        make_bank({int(c) for c in a.ch.split(',') if c.strip()} or None)
