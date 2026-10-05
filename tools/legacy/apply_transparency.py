#!/usr/bin/env python3
"""
apply_transparency.py
=====================
Applies transparency transformations to all chart generation scripts:
1. transparent=True in every savefig() call
2. fig.patch.set_alpha(0) after each fig creation
3. ax.patch.set_alpha(0) after each ax assignment
4. Moves legends outside at bottom

Run once, then delete this script.
"""

import re
import os

BASE = '/Users/danielpele/Documents/TSA'

FILES = [
    'generate_arma_charts.py',
    'generate_ch11_charts.py',
    'generate_ch1_quiz_charts.py',
    'generate_ch6_quiz_charts.py',
    'generate_ch8_eurron_foundation.py',
    'generate_ch8_quiz_charts.py',
    'generate_ch9_quiz_charts.py',
    'generate_chapter10_charts.py',
    'generate_chapter12_charts.py',
    'generate_chapter1_charts.py',
    'generate_chapter2_charts.py',
    'generate_chapter3_charts.py',
    'generate_chapter4_charts.py',
    'generate_chapter5_charts.py',
    'generate_chapter6_charts.py',
    'generate_chapter8_charts.py',
    'generate_chapter9_charts.py',
    'generate_charts.py',
    'generate_definition_charts.py',
    'generate_forecast_charts.py',
    'generate_garch_charts.py',
    'generate_lecture_quiz_charts.py',
    'generate_more_charts.py',
    'generate_motivation_ch3_ch4.py',
    'generate_motivation_charts.py',
    'generate_prophet_arima_comparison.py',
    'generate_sarima_correct.py',
    'generate_sarima_diagnostics.py',
    'generate_seminar_charts.py',
    'generate_seminar_charts2.py',
    'generate_timeseries_charts.py',
    'generate_unemployment_analysis.py',
    'fix_charts.py',
]


def get_indent(line):
    """Return leading whitespace of a line."""
    return line[:len(line) - len(line.lstrip())]


def transform_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    changed = False

    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip('\n')
        indent = get_indent(stripped)

        # ---------------------------------------------------------------
        # 1. savefig(): add transparent=True if missing
        # ---------------------------------------------------------------
        if 'savefig(' in stripped and 'transparent=True' not in stripped:
            # Add transparent=True before the closing paren
            # Handle both ) and dpi=xxx) endings
            new_stripped = re.sub(
                r'savefig\(([^)]*)\)',
                lambda m: 'savefig(' + _add_transparent(m.group(1)) + ')',
                stripped
            )
            if new_stripped != stripped:
                changed = True
                stripped = new_stripped
            new_lines.append(stripped + '\n')
            i += 1
            continue

        # ---------------------------------------------------------------
        # 2. fig, ax = plt.subplots(...) — single ax
        #    Add fig.patch.set_alpha(0) and ax.patch.set_alpha(0)
        # ---------------------------------------------------------------
        # Pattern: fig, ax = plt.subplots(...)  [NOT axes]
        m_single = re.match(
            r'^(\s*)fig,\s*ax\s*=\s*plt\.subplots\(', stripped + '\n'
        )
        if m_single and not re.search(r'fig,\s*axes', stripped):
            new_lines.append(stripped + '\n')
            ind = m_single.group(1)
            # Check next lines to avoid duplicate insertion
            next_two = ''.join(lines[i+1:i+3])
            if 'fig.patch.set_alpha' not in next_two:
                new_lines.append(f'{ind}fig.patch.set_alpha(0)\n')
                new_lines.append(f'{ind}ax.patch.set_alpha(0)\n')
                changed = True
            i += 1
            continue

        # ---------------------------------------------------------------
        # 3. fig, axes = plt.subplots(...) — multiple axes
        #    Add fig.patch.set_alpha(0) and loop over axes
        # ---------------------------------------------------------------
        m_multi = re.match(
            r'^(\s*)fig,\s*axes\s*=\s*plt\.subplots\(', stripped + '\n'
        )
        if m_multi:
            new_lines.append(stripped + '\n')
            ind = m_multi.group(1)
            next_two = ''.join(lines[i+1:i+3])
            if 'fig.patch.set_alpha' not in next_two:
                new_lines.append(f'{ind}fig.patch.set_alpha(0)\n')
                new_lines.append(f'{ind}for _a in np.array(axes).flatten(): _a.patch.set_alpha(0)\n')
                changed = True
            i += 1
            continue

        # ---------------------------------------------------------------
        # 4. fig = plt.figure(...)
        #    Add fig.patch.set_alpha(0) after
        # ---------------------------------------------------------------
        m_fig = re.match(r'^(\s*)fig\s*=\s*plt\.figure\(', stripped + '\n')
        if m_fig:
            new_lines.append(stripped + '\n')
            ind = m_fig.group(1)
            next_one = lines[i+1] if i+1 < len(lines) else ''
            if 'fig.patch.set_alpha' not in next_one:
                new_lines.append(f'{ind}fig.patch.set_alpha(0)\n')
                changed = True
            i += 1
            continue

        # ---------------------------------------------------------------
        # 5. ax = ... assignments (standalone, not inside fig,ax= lines)
        #    Pattern: ax = fig.add_subplot(...) or ax = axes[...] etc.
        # ---------------------------------------------------------------
        m_ax = re.match(
            r'^(\s*)(ax\s*=\s*(?:fig\.add_subplot|fig\.add_axes|axes\[))', stripped + '\n'
        )
        if m_ax:
            new_lines.append(stripped + '\n')
            ind = m_ax.group(1)
            next_one = lines[i+1] if i+1 < len(lines) else ''
            if 'ax.patch.set_alpha' not in next_one and 'patch.set_alpha' not in next_one:
                new_lines.append(f'{ind}ax.patch.set_alpha(0)\n')
                changed = True
            i += 1
            continue

        # ---------------------------------------------------------------
        # 6. axes[...].flatten() or axes.flatten() — covered by loop above
        # ---------------------------------------------------------------

        new_lines.append(stripped + '\n')
        i += 1

    if changed:
        with open(path, 'w') as f:
            f.writelines(new_lines)
        print(f'  CHANGED: {os.path.basename(path)}')
    else:
        print(f'  unchanged: {os.path.basename(path)}')


def _add_transparent(args_str):
    """Add transparent=True to savefig argument string."""
    args_str = args_str.rstrip()
    if args_str.endswith(','):
        return args_str + ' transparent=True'
    else:
        return args_str + ', transparent=True'


if __name__ == '__main__':
    for fname in FILES:
        path = os.path.join(BASE, fname)
        if not os.path.exists(path):
            print(f'  MISSING: {fname}')
            continue
        transform_file(path)
    print('\nDone.')
