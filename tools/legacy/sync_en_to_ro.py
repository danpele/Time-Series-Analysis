#!/usr/bin/env python3
"""
sync_en_to_ro.py - Synchronize EN Beamer formatting to match RO page counts.

For each chapter pair (EN .tex + RO .tex):
1. Parses both files into frames (1:1 matching by position)
2. Copies formatting structure from RO into EN:
   - cminipage wrappers
   - \setlength{\itemsep}{0pt} on lists
   - Font wrappers ({\small, {\footnotesize, \small, etc.)
   - Leading \vspace{-Xcm} compression
3. Converts existing \hfill\begin{minipage} to \begin{cminipage}
4. Ensures EN preamble has cminipage definition

Usage:
    python sync_en_to_ro.py ch3             # Single chapter
    python sync_en_to_ro.py ch3 ch4 ch5     # Multiple chapters
    python sync_en_to_ro.py all             # All 8 chapters
    python sync_en_to_ro.py ch3 --dry-run   # Preview only
"""

import re
import sys
import os


# ─── Frame Parsing ───────────────────────────────────────────────────────────

def find_frame_ranges(lines):
    """Find all frame boundaries as (start_line, end_line) tuples."""
    frames = []
    i = 0
    while i < len(lines):
        if r'\begin{frame}' in lines[i]:
            start = i
            depth = 0
            for j in range(i, len(lines)):
                if r'\begin{frame}' in lines[j]:
                    depth += 1
                if r'\end{frame}' in lines[j]:
                    depth -= 1
                    if depth == 0:
                        frames.append((start, j))
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1
    return frames


def extract_frame_text(lines, start, end):
    """Extract frame text from line range."""
    return '\n'.join(lines[start:end + 1])


# ─── Detection Functions ─────────────────────────────────────────────────────

def has_cminipage(text):
    return r'\begin{cminipage}' in text


def has_hfill_minipage(text):
    return r'\hfill\begin{minipage}' in text


def detect_font_wrapper(frame_text):
    """Detect frame-level font wrapper ({\small, {\footnotesize, \small, etc.)."""
    lines = frame_text.split('\n')
    for line in lines:
        stripped = line.strip()
        # Skip frame header, cminipage, vspace, comments
        if stripped.startswith(r'\begin{frame}'):
            continue
        if stripped.startswith(r'\end{frame}'):
            break
        if stripped.startswith(r'\begin{cminipage}'):
            continue
        if stripped.startswith(r'\vspace'):
            continue
        if stripped.startswith('%') or not stripped:
            continue
        # Check for font wrapper patterns
        if stripped in (r'{\small', r'{\footnotesize', r'{\scriptsize'):
            return stripped
        if stripped in (r'\small', r'\footnotesize', r'\scriptsize'):
            return stripped
        # Not a font wrapper - hit real content
        break
    return None


def detect_leading_vspace(frame_text):
    """Detect leading negative vspace in frame body."""
    lines = frame_text.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(r'\begin{frame}'):
            continue
        if stripped.startswith(r'\begin{cminipage}'):
            continue
        if stripped.startswith('%') or not stripped:
            continue
        m = re.match(r'(\\vspace\{-[^}]+\})', stripped)
        if m:
            return m.group(1)
        # Hit non-vspace content
        break
    return None


def count_itemsep(text):
    return len(re.findall(r'\\setlength\{\\itemsep\}', text))


def count_lists(text):
    return len(re.findall(r'\\begin\{(?:itemize|enumerate)\}', text))


def is_title_or_toc_frame(frame_text):
    """Check if frame is title page or outline (skip these)."""
    if r'\titlepage' in frame_text:
        return True
    return False


# ─── Transformation Functions ─────────────────────────────────────────────────

def convert_hfill_minipage(frame_text):
    """Convert \\hfill\\begin{minipage}{...} to \\begin{cminipage}{0.95\\textwidth}."""
    lines = frame_text.split('\n')
    result_lines = []
    i = 0
    converted = False

    while i < len(lines):
        if r'\hfill\begin{minipage}' in lines[i]:
            # Convert this line
            new_line = re.sub(
                r'\\hfill\\begin\{minipage\}\{[^}]+\}',
                r'\\begin{cminipage}{0.95\\textwidth}',
                lines[i]
            )
            result_lines.append(new_line)
            converted = True

            # Find matching \end{minipage} by tracking nesting
            depth = 1
            i += 1
            while i < len(lines) and depth > 0:
                opens = len(re.findall(r'\\begin\{minipage\}', lines[i]))
                closes = len(re.findall(r'\\end\{minipage\}', lines[i]))
                depth += opens - closes
                if depth <= 0:
                    result_lines.append(
                        lines[i].replace(r'\end{minipage}', r'\end{cminipage}', 1)
                    )
                else:
                    result_lines.append(lines[i])
                i += 1
        else:
            result_lines.append(lines[i])
            i += 1

    return '\n'.join(result_lines), converted


def add_cminipage(frame_text):
    """Wrap frame content in \\begin{cminipage}{0.95\\textwidth}."""
    lines = frame_text.split('\n')

    # Find frame title line
    title_idx = None
    for i, line in enumerate(lines):
        if r'\begin{frame}' in line:
            title_idx = i
            break
    if title_idx is None:
        return frame_text

    # Determine indentation from first content line
    indent = '    '
    for j in range(title_idx + 1, len(lines)):
        if lines[j].strip() and not lines[j].strip().startswith('%'):
            indent = ' ' * max(4, len(lines[j]) - len(lines[j].lstrip()))
            break

    # Insert \begin{cminipage} right after frame title
    lines.insert(title_idx + 1, indent + r'\begin{cminipage}{0.95\textwidth}')

    # Find \end{frame} and insert \end{cminipage} before it
    for j in range(len(lines) - 1, -1, -1):
        if r'\end{frame}' in lines[j]:
            lines.insert(j, indent + r'\end{cminipage}')
            break

    return '\n'.join(lines)


def add_itemsep_to_lists(frame_text):
    """Add \\setlength{\\itemsep}{0pt} to lists that don't have it."""
    # Add to \begin{itemize}[optional] not followed by \setlength
    result = re.sub(
        r'(\\begin\{itemize\}(?:\[[^\]]*\])?)(?!\\setlength)',
        r'\1\\setlength{\\itemsep}{0pt}',
        frame_text
    )
    # Add to \begin{enumerate}[optional] not followed by \setlength
    result = re.sub(
        r'(\\begin\{enumerate\}(?:\[[^\]]*\])?)(?!\\setlength)',
        r'\1\\setlength{\\itemsep}{0pt}',
        result
    )
    return result


def add_font_wrapper(frame_text, font_cmd):
    """Add font wrapper to frame body."""
    lines = frame_text.split('\n')

    # Find insertion point: after frame title, cminipage, and vspace
    insert_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(r'\begin{frame}'):
            insert_idx = i + 1
            continue
        if insert_idx is not None:
            if stripped.startswith(r'\begin{cminipage}'):
                insert_idx = i + 1
                continue
            if stripped.startswith(r'\vspace'):
                insert_idx = i + 1
                continue
            if stripped.startswith('%') or not stripped:
                insert_idx = i + 1
                continue
            break

    if insert_idx is None:
        return frame_text

    indent = '    '

    if font_cmd.startswith('{'):
        # Brace-wrapped style: {\small ... }
        lines.insert(insert_idx, indent + font_cmd)

        # Find closing point: prefer before \end{cminipage}, else before \end{frame}
        close_idx = None
        for j in range(len(lines) - 1, insert_idx, -1):
            stripped = lines[j].strip()
            if stripped == r'\end{cminipage}':
                close_idx = j
                break
            if stripped == r'\end{frame}' and close_idx is None:
                close_idx = j
        if close_idx is not None:
            lines.insert(close_idx, indent + '}')
    else:
        # Command style: \small (no closing brace needed)
        lines.insert(insert_idx, indent + font_cmd)

    return '\n'.join(lines)


def add_leading_vspace(frame_text, vspace_cmd):
    """Add leading negative vspace after frame title / cminipage."""
    lines = frame_text.split('\n')

    insert_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(r'\begin{frame}'):
            insert_idx = i + 1
            continue
        if insert_idx is not None:
            if stripped.startswith(r'\begin{cminipage}'):
                insert_idx = i + 1
                continue
            break

    if insert_idx is None:
        return frame_text

    indent = '    '
    lines.insert(insert_idx, indent + vspace_cmd)
    return '\n'.join(lines)


# ─── Frame Transformation Pipeline ──────────────────────────────────────────

def transform_frame(en_frame, ro_frame):
    """Apply RO formatting to EN frame, preserving English text."""
    stats = {
        'hfill_converted': False,
        'cminipage_added': False,
        'itemsep_added': False,
        'font_added': False,
        'vspace_added': False,
    }

    # Skip title pages
    if is_title_or_toc_frame(en_frame):
        return en_frame, stats

    # Step 1: Convert hfill+minipage → cminipage
    if has_hfill_minipage(en_frame):
        en_frame, converted = convert_hfill_minipage(en_frame)
        stats['hfill_converted'] = converted

    # Step 2: Add cminipage wrapper if RO has it but EN doesn't
    if has_cminipage(ro_frame) and not has_cminipage(en_frame):
        en_frame = add_cminipage(en_frame)
        stats['cminipage_added'] = True

    # Step 3: Add itemsep to lists where RO has it but EN doesn't
    ro_has_itemsep = count_itemsep(ro_frame) > 0
    en_lists = count_lists(en_frame)
    en_itemseps = count_itemsep(en_frame)
    if ro_has_itemsep and en_lists > en_itemseps:
        en_frame = add_itemsep_to_lists(en_frame)
        stats['itemsep_added'] = True

    # Step 4: Add font wrapper if RO has one but EN doesn't
    ro_font = detect_font_wrapper(ro_frame)
    en_font = detect_font_wrapper(en_frame)
    if ro_font and not en_font:
        en_frame = add_font_wrapper(en_frame, ro_font)
        stats['font_added'] = True

    # Step 5: Add leading negative vspace if RO has one but EN doesn't
    ro_vspace = detect_leading_vspace(ro_frame)
    en_vspace = detect_leading_vspace(en_frame)
    if ro_vspace and not en_vspace:
        en_frame = add_leading_vspace(en_frame, ro_vspace)
        stats['vspace_added'] = True

    return en_frame, stats


# ─── Preamble ────────────────────────────────────────────────────────────────

CMINIPAGE_DEF = r"""
%=============================================================================
% CENTRED MINIPAGE
%=============================================================================
\newenvironment{cminipage}[1]{%
    \par\noindent\hfill\begin{minipage}{#1}\ignorespaces
}{%
    \end{minipage}\hfill\null\par
}
"""


def ensure_cminipage_definition(content):
    """Add cminipage definition to preamble if missing."""
    if r'\newenvironment{cminipage}' in content:
        return content, False
    # Insert before \begin{document}
    if r'\begin{document}' in content:
        content = content.replace(
            r'\begin{document}',
            CMINIPAGE_DEF.strip() + '\n\n' + r'\begin{document}'
        )
        return content, True
    return content, False


# ─── Main Sync Function ─────────────────────────────────────────────────────

def sync_chapter(en_path, ro_path, dry_run=False):
    """Sync formatting from RO chapter to EN chapter."""
    with open(en_path, 'r') as f:
        en_content = f.read()
    with open(ro_path, 'r') as f:
        ro_content = f.read()

    en_lines = en_content.split('\n')
    ro_lines = ro_content.split('\n')

    en_frame_ranges = find_frame_ranges(en_lines)
    ro_frame_ranges = find_frame_ranges(ro_lines)

    basename = os.path.basename(en_path)
    print(f"\n{'='*60}")
    print(f"Processing: {basename}")
    print(f"  EN frames: {len(en_frame_ranges)}, RO frames: {len(ro_frame_ranges)}")

    if len(en_frame_ranges) != len(ro_frame_ranges):
        print(f"  WARNING: Frame count mismatch!")

    n_frames = min(len(en_frame_ranges), len(ro_frame_ranges))

    # Aggregate stats
    totals = {
        'hfill_converted': 0,
        'cminipage_added': 0,
        'itemsep_added': 0,
        'font_added': 0,
        'vspace_added': 0,
    }

    # Build output: reconstruct file with modified frames
    output_parts = []
    prev_end = -1

    for i in range(len(en_frame_ranges)):
        en_start, en_end = en_frame_ranges[i]

        # Add inter-frame text (between previous frame end and this frame start)
        if prev_end + 1 <= en_start - 1:
            inter_text = '\n'.join(en_lines[prev_end + 1:en_start])
            output_parts.append(inter_text)
        elif prev_end >= 0:
            # No gap - just need newline separator
            pass

        # Extract frames
        en_frame = extract_frame_text(en_lines, en_start, en_end)

        if i < n_frames:
            ro_start, ro_end = ro_frame_ranges[i]
            ro_frame = extract_frame_text(ro_lines, ro_start, ro_end)
            new_frame, stats = transform_frame(en_frame, ro_frame)

            # Accumulate stats
            for k in totals:
                if stats[k]:
                    totals[k] += 1
        else:
            new_frame = en_frame

        output_parts.append(new_frame)
        prev_end = en_end

    # Add postamble (everything after last frame)
    if prev_end + 1 < len(en_lines):
        post_text = '\n'.join(en_lines[prev_end + 1:])
        output_parts.append(post_text)

    result = '\n'.join(output_parts)

    # Ensure cminipage definition in preamble
    result, preamble_added = ensure_cminipage_definition(result)

    # Print summary
    print(f"  Changes:")
    print(f"    cminipage definition added to preamble: {preamble_added}")
    print(f"    hfill→cminipage conversions: {totals['hfill_converted']}")
    print(f"    cminipage wrappers added: {totals['cminipage_added']}")
    print(f"    itemsep additions: {totals['itemsep_added']}")
    print(f"    font wrappers added: {totals['font_added']}")
    print(f"    vspace additions: {totals['vspace_added']}")

    if not dry_run:
        with open(en_path, 'w') as f:
            f.write(result)
        print(f"  Written: {en_path}")
    else:
        print(f"  DRY RUN - no changes written")

    return totals


# ─── CLI ─────────────────────────────────────────────────────────────────────

CHAPTERS = {
    'ch3': ('EN/Courses/chapter3_arima_models.tex',
            'RO/Courses/chapter3_arima_models_ro.tex'),
    'ch4': ('EN/Courses/chapter4_sarima_models.tex',
            'RO/Courses/chapter4_sarima_models_ro.tex'),
    'ch5': ('EN/Courses/chapter5_garch_volatility.tex',
            'RO/Courses/chapter5_garch_volatility_ro.tex'),
    'ch6': ('EN/Courses/chapter6_var_granger.tex',
            'RO/Courses/chapter6_var_granger_ro.tex'),
    'ch7': ('EN/Courses/chapter7_cointegration_vecm.tex',
            'RO/Courses/chapter7_cointegration_vecm_ro.tex'),
    'ch8': ('EN/Courses/chapter8_modern_extensions.tex',
            'RO/Courses/chapter8_modern_extensions_ro.tex'),
    'ch9': ('EN/Courses/chapter9_prophet_tbats.tex',
            'RO/Courses/chapter9_prophet_tbats_ro.tex'),
    'ch10': ('EN/Courses/chapter10_comprehensive_review.tex',
             'RO/Courses/chapter10_comprehensive_review_ro.tex'),
}


if __name__ == '__main__':
    args = sys.argv[1:]
    dry_run = '--dry-run' in args
    args = [a for a in args if a != '--dry-run']

    if not args:
        print("Usage: python sync_en_to_ro.py <ch3|ch4|...|ch10|all> [--dry-run]")
        sys.exit(1)

    if 'all' in args:
        chapters_to_process = list(CHAPTERS.keys())
    else:
        chapters_to_process = args

    for ch in chapters_to_process:
        if ch not in CHAPTERS:
            print(f"Unknown chapter: {ch}")
            continue
        en_path, ro_path = CHAPTERS[ch]
        sync_chapter(en_path, ro_path, dry_run=dry_run)

    print(f"\n{'='*60}")
    print("Done.")
