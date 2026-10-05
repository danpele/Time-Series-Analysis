"""
exam_common.py -- helpers shared by the chart and output generators of the TSA exam materials
(exam/make_figs.py, exam/practice/figs_practice.py and the instructor-only exam/bank/figs_chNN.py modules).

  * course data and style:  td (Quantlets/common/tsa_data.py), st (Quantlets/common/tsa_style.py)
  * save(fig, out_dir, name)            transparent PDF, legend outside at the bottom, no grey (st.check_no_grey)
  * ro_fmt(dec) / en_fmt(dec)           tick formatters (RO: decimal comma, true minus sign)
  * write_out(text, out_dir, name)      software output as a text file <out_dir>/<name>.txt (at most 84 characters
                                        per line, so that \\softout prints it in \\scriptsize without overflow)
  * sm_tables(res, which)               selected tables of a statsmodels summary, as text
  * correlogram(x, nlags)               correlogram table (lag, AC, PAC, Q-Stat, Prob), as in econometric software
  * coef_table(rows)                    a compact coefficient table (name, coef, std err, z, P>|z|)

Time Series Analysis - Daniel Traian PELE
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Quantlets', 'common'))
import tsa_data as td      # noqa: E402,F401
import tsa_style as st     # noqa: E402

MAXW = 84


def style():
    """Course style with font sizes for A4 exam sheets (charts 6-7.5 inches wide, shown at 0.6-0.75 textwidth)."""
    st.apply()
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
                         'legend.fontsize': 9, 'axes.titlesize': 10.5})


def ro_fmt(dec):
    """Tick formatter with a decimal comma, a space as thousands separator and a true minus sign."""
    return matplotlib.ticker.FuncFormatter(
        lambda v, _: f'{v:,.{dec}f}'.replace(',', ' ').replace('.', ',').replace('-', '−'))


def en_fmt(dec):
    return matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.{dec}f}'.replace('-', '−'))


def fmt(lang):
    return ro_fmt if lang == 'ro' else en_fmt


def save(fig, out_dir, name):
    st.check_no_grey(fig)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name + '.pdf'), bbox_inches='tight', transparent=True)
    plt.close(fig)
    print('   saved', os.path.relpath(os.path.join(out_dir, name + '.pdf'), HERE))


def write_out(text, out_dir, name):
    """Save a software output as <out_dir>/<name>.txt; raise if a line is longer than MAXW characters."""
    lines = [ln.rstrip() for ln in text.strip('\n').splitlines()]
    long = [ln for ln in lines if len(ln) > MAXW]
    if long:
        raise ValueError(f'{name}: {len(long)} line(s) longer than {MAXW} characters, e.g. {long[0]!r}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name + '.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('   output', os.path.relpath(os.path.join(out_dir, name + '.txt'), HERE))


def sm_tables(res, which=(0, 1, 2)):
    """Selected tables of a statsmodels results summary (0 = header, 1 = coefficients, 2 = diagnostics)."""
    s = res.summary()
    return '\n'.join(s.tables[i].as_text() for i in which if i < len(s.tables))


def correlogram(x, nlags=12, title=''):
    """Correlogram as printed by econometric software: lag, AC, PAC, Ljung--Box Q and its p-value."""
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    a = acf(x, nlags=nlags, fft=False)[1:]
    p = pacf(x, nlags=nlags, method='ywm')[1:]
    lb = acorr_ljungbox(x, lags=list(range(1, nlags + 1)))
    out = []
    if title:
        out.append(title)
    out.append(f'Sample size: {len(x)}')
    out.append(f'{"Lag":>4} {"AC":>8} {"PAC":>8} {"Q-Stat":>10} {"Prob":>8}')
    out.append('-' * 42)
    for k in range(nlags):
        out.append(f'{k + 1:>4} {a[k]:>8.3f} {p[k]:>8.3f} {lb["lb_stat"].iloc[k]:>10.2f} {lb["lb_pvalue"].iloc[k]:>8.3f}')
    return '\n'.join(out)


def coef_table(rows, header=('coef', 'std err', 'z', 'P>|z|'), title=''):
    """Compact coefficient table; rows = [(name, coef, se)] (z and p are computed) or (name, coef, se, z, p)."""
    from scipy.stats import norm
    out = [title] if title else []
    w = max(10, max(len(r[0]) for r in rows) + 1)
    out.append(f'{"":<{w}}' + ''.join(f'{h:>11}' for h in header))
    out.append('-' * (w + 11 * len(header)))
    for r in rows:
        name, c, se = r[:3]
        z = r[3] if len(r) > 3 else c / se
        p = r[4] if len(r) > 4 else 2 * norm.sf(abs(z))
        out.append(f'{name:<{w}}{c:>11.4f}{se:>11.4f}{z:>11.3f}{p:>11.3f}')
    return '\n'.join(out)


def num_ro(v, dec=2):
    """A number with a decimal comma (for checking the RO statement against the computed values)."""
    return f'{v:.{dec}f}'.replace('.', ',').replace('-', '−')
