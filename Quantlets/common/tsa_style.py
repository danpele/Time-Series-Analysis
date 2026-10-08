"""
tsa_style.py -- chart style of the TSA course (the same as MFM)
===============================================================
  * transparent background (figure, axes, saved files), no grid, no top/right spines;
  * the legend always OUTSIDE the plot, at the bottom centre (legend_outside_bottom), without a frame;
  * colours from the course palette (the LaTeX colours of latex/preamble.tex); no grey series and no grey text:
    text and axes are dark (DarkText), reference lines use a palette colour (dashed);
  * charts saved as PDF (for the slides) and PNG (for the notebooks and the site) in charts/.

Use:
    import tsa_style as st
    st.apply()                                   # once, before the first chart
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(x, y, color=st.COL['sp500'], label='S&P 500')
    st.legend_outside_bottom(ax, ncol=3)
    st.save_fig('tsa_ch1_returns')               # charts/tsa_ch1_returns.pdf + .png
    st.check_no_grey(fig)                        # optional: raises if a series or a text is grey

Time Series Analysis - Daniel Traian PELE
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# Palette (RGB values of latex/preamble.tex)
MainBlue = '#1A3A6E'
IDAred = '#CD0000'
Forest = '#2E7D32'
Amber = '#B5853F'
Orange = '#E67E22'
Purple = '#8E44AD'
Teal = '#17A2B8'
Crimson = '#DC3545'
DarkText = '#1F2A44'          # text, axes and ticks (dark navy, not grey)

PALETTE = [MainBlue, IDAred, Forest, Amber, Purple, Orange, Teal, Crimson]
# fixed colours for the series used in several chapters
COL = {'sp500': MainBlue, 'bet': IDAred, 'bettr': Orange, 'dax': Teal, 'btc': Amber, 'eth': Crimson,
       'eurron': Forest, 'gold': Purple, 'vix': Crimson, 'stoxx50': Forest, 'ndx': Teal}

_HERE = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(_HERE, '..', '..', 'charts')


def apply():
    """Set the course style for matplotlib."""
    rc = plt.rcParams
    rc['figure.facecolor'] = 'none'
    rc['axes.facecolor'] = 'none'
    rc['savefig.facecolor'] = 'none'
    rc['savefig.transparent'] = True
    rc['axes.grid'] = False
    rc['font.family'] = 'sans-serif'
    rc['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    # default size about 7 inches wide (as in MFM); a chart is shown 11-13 cm wide on a slide, so these font sizes
    # give 7-9 pt text there. Multi-panel charts should stay near 7-7.6 inches wide: at 10-11 inches the axis
    # text drops to about 5 pt on the slide.
    rc['figure.figsize'] = (7.0, 3.4)
    rc['font.size'] = 12
    rc['axes.labelsize'] = 13
    rc['axes.titlesize'] = 13
    rc['xtick.labelsize'] = 11.5
    rc['ytick.labelsize'] = 11.5
    rc['legend.fontsize'] = 11
    rc['axes.spines.top'] = False
    rc['axes.spines.right'] = False
    rc['axes.linewidth'] = 0.6
    rc['lines.linewidth'] = 1.4
    rc['legend.facecolor'] = 'none'
    rc['legend.framealpha'] = 0
    rc['legend.frameon'] = False
    for k in ('text.color', 'axes.labelcolor', 'axes.edgecolor', 'xtick.color', 'ytick.color', 'axes.titlecolor'):
        rc[k] = DarkText
    rc['axes.prop_cycle'] = mpl.cycler(color=PALETTE)
    _remember_tight_layout()


def _remember_tight_layout():
    """Figure.tight_layout also stores its arguments on the figure, so that save_fig can re-apply the same layout
    after it has wrapped a label or changed the number of legend columns (see finalize)."""
    from matplotlib.figure import Figure
    if getattr(Figure.tight_layout, '_tsa', False):
        return
    orig = Figure.tight_layout

    def tight_layout(self, *a, **k):
        self._tsa_tight = (a, k)
        return orig(self, *a, **k)
    tight_layout._tsa = True
    Figure.tight_layout = tight_layout


def legend_outside_bottom(ax, ncol=2, y=-0.22, **kw):
    """Place the legend outside the plot, bottom centre (for a figure with several axes, pass the last one
    or use fig.legend with the same arguments)."""
    return ax.legend(loc='upper center', bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, **kw)


def fig_legend_bottom(fig, handles=None, labels=None, ncol=3, y=-0.02):
    """One legend for a whole figure (several panels), below the panels."""
    if handles is None:
        handles, labels = [], []
        for ax in fig.axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels and not l.startswith('_'):
                    handles.append(h)
                    labels.append(l)
    return fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False)


def place_bottom_legends(fig, pad=3.0):
    """Legends outside at the bottom ('upper center' anchored below the plot): put the top of each legend `pad` points
    below the lowest tick label / axis label, so that a legend never covers the x axis text, whatever the figure size
    (an anchor such as y=-0.22 is a fraction of the axes height and is too close on a small chart)."""
    r = fig.canvas.get_renderer()
    upper_center = 9
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None or not leg.get_visible() or leg._loc not in (upper_center, 'upper center'):
            continue
        anchor = leg.get_bbox_to_anchor()
        if anchor.y0 >= ax.get_window_extent(r).y0:
            continue                                   # legend inside the plot area
        leg.set_visible(False)
        bb = ax.get_tightbbox(r)
        leg.set_visible(True)
        x = ax.transAxes.inverted().transform((anchor.x0 + anchor.width / 2, 0))[0]
        y = ax.transAxes.inverted().transform((0, bb.y0 - pad * fig.dpi / 72))[1]
        leg.set_bbox_to_anchor((x, y), transform=ax.transAxes)
    if fig.legends and fig.axes:
        low = min(ax.get_tightbbox(r).y0 for ax in fig.axes if ax.get_visible())
        for leg in fig.legends:
            if leg._loc not in (upper_center, 'upper center'):
                continue
            anchor = leg.get_bbox_to_anchor()
            if anchor.y0 > fig.bbox.height / 2:
                continue                               # legend at the top of the figure
            x = fig.transFigure.inverted().transform((anchor.x0 + anchor.width / 2, 0))[0]
            y = fig.transFigure.inverted().transform((0, low - pad * fig.dpi / 72))[1]
            leg.set_bbox_to_anchor((x, y), transform=fig.transFigure)


def _wrap(text):
    """Break a label in two lines at the space nearest its middle (never inside $...$)."""
    if '\n' in text:
        return text
    spaces, inside = [], False
    for i, ch in enumerate(text):
        if ch == '$':
            inside = not inside
        elif ch == ' ' and not inside:
            spaces.append(i)
    if not spaces:
        return text
    i = min(spaces, key=lambda j: abs(j - len(text) / 2))
    return text[:i] + '\n' + text[i + 1:]


def _set_ncols(leg, n):
    """Re-create a legend with n columns (same entries, anchor and text properties); returns the new legend."""
    from matplotlib.axes import Axes
    parent = leg.parent
    handles = list(leg.legend_handles)
    labels = [t.get_text() for t in leg.texts]
    colors = [t.get_color() for t in leg.texts]
    tr = parent.transAxes if isinstance(parent, Axes) else parent.transFigure
    a = leg.get_bbox_to_anchor()
    (x0, y0), (x1, y1) = tr.inverted().transform([(a.x0, a.y0), (a.x1, a.y1)])
    anchor = (x0, y0) if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9 else (x0, y0, x1 - x0, y1 - y0)
    title = leg.get_title().get_text()
    kw = dict(loc=leg._loc, ncols=n, frameon=leg.get_frame_on(), prop=leg.prop, handlelength=leg.handlelength,
              handletextpad=leg.handletextpad, columnspacing=leg.columnspacing, labelspacing=leg.labelspacing,
              borderaxespad=leg.borderaxespad, markerscale=leg.markerscale, numpoints=leg.numpoints,
              scatterpoints=leg.scatterpoints, bbox_to_anchor=anchor, bbox_transform=tr)
    if title:
        kw['title'] = title
    leg.remove()
    new = parent.legend(handles, labels, **kw)
    for t, c in zip(new.texts, colors):
        t.set_color(c)
    return new


def _thin_ticks(axis, setter, r, gap=2.0):
    """If neighbouring (horizontal) tick labels of an axis overlap, keep every second tick. True if thinned."""
    ticks = [t for t in axis._update_ticks() if t.label1.get_visible() and t.label1.get_text()
             and t in axis.get_major_ticks()]
    if len(ticks) < 3:
        return False
    rot = ticks[0].label1.get_rotation() % 180
    vertical = axis.axis_name == 'y'
    from matplotlib.ticker import FixedFormatter, FuncFormatter, FixedLocator
    labels_set = isinstance(axis.get_major_formatter(), (FixedFormatter, FuncFormatter)) and \
        isinstance(axis.get_major_locator(), FixedLocator)
    if labels_set and any(c.isalpha() for t in ticks for c in t.label1.get_text()):
        # category names: never drop one; wrap them in two lines, then slant them
        if vertical or rot not in (0, 90):
            return False
        boxes = sorted((t.label1.get_window_extent(r) for t in ticks), key=lambda b: b.x0)
        if not any(b.x0 < a.x1 + gap for a, b in zip(boxes, boxes[1:])):
            return False
        texts = [t.get_text() for t in axis.get_ticklabels()]
        wrapped = [_wrap(x) for x in texts]
        if wrapped != texts:
            axis.set_ticklabels(wrapped)
        else:                                # wrapping was not enough: one line, slanted
            axis.set_ticklabels([x.replace('\n', ' ') for x in texts])
            for t in axis.get_ticklabels():
                t.set_rotation(45); t.set_ha('right'); t.set_rotation_mode('anchor')
        return True
    if rot not in (0, 90):
        # rotated labels: parallel slanted strings do not overlap if their perpendicular distance exceeds the height
        import math
        h = ticks[0].label1.get_fontsize() * r.points_to_pixels(1.0) * 1.15
        pos = sorted(t.label1.get_window_extent(r).x1 if not vertical else t.label1.get_window_extent(r).y1 for t in ticks)
        if min(b - a for a, b in zip(pos, pos[1:])) * math.sin(math.radians(rot)) < h:
            locs = [t.get_loc() for t in ticks]
            labels = [t.label1.get_text() for t in ticks]
            setter(locs[::2])
            axis.set_ticklabels(labels[::2])
            return True
        return False
    boxes = sorted((t.label1.get_window_extent(r) for t in ticks), key=lambda b: b.y0 if vertical else b.x0)
    for a, b in zip(boxes, boxes[1:]):
        if (b.y0 < a.y1 + gap) if vertical else (b.x0 < a.x1 + gap):
            locs = [t.get_loc() for t in ticks]
            labels = [t.label1.get_text() for t in ticks]
            setter(locs[::2])
            axis.set_ticklabels(labels[::2])
            return True
    return False


def _tidy_log_axis(axis):
    """Log axis: no minor tick labels (they pile up on a short axis); an axis with fewer than two powers of ten in view
    gets plain-number ticks at 1, 2 and 5 times the powers of ten (instead of a single 10^k)."""
    import math
    from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
    if axis.get_scale() != 'log':
        return
    lo, hi = sorted(axis.get_view_interval())
    if lo <= 0:
        return
    axis.set_minor_formatter(NullFormatter())
    powers = math.floor(math.log10(hi)) - math.ceil(math.log10(lo)) + 1      # powers of ten inside the view
    if powers < 2:
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'.replace(',', ' ') if v >= 1000 else f'{v:g}'))


def finalize(fig):
    """Make a (small) slide chart readable without overlaps: wrap a y label taller than its panel and a panel
    title wider than its panel, use fewer legend columns when a bottom legend is wider than the plots, re-apply the
    figure's tight_layout, then place the bottom legends under the x axis text (place_bottom_legends)."""
    r = fig.canvas.get_renderer()
    axs = [a for a in fig.axes if a.get_visible()]
    for ax in axs:
        _tidy_log_axis(ax.xaxis)
        _tidy_log_axis(ax.yaxis)
    for _ in range(3):
        changed = False
        for ax in axs:
            e = ax.get_window_extent(r)
            yl = ax.yaxis.label
            if yl.get_text() and yl.get_window_extent(r).height > e.height * 1.02:
                t = _wrap(yl.get_text())
                if t != yl.get_text():
                    yl.set_text(t); changed = True
            if len(axs) > 1:
                for ti in (ax.title, ax._left_title, ax._right_title):
                    if ti.get_text() and ti.get_window_extent(r).width > e.width * 1.04:
                        t = _wrap(ti.get_text())
                        if t != ti.get_text():
                            ti.set_text(t); changed = True
        legs = list(fig.legends) + [a.get_legend() for a in axs if a.get_legend() is not None]
        span = None
        if axs:
            vis = [lg.get_visible() for lg in legs]
            for lg in legs:
                lg.set_visible(False)
            x0 = min(a.get_tightbbox(r).x0 for a in axs)
            x1 = max(a.get_tightbbox(r).x1 for a in axs)
            for lg, v in zip(legs, vis):
                lg.set_visible(v)
            span = max(x1 - x0, fig.bbox.width)      # a square plot may have a legend wider than itself
        for leg in legs:
            if leg._loc not in (9, 'upper center') or leg._ncols <= 1 or span is None:
                continue
            limit = span
            if leg not in fig.legends and len(axs) > 1:      # the legend of one panel among several: as wide as its panel
                leg.set_visible(False)
                limit = leg.axes.get_tightbbox(r).width * 1.1
                leg.set_visible(True)
            while leg._ncols > 1 and leg.get_window_extent(r).width > limit * 1.02:
                leg = _set_ncols(leg, leg._ncols - 1); changed = True
        for ax in axs:
            for axis, setter in ((ax.xaxis, ax.set_xticks), (ax.yaxis, ax.set_yticks)):
                if _thin_ticks(axis, setter, r):
                    changed = True
        if not changed:
            break
        if getattr(fig, '_tsa_tight', None):
            a, k = fig._tsa_tight
            fig.tight_layout(*a, **k)
    place_bottom_legends(fig)


def save_fig(name, out_dir=None, show=False):
    """Save the current figure as transparent PDF and PNG (charts/ by default), after finalize()."""
    d = out_dir or CHART_DIR
    os.makedirs(d, exist_ok=True)
    finalize(plt.gcf())
    plt.savefig(os.path.join(d, f'{name}.pdf'), bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(d, f'{name}.png'), bbox_inches='tight', transparent=True, dpi=180)
    if show:
        plt.show()
    plt.close()
    print(f'   saved {name}')


def _is_grey(c, tol=0.06):
    try:
        r, g, b = to_rgb(c)
    except ValueError:
        return False
    return max(r, g, b) - min(r, g, b) < tol and 0.25 < (r + g + b) / 3 < 0.95


def check_no_grey(fig):
    """House rule: no grey series and no grey text. Raises ValueError listing the offending elements."""
    bad = []
    for ax in fig.axes:
        for ln in ax.get_lines():
            if not ln.get_label().startswith('_') and _is_grey(ln.get_color()):
                bad.append(f'line {ln.get_label()!r}')
        for coll in ax.collections:
            fc = coll.get_facecolor()
            if len(fc) and coll.get_label() and not coll.get_label().startswith('_') and _is_grey(fc[0][:3]):
                bad.append(f'series {coll.get_label()!r}')
        for t in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            if t.get_text() and _is_grey(t.get_color()):
                bad.append(f'text {t.get_text()[:30]!r}')
    if bad:
        raise ValueError('grey elements: ' + ', '.join(bad))
