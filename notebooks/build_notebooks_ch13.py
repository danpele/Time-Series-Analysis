"""
build_notebooks_ch13.py -- lecture notebook of Chapter 13 (TSA): speculative bubbles and LPPL models
===================================================================================================
Output: notebooks/EN/chapter13_lecture_notebook.ipynb (self-study chapter: no seminar notebook)
The code is taken from Quantlets/Ch_13/generate_all_charts.py (inspect.getsource), so the notebook stays in sync
with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch13.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter13_lecture_notebook.ipynb
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(13)
import generate_all_charts as g   # noqa: E402
import build_quantlets as bq      # noqa: E402

CONSTS = '\n'.join(bq.CONSTS).replace('HERE = "."', 'HERE = next((p for p in ("../../Quantlets/Ch_13", "Quantlets/Ch_13", ".") '
                                                     'if os.path.isdir(p)), ".")')
CORE = bq.CORE
INSTALL = bq.INSTALL

SIMULATION_CHECK = '''# A check of the estimator on simulated data: an LPPL path with known parameters plus noise
rng = np.random.default_rng(SEED)
true = dict(A=8.0, B=-0.6, C1=0.03, C2=0.02, tc=2.0, m=0.45, w=8.0)
t = np.linspace(0.0, 1.9, 500)                            # the window ends 0.1 years before tc
y = lppl_design(t, true['tc'], true['m'], true['w']) @ np.array([true['A'], true['B'], true['C1'], true['C2']])
y_obs = y + 0.01 * rng.standard_normal(len(t))
fit = lppl_fit(t, y_obs)
print({k: round(fit[k], 3) for k in ('tc', 'm', 'w', 'A', 'B', 'C1', 'C2')})
print('conditions:', lppl_conditions(fit))'''

LECTURE = [
    md("# Time Series Analysis — Chapter 13: Speculative bubbles and LPPL models\n\n"
       "*Lecture notebook (self-study chapter). Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Six run-ups and crashes in the course data: S&P 500 and Nasdaq 100 (2000), BET (2007), Shanghai Composite (2015), "
       "Bitcoin (2017, 2021).\n"
       "- Rational bubbles and explosive roots: right-tailed ADF, SADF, GSADF and BSADF date-stamping.\n"
       "- The LPPL model of Johansen, Ledoit and Sornette; estimation by the two-step method of Filimonov and Sornette.\n"
       "- Confidence from many windows (the LPPLS confidence indicator) and an honest evaluation of crash prediction.\n"
       "- The first code cell installs the `arch` package if it is missing (`pip install arch`).\n"
       "- References: Blanchard and Watson (1982); Phillips, Wu and Yu (2011); Phillips, Shi and Yu (2015); Johansen, "
       "Ledoit and Sornette (2000); Filimonov and Sornette (2013); Shu and Zhu (2020)."),
    *common_cells(INSTALL),
    md("## Definitions and helpers used in the whole notebook\n\n"
       "- `psy(y)`: ADF, SADF, GSADF and the BSADF sequence of a log price; `psy_cv(T, w0)`: Monte Carlo critical values.\n"
       "- `lppl_fit(t, y)`: LPPL calibration (OLS for A, B, C1, C2; grid search and Nelder–Mead for tc, m, omega); "
       "`t` in decimal years, `y` the log price.\n"
       "- `lppl_conditions(fit)`: the filter conditions of Shu and Zhu (2020); `qualified(c, 'param' | 'full')`.\n"
       "- `confidence_series(s, start, end)`: the LPPLS confidence indicator; `cached_ci` reads the series computed for "
       "the slides (`Quantlets/Ch_13/ch13_ci_*.csv`), because the full computation takes several minutes."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Six run-ups and crashes\n\n- The low before the run-up, the peak, the rise and the fall in the next year."),
    code(src(g.fig_episodes) + "\n\n\nep = fig_episodes(save_it=False)\n"
         "print(pd.DataFrame(ep).T[['low', 'peak', 'runup', 'growth', 'fall1y']].round(3))"),
    md("## 2. Rational bubbles and explosive roots\n\n- A Blanchard–Watson bubble; stationary, unit-root and explosive AR(1)."),
    code(src(g.fig_rational) + "\n\n\nprint(worked_examples()['bw'])\nprint(fig_rational(save_it=False))"),
    md("## 3. Right-tailed unit-root tests: SADF, GSADF, BSADF\n\n"
       "- Weekly log prices; critical values by Monte Carlo (1000 random walks with a weak drift).\n"
       "- The worked example: the BSADF window of the Nasdaq 100 at the week of the peak."),
    code(src(g.fig_psy_ndx, g.fig_psy_panel) + "\n\n\npn = fig_psy_ndx(save_it=False)\n"
         "print({k: pn[k] for k in ('adf', 'sadf', 'gsadf', 'cv')})\nprint(pn['episodes'])\nprint(pn['example'])\n"
         "pp = fig_psy_panel(save_it=False)\nprint({k: (round(v['gsadf'], 2), round(v['cv95'], 2), v['episodes']) for k, v in pp.items()})"),
    md("## 4. The LPPL model\n\n- Exponential and super-exponential growth; the pieces of the LPPL equation."),
    code(src(g.fig_growth, g.fig_lppl_components) + "\n\n\nprint(fig_growth(save_it=False))\nprint(fig_lppl_components(save_it=False))\n"
         "print(worked_examples()['1.75'])"),
    md("## 5. Estimation\n\n- First a check on simulated data: does the estimator recover known parameters?\n"
       "- Then the cost landscape of the Nasdaq 100 and the fits 30 days before six peaks."),
    code(SIMULATION_CHECK),
    code(src(g.fig_cost, g.fig_fits) + "\n\n\nprint(fig_cost(save_it=False))\nfits = fig_fits(save_it=False)\n"
         "print(pd.DataFrame({k: v for k, v in fits.items() if k != 'example'}).T[['t2', 'tc', 'tc_err', 'm', 'w', 'q_param', 'q_full']])"),
    md("## 6. Confidence from many windows\n\n- 29 windows ending at one date; the critical time as the window end moves; the "
       "indicator around four peaks (read from the cached series)."),
    code(src(g.fig_windows, g.fig_tc_path, g.fig_ci) + "\n\n\nprint(fig_windows(save_it=False))\nprint(fig_tc_path(save_it=False))\n"
         "print(fig_ci(save_it=False))"),
    md("### Computing the indicator yourself\n\n- A short stretch of Bitcoin in 2017, every 21 days, sequentially (about a "
       "minute). Change the window list or the filter and compare."),
    code("s17 = prices('BTC-USD.CC', crypto=True)\n"
         "ci_short = confidence_series(s17, '2017-09-01', '2018-01-15', step=21, procs=1)\nprint(ci_short.round(3))"),
    md("## 7. An honest evaluation\n\n- Alarms (indicator >= 0.2) against falls of at least 20% within 182 days, over the "
       "whole S&P 500 and Bitcoin samples; hit rate against the base rate."),
    code(src(g.fig_eval) + "\n\n\nev = fig_eval(save_it=False)\n"
         "for k, v in ev.items():\n    print(k, 'base rate', round(v['hit']['base'], 3), "
         "{c: round(v['hit'][c]['hit'], 3) for c in ('0.1', '0.2', '0.3')})\n"
         "    print('   drawdowns of 20% or more:', [(f['peak'], round(f['fall'], 2), f['alarm_before']) for f in v['falls']])"),
    md("## Exercises\n\n"
       "1. Run `run_psy` on the weekly BET-TR (`prices('BETTR.INDX')`, since 2014): which episodes do you find?\n"
       "2. Fit LPPL to Bitcoin between 1 January 2017 and 1 December 2017 and check the filter conditions one by one.\n"
       "3. Replace the 15% error condition by 25% (`LPPLS_FILTER['rel_err_max']`) and recompute the indicator for Bitcoin "
       "in 2017 with `confidence_series`: how does the picture change?\n"
       "4. Change the crash definition to a fall of 15% within 90 days and recompute the hit rates with `hit_rates`."),
]

if __name__ == '__main__':
    build(LECTURE, 13, 'lecture')
