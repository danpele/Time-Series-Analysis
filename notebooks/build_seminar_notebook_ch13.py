"""
build_seminar_notebook_ch13.py -- seminar notebook of Chapter 13 (TSA): bubbles, explosive-root tests, LPPL
==========================================================================================================
Output: notebooks/EN/chapter13_seminar_notebook.ipynb
Chapter 13 is a self-study chapter; its lecture notebook is built by build_notebooks_ch13.py, the seminar notebook
here (a separate script, so that the lecture builder stays untouched). The code is taken from
Quantlets/Ch_13/generate_all_charts.py and seminar13.py (inspect.getsource).
Run:  python3 notebooks/build_seminar_notebook_ch13.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter13_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 13
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(13)
import seminar13 as s             # noqa: E402
import build_quantlets as bq      # noqa: E402

CONSTS = '\n'.join(bq.CONSTS).replace('HERE = "."', 'HERE = next((p for p in ("../../Quantlets/Ch_13", "Quantlets/Ch_13", ".") '
                                                     'if os.path.isdir(p)), ".")')
SEM_CONSTS = (f"SP_SAMPLE = {s.SP_SAMPLE!r}\nSP_PEAK_WEEK = {s.SP_PEAK_WEEK!r}\nLPPL_LAGS = {s.LPPL_LAGS!r}\n"
              f"LPPL_EX = {s.LPPL_EX!r}\nBW_EX = {s.BW_EX!r}")
SEM_FUNCS = [s.lecture_numbers, s.sp500_psy, s.b1_sp500, s.a1_adf_window, s.a3_datestamp, s.a5_rational]
SOL_FUNCS = [s.a2_reading, s.a4_lppl, s.b2_btc_lppl, s.c1_check]

SEMINAR = [
    md("# Time Series Analysis — Seminar 13: Speculative bubbles, explosive-root tests and LPPL models\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Chapter 13 is a self-study chapter: read it first; the slides recap the formulas used here.\n"
       "- [Solved] tasks: full solution here and in the slides; [Proposed] tasks: write your own code in the empty cell.\n"
       "- The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class.\n"
       "- The first code cell installs the `arch` package if it is missing."),
    *common_cells(bq.INSTALL),
    md("## Seminar functions\n\n- Helpers of Chapter 13 (`adf_window`, `psy`, `psy_cv`, `episodes_above`, `lppl_fit`, "
       "`lppl_conditions`, `fit_window`) and the seminar functions: `sp500_psy`, `b1_sp500`, `a1_adf_window`, "
       "`a3_datestamp`, `a5_rational`."),
    code(CONSTS + '\n' + SEM_CONSTS + '\n\n\n' + src(*bq.CORE) + '\n\n\n' + src(*SEM_FUNCS)),
    code("# [solutions only]\n" + src(*SOL_FUNCS)),
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code. "
       "A1 and A3 use the S&P 500 sample of B1: the next cell computes it once (about ten seconds)."),
    code("res = sp500_psy()      # weekly S&P 500 1990-2004: PSY statistics, Monte Carlo critical values, dated episodes"),
    md("## A1 [Solved]: the right-tailed ADF on one window\n\n"
       "**Context.** Weekly S&P 500 log prices on the window that gives the BSADF value at the week of the peak (24 March 2000).\n\n"
       "1. Compute the ADF statistic and the root $\\hat\\rho = 1 + \\hat\\delta$.\n"
       "2. Compare with the 95% critical value of BSADF for this week.\n"
       "3. Is the run of weeks above the critical value around the peak a dated episode ($L = [\\ln T]$)?\n\n"
       "**Report:** one statistic, one root, two decisions."),
    code("# Solution\na1 = a1_adf_window(res)\nprint({k: (round(v, 5) if isinstance(v, float) else v) for k, v in a1.items()})"),
    md("**Interpretation of the result.** The statistic is above the right-tail critical value: the week is explosive; "
       "the run around the peak is shorter than $L = 7$ weeks, so it is not a dated episode."),
    md("## A2 [Proposed]: reading SADF and GSADF\n\n"
       "**Context.** The SADF and GSADF statistics of Chapter 13 (Nasdaq 100, Bitcoin, BET, Shanghai) with their 95% "
       "critical values (`lecture_numbers()['psy_ndx']`, `lecture_numbers()['psy']`). Model: A1.\n\n"
       "1. Compute $r_0 = 0.01 + 1.8/\\sqrt{T}$ and $w_0 = \\lfloor r_0T\\rfloor$ for the Nasdaq 100 and for Bitcoin.\n"
       "2. For each series, decide on SADF and on GSADF at 5%.\n3. Why is GSADF never below SADF? Why are they equal for Bitcoin?\n"
       "4. Why does the whole-sample ADF of the Nasdaq 100 not reject?\n\n**Report:** two windows, eight decisions, two sentences."),
    code("# Solution\npd.DataFrame(a2_reading()).T"),
    md("## A3 [Solved]: dating a bubble from BSADF\n\n"
       "**Context.** BSADF and its 95% critical value in the weeks around the first crossing that lasts.\n\n"
       "1. Compute the minimum duration $L = [\\ln T]$.\n2. Mark the weeks with BSADF above the critical value.\n"
       "3. Give the start of the first episode. Why is the first crossing not the start?\n\n**Report:** $L$, a start date, one sentence."),
    code("# Solution\na3 = a3_datestamp(res)\nprint(pd.DataFrame(a3['rows']).round(3))\n"
         "print({k: a3[k] for k in ('T', 'lnT', 'L', 'start', 'end', 'weeks', 'n_cross', 'n_short')})"),
    md("**Interpretation of the result.** $L = 7$; the one-week crossing of 2 June 1995 is dropped; the episode starts on "
       "16 June 1995 and lasts 57 weeks."),
    md("## A4 [Proposed]: the LPPL formula and its filter\n\n"
       "**Context.** LPPL with $A = 8$, $B = -1$, $C = 0.1$, $m = 0.4$, $\\omega = 7$, $\\phi = 0$, $t_c = 1$. Model: A3.\n\n"
       "1. Compute $\\ln p(t)$ at $t = 0, 0.5, 0.9$, term by term.\n2. Compute $\\lambda = e^{2\\pi/\\omega}$ and the first "
       "three peaks of the cosine after $t = 0$.\n3. For $[t_1, t_2] = [0, 0.9]$, compute the oscillations and the damping. "
       "Which filter condition fails?\n\n**Report:** three log prices, $\\lambda$, three dates, two numbers, one verdict."),
    code("# Solution\na4_lppl()"),
    md("## A5 [Solved]: the recursion of a rational bubble\n\n"
       "**Context.** Blanchard–Watson bubble with $r = 0.01$, $\\pi = 0.95$, $b_0 = 1$, no noise; dividend $D = 2$.\n\n"
       "1. Compute the growth factor while the bubble survives and $b_1, b_2, b_3$.\n2. Check $E_0(b_1) = (1 + r)b_0$.\n"
       "3. Compute the probability of surviving three periods, the mean and the median duration.\n"
       "4. Compute $p^*$ and the share of the bubble in the price after three periods.\n\n"
       "**Report:** one factor, three values, three probabilities or durations, one share."),
    code("# Solution\na5_rational()"),
    md("**Interpretation of the result.** The bubble grows by 6.3% per period while it survives and by 1% on average; "
       "after three periods it is still 0.6% of the price."),
    md("# Part B: real data and interpretation"),
    md("## B1 [Solved]: the S&P 500 in the dot-com years\n\n"
       "**Question.** Was the S&P 500 explosive in the late 1990s, and when, compared with the Nasdaq 100 of Chapter 13?\n\n"
       "1. Compute ADF, SADF, GSADF, the BSADF sequence and their critical values (1000 replications).\n"
       "2. Date the episodes with the 95% critical value and $L = [\\ln T]$.\n"
       "3. Compare with the Nasdaq 100.\n4. Interpretation: how strong is the evidence of a bubble in the broad market?\n\n"
       "**Report:** three statistics with their critical values, a list of episodes, one sentence."),
    code("# Solution\nb1 = b1_sp500(save_it=False, res=res)\n"
         "print({k: b1[k] for k in ('T', 'w0', 'adf', 'sadf', 'gsadf', 'cv', 'min_len', 'n_ep', 'weeks_above')})\n"
         "print(pd.DataFrame(b1['episodes'], columns=['start', 'end', 'weeks']))\nprint('Nasdaq 100:', b1['ndx'])"),
    md("**Interpretation of the result.** SADF rejects at 5%, GSADF only at 10%: weaker evidence than for the Nasdaq 100; "
       "the explosive phase of the S&P 500 ends before the peak of March 2000."),
    md("## B2 [Proposed]: LPPL fits of Bitcoin in 2021\n\n"
       "**Question.** Would an LPPL fit have announced the Bitcoin peak of April 2021, and how does $t_c$ move as the window "
       "grows? Model: A4.\n\n"
       "1. Fit the LPPL model (`fit_window`) from the low before the run-up to 90, 60 and 30 days before the peak.\n"
       "2. Report $t_c$, $m$, $\\omega$ and the distance between $t_c$ and the actual peak.\n"
       "3. Apply the filter (`lppl_conditions`) and say which conditions fail.\n4. Interpretation: is $t_c$ a forecast of the crash date?\n\n"
       "**Report:** three critical times, six parameters, three lists of failed conditions, one sentence."),
    code("# Solution\nb2 = b2_btc_lppl(save_it=False)\n"
         "print(pd.DataFrame(b2['fits']).T[['t2', 'tc', 'tc_err', 'lead', 'm', 'w', 'damping', 'osc', 'rel_err', 'failed']])"),
    md("# Part C: AI critique"),
    md("## C1 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant answered:\n\n"
       "- (a) to test for a bubble, reject the unit root when the ADF statistic is below -2.86;\n"
       "- (b) the SADF of the Nasdaq 100 is above its critical value, so the Nasdaq was in a bubble from 1990 to 2004;\n"
       "- (c) GSADF can be smaller than SADF when the bubble is at the end of the sample;\n"
       "- (d) the LPPL critical time tc is the date of the crash: sell the day before tc;\n"
       "- (e) a fit with m = 0.999 and omega = 1 is a valid bubble signal if its errors are small;\n"
       "- (f) in a rational bubble the expected bubble grows at the rate r, although it grows faster than r while it survives.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc1_check()"),
]

if __name__ == '__main__':
    build(SEMINAR, 13, 'seminar')
