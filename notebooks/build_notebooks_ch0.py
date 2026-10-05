"""
build_notebooks_ch0.py -- the lecture and seminar notebooks of Chapter 0 (TSA): Introduction
============================================================================================
Output: notebooks/EN/chapter0_lecture_notebook.ipynb, notebooks/EN/chapter0_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_00/generate_all_charts.py and Quantlets/Ch_00/seminar0.py (inspect.getsource),
so the notebooks stay in sync with the Quantlets and the slides. seminar0.py is an instructor file: the seminar
notebook is built in its full version and then split with  python3 notebooks/split_seminar_notebooks.py 0.
Run:  python3 notebooks/build_notebooks_ch0.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter0_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter0_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 0
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(0)
import generate_all_charts as g   # noqa: E402

CONSTS = (f'GDP_NSA = {g.GDP_NSA!r}\nGDP_SCA = {g.GDP_SCA!r}\nHICP = {g.HICP!r}\nELEC = {g.ELEC!r}\n'
          f'INFL_START = {g.INFL_START!r}\nTEST_H = {g.TEST_H!r}\nSEASON = {g.SEASON!r}\nSERIES_LABEL = {g.SERIES_LABEL!r}')

LECTURE = [
    md("# Time Series Analysis — Chapter 0: Introduction, Components and Exponential Smoothing\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Reproduces every chart and number of the Chapter 0 lecture.\n"
       "- Data: Eurostat (Romanian GDP, HICP, electricity generation), the BNR reference rate (EUR/RON), daily market "
       "data from EODHD (BET, S&P 500), the CO2 data set of statsmodels.\n"
       "- Sections: six series on real data; Slutsky's experiment; decomposition (classical and STL); transformations "
       "and the ACF; simple exponential smoothing; forecasts and their evaluation."),
    *common_cells(),
    md("## Chapter constants and series\n\n"
       "- Eurostat series keys (public SDMX API, no key) and the test-set lengths of the forecast comparison.\n"
       "- One function per series: each returns a pandas Series (or DataFrame) indexed by date."),
    code(CONSTS + '\n\n\n' + src(g.gdp_series, g.hicp_series, g.electricity_series, g.co2_series)),
    md("## 1. Six time series\n\n"
       "- Romanian real GDP, unadjusted and seasonally adjusted: trend, seasonality and recessions.\n"
       "- Consumer prices (HICP) and the annual inflation rate $100\\,(P_t/P_{t-12} - 1)$."),
    code(src(g.fig_gdp, g.fig_hicp) + "\n\n\ngdp = fig_gdp(save=False)\nhicp = fig_hicp(save=False)\n"
         "full = gdp['nsa'].groupby(gdp.index.year).sum()\n"
         "print('annual GDP growth 2009 and 2020, %:', round(100 * (full[2009] / full[2008] - 1), 1), round(100 * (full[2020] / full[2019] - 1), 1))\n"
         "print('latest annual inflation, %:', round(hicp['inflation'].iloc[-1], 1))"),
    md("- EUR/RON (BNR reference rate): a stochastic trend, no seasonality.\n"
       "- BET and S&P 500: value of 100 invested in January 2000, log scale."),
    code(src(g.fig_eurron, g.fig_markets) + "\n\n\nfx = fig_eurron(save=False)\nidx = fig_markets(save=False)\n"
         "print('EUR/RON first and last:', fx.iloc[0], fx.iloc[-1])\nprint(idx.iloc[-1].round(0))"),
    md("- Electricity generation in Romania: a strong winter peak and a falling level.\n"
       "- CO2 at Mauna Loa: a smooth trend and an additive yearly wave."),
    code(src(g.fig_electricity, g.fig_co2) + "\n\n\nel = fig_electricity(save=False)\nco2 = fig_co2(save=False)\n"
         "print((el.groupby(el.index.month).mean() / 1000).round(2))"),
    md("## 2. Slutsky's experiment\n\n"
       "- A moving sum of 10 independent shocks: $y_t = \\varepsilon_t + \\dots + \\varepsilon_{t-9}$.\n"
       "- Waves appear although no cycle was put in."),
    code(src(g.fig_slutsky) + "\n\n\neps, ma, up = fig_slutsky(save=False)\nprint('upward zero crossings:', up)"),
    md("## 3. Decomposition\n\n"
       "- Classical multiplicative decomposition of GDP: $2 \\times 4$ centred moving average, seasonal factors, remainder.\n"
       "- STL (robust, period 12) of the CO2 series: the seasonal pattern may change slowly."),
    code(src(g.classical_decomposition, g.fig_components) + "\n\n\ny, trend, seas, rem, q = fig_components(save=False)\n"
         "print('seasonal factors:', {k: round(v, 3) for k, v in q.items()})"),
    code(src(g.fig_stl) + "\n\n\nc, res = fig_stl(save=False)\nprint('remainder sd (ppm):', round(res.resid.std(), 3))"),
    md("## 4. Transformations and the ACF\n\n"
       "- Daily log returns of the BET: $r_t = 100(\\ln P_t - \\ln P_{t-1})$.\n"
       "- Sample ACF $r_k = c_k / c_0$ with the band $\\pm 1.96/\\sqrt{T}$: slow decay (trend), peaks at 12 (season), "
       "almost no memory (returns)."),
    code(src(g.fig_returns, g.sample_acf, g.fig_acf) + "\n\n\np, r = fig_returns(save=False)\nout = fig_acf(save=False)\n"
         "for k, (a, band, n) in out.items():\n    print(f'{k}: r1 = {a[0]:.3f}, r12 = {a[11]:.3f}, band = {band:.3f}')"),
    md("## 5. Simple exponential smoothing\n\n"
       "- $\\ell_t = \\alpha y_t + (1-\\alpha)\\ell_{t-1}$; the one-step forecast is $\\ell_{t-1}$.\n"
       "- Worked example of the slides: $y = 10, 12, 11, 13$, $\\alpha = 0.5$, $\\ell_0 = 10$."),
    code(src(g.ses_path, g.eurron_monthly, g.fig_ses) + "\n\n\n"
         "print(ses_path(pd.Series([10.0, 12.0, 11.0, 13.0]), 0.5, level0=10).values)   # levels 10, 11, 11, 12\n"
         "y, alpha = fig_ses(save=False)\nprint('estimated alpha for EUR/RON:', round(alpha, 3))"),
    md("## 6. Forecasts and their evaluation\n\n"
       "- Training set and test set split by time; naive, seasonal naive, SES and Holt-Winters.\n"
       "- MAE, RMSE, MAPE and MASE (scaled by the in-sample MAE of the seasonal naive method)."),
    code(src(g.method_forecasts, g.accuracy, g.fig_forecast) + "\n\n\ntrain, test, f, acc = fig_forecast(save=False)\n"
         "pd.DataFrame(acc).T.round(3)"),
    md("- A small M-competition: the four methods on four seasonal series."),
    code(src(g.benchmark_data, g.fig_benchmarks) + "\n\n\ntab, piv = fig_benchmarks(save=False)\npiv.round(2)"),
]


def seminar_cells():
    import seminar0 as s0
    fun = src(s0.growth_rates, s0.sample_acf, s0.paper_forecasts, s0.gdp_growth, s0.fig_gdp_growth, s0.eurron_acf,
              s0.fig_acf_pair, s0.hicp_rates, s0.electricity, s0.fig_elec_forecast)
    return [
        md("# Time Series Analysis — Seminar 0: First Steps with Time Series\n\n"
           "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
           "- The seminar comes before the Chapter 0 lecture: the definitions are in the primer below.\n"
           "- **[Solved]** exercises show the full code and output: use them as models. **[Proposed]** exercises have an "
           "empty code cell: solve them yourself; the solutions are discussed in class. Nothing is handed in."),
        *common_cells(),
        md("## Setup\n\n"
           "### Primer\n\n"
           "- Growth over one period: $100(y_t/y_{t-1} - 1)$; over a year: $100(y_t/y_{t-m} - 1)$, with $m = 4$ for "
           "quarterly and $m = 12$ for monthly data; log difference $100(\\ln y_t - \\ln y_{t-1})$.\n"
           "- Sample autocorrelation $r_k = \\sum_{t>k}(y_t-\\bar y)(y_{t-k}-\\bar y) / \\sum_t (y_t-\\bar y)^2$; "
           "band $\\pm 1.96/\\sqrt T$.\n"
           "- Naive forecast $\\hat y_{T+h} = y_T$; seasonal naive $\\hat y_{T+h} = y_{T+h-m}$.\n"
           "- Split by time into a training set and a test set; MAE $= \\frac1h\\sum|e|$, RMSE $= \\sqrt{\\frac1h\\sum e^2}$.\n\n"
           "### Seminar functions"),
        code(f"GDP_NSA = {s0.GDP_NSA!r}\nHICP = {s0.HICP!r}\nELEC = {s0.ELEC!r}\nSTART = {s0.START!r}\n\n\n" + fun),
        md("### Setup check\n\n- Romanian real GDP, quarterly, unadjusted, billion EUR (Eurostat revises its data: "
           "small differences in the last value are normal)."),
        code("g = read_eurostat('namq_10_gdp', 'Q.CLV10_MEUR.NSA.B1GQ.RO') / 1000\n"
             "print(len(g), g.index[0].date(), g.index[-1].date())\nprint(g.tail(2).round(1))\ng.plot()\nplt.show()"),
        # ---------------------------------------------------------------- A1
        md("## A1 [Solved]: Growth rates\n\n"
           "**Question:** did the Romanian economy collapse in the first quarter of 2025? **Inputs:** real GDP, billion EUR: "
           "38.7, 45.6, 52.8, 59.2 (2024 Q1–Q4), 38.9, 46.8 (2025 Q1–Q2).\n\n"
           "1. Compute the quarter-on-quarter growth rates.\n"
           "2. Compute the year-on-year growth rates for 2025 Q1 and Q2.\n"
           "3. Compute the log difference for 2025 Q1 and compare it with the quarter-on-quarter rate.\n\n"
           "**Report:** the two kinds of growth rates and one sentence that answers the question."),
        code("# Solution\na1 = growth_rates([38.7, 45.6, 52.8, 59.2, 38.9, 46.8], 4)\na1.round(2)"),
        md("**Interpretation:** the fall of a third from Q4 to Q1 happens every winter; compared with a year earlier, GDP grew."),
        # ---------------------------------------------------------------- A2
        md("## A2 [Solved]: An autocorrelation by hand\n\n"
           "**Inputs:** $y = 2, 4, 3, 5, 4, 6$.\n\n"
           "1. Compute the mean and the deviations.\n2. Compute the sums of squares and of products.\n"
           "3. Compute $r_1$, $r_2$ and the band $\\pm 1.96/\\sqrt T$."),
        code("# Solution\nx = np.array([2, 4, 3, 5, 4, 6], dtype=float)\nd = x - x.mean()\n"
             "print('mean:', x.mean(), ' sum of squares:', (d ** 2).sum(), ' sum of lag-1 products:', (d[1:] * d[:-1]).sum())\n"
             "print('r1, r2:', sample_acf(x, 2).round(3), ' band:', round(1.96 / np.sqrt(len(x)), 3))"),
        md("**Interpretation:** both values lie inside the band: 6 observations are too few to detect memory."),
        # ---------------------------------------------------------------- A3
        md("## A3 [Solved]: Naive forecasts and their errors\n\n"
           "**Inputs:** training set 10, 14, 18, 12, 11, 15, 20, 13; test set 12, 16, 21, 14; $m = 4$.\n\n"
           "1. Write the naive and the seasonal naive forecasts.\n2. Compute the errors.\n3. Compute the MAE and the RMSE."),
        code("# Solution\na3 = paper_forecasts([10, 14, 18, 12, 11, 15, 20, 13], [12, 16, 21, 14], 4)\n"
             "for k, v in a3.items():\n    print(k, 'forecasts', v['fc'], 'errors', v['e'], 'MAE', v['mae'], 'RMSE', round(v['rmse'], 2))"),
        md("**Interpretation:** the seasonal naive forecast keeps the seasonal pattern and wins."),
        # ---------------------------------------------------------------- B1
        md("## B1 [Solved]: Romanian GDP\n\n"
           "**Question:** which growth rate should be reported for an unadjusted quarterly series?\n\n"
           "1. Compute the quarter-on-quarter and the year-on-year growth rates.\n"
           "2. Plot the two rates since 2010.\n"
           "3. Count the negative values of each rate; compute the average q/q growth of the first quarters.\n"
           "4. Interpretation: which of the two rates shows the recessions?"),
        code("# Solution\nd = gdp_growth()\n"
             "print('negative q/q:', int((d['pop'] < 0).sum()), 'of', int(d['pop'].notna().sum()))\n"
             "print('negative y/y:', int((d['yoy'] < 0).sum()), 'of', int(d['yoy'].notna().sum()))\n"
             "print('average q/q growth of the first quarters, %:', round(d['pop'][d.index.quarter == 1].mean(), 1))\n"
             "print(d.tail(2).round(2))\nfig_gdp_growth(d, save=False)"),
        md("**Interpretation:** only the year-on-year rate shows the recessions (2009–2010, 2020) and the slowdown of "
           "2025–2026; for unadjusted data, report year-on-year growth."),
        # ---------------------------------------------------------------- B2
        md("## B2 [Solved]: EUR/RON, memory of levels and of changes\n\n"
           "1. Compute the daily log returns.\n2. Compute the ACF of the level and of the returns for lags 1–20.\n"
           "3. Compare $r_1$ of the returns with the band and count the lags outside it.\n"
           "4. Interpretation: is the naive forecast a sensible benchmark for the exchange rate?"),
        code("# Solution\nb2 = eurron_acf()\n"
             "print('level: r1', round(b2['acf_level'][0], 3), ' r20', round(b2['acf_level'][-1], 3))\n"
             "print('returns: r1', round(b2['acf_ret'][0], 3), ' r2', round(b2['acf_ret'][1], 4), ' band', round(b2['band_ret'], 3))\n"
             "print('lags outside the band:', int((np.abs(b2['acf_ret']) > b2['band_ret']).sum()))\n"
             "fig_acf_pair(b2['acf_level'], b2['acf_ret'], b2['band_level'], b2['band_ret'], 'EUR/RON level', "
             "'EUR/RON daily log returns', 'ch0_sem_b2_acf', save=False)"),
        md("**Interpretation:** today's level is the best simple forecast of tomorrow's level; past changes add very "
           "little, so the naive forecast is the benchmark to beat."),
        # ---------------------------------------------------------------- A4
        md("## A4 [Proposed]: New numbers\n\n"
           "**Model:** A1, A2 and A3 [Solved]. **Inputs:** the quarterly series 120, 150, 132, 168, 126; the series "
           "5, 7, 6, 8, 9; a series with $m = 3$: training 20, 30, 25, 22, 32, 27, test 23, 33, 29.\n\n"
           "1. Compute the growth rates, the log differences and the year-on-year rate of the last quarter.\n"
           "2. Compute $r_1$ of the series 5, 7, 6, 8, 9 and the band.\n"
           "3. Compute the MAE and RMSE of the naive and seasonal naive forecasts."),
        code("# Solution\nprint(growth_rates([120, 150, 132, 168, 126], 4).round(2))\n"
             "print('r1:', sample_acf([5, 7, 6, 8, 9], 1).round(3), ' band:', round(1.96 / np.sqrt(5), 3))\n"
             "a4 = paper_forecasts([20, 30, 25, 22, 32, 27], [23, 33, 29], 3)\n"
             "for k, v in a4.items():\n    print(k, 'MAE', round(v['mae'], 2), 'RMSE', round(v['rmse'], 2))"),
        # ---------------------------------------------------------------- B3
        md("## B3 [Proposed]: Romanian inflation\n\n"
           "**Model:** B1 and B2 [Solved]. Use `hicp_rates()` (monthly HICP since 2015; columns `pop` = m/m, `yoy` = y/y).\n\n"
           "1. Compute the monthly and the annual inflation rates.\n"
           "2. Compute the average monthly inflation for each calendar month.\n"
           "3. Compute the ACF of the monthly rate for lags 1–24 and of the annual rate at lag 1.\n"
           "4. Interpretation: why is the annual rate so persistent?"),
        code("# Solution\nh = hicp_rates()\nmm = h['pop'].dropna()\nprint(h.tail(2).round(2))\n"
             "print((mm.groupby(mm.index.month).mean()).round(2))\nacf_mm = sample_acf(mm, 24)\n"
             "print('m/m: r1', round(acf_mm[0], 3), ' r12', round(acf_mm[11], 3), ' band', round(1.96 / np.sqrt(len(mm)), 3))\n"
             "print('y/y: r1', round(sample_acf(h['yoy'].dropna(), 1)[0], 3))\n"
             "fig_acf_pair(acf_mm, sample_acf(h['yoy'].dropna(), 24), 1.96 / np.sqrt(len(mm)), 1.96 / np.sqrt(h['yoy'].notna().sum()),\n"
             "             'Monthly inflation (m/m)', 'Annual inflation (y/y)', 'ch0_sem_b3_acf', col_a=st.IDAred, col_b=st.Purple, save=False)"),
        # ---------------------------------------------------------------- B4
        md("## B4 [Proposed]: Electricity, naive or seasonal naive?\n\n"
           "**Model:** A3 [Solved]. Use `electricity()` (TWh per month).\n\n"
           "1. Split by time: the last 24 months form the test set.\n"
           "2. Forecast the test set with the naive and the seasonal naive method ($m = 12$).\n"
           "3. Compute the MAE and RMSE and plot both forecasts against the test data.\n"
           "4. Interpretation: why does the seasonal naive method not win clearly here?"),
        code("# Solution\ne = electricity()\ntrain, test = e.iloc[:-24], e.iloc[-24:]\nout = paper_forecasts(train.values, test.values, 12)\n"
             "for k, v in out.items():\n    v['fc'] = pd.Series(v['fc'], index=test.index)\n"
             "    print(k, 'MAE', round(v['mae'], 3), 'RMSE', round(v['rmse'], 3))\n"
             "fig_elec_forecast(train, test, out, save=False)"),
        # ---------------------------------------------------------------- C2
        md("## C2 [Proposed]: Critique an AI answer\n\n"
           "**Prompt** sent to an AI assistant: *From the quarterly Romanian real GDP, compute the average annual growth rate "
           "and the RMSE of a naive forecast.* The AI's code is below; it runs without an error message. Its conclusion: "
           "*forecasting GDP is hopeless.*\n\n"
           "**Model:** A1, A3 and B1 [Solved].\n\n"
           "1. Run the AI's code and read its output.\n"
           "2. Find the three errors planted in the code.\n"
           "3. For each error, say how you would detect it.\n"
           "4. Write the corrected code, with the last 8 quarters as the test set, and report the corrected numbers.\n"
           "5. Say whether the conclusion survives."),
        code("# The AI's answer (do not edit this cell)\n"
             "g = read_eurostat('namq_10_gdp', 'Q.CLV10_MEUR.NSA.B1GQ.RO') / 1000\n"
             "growth = g.diff()                                # annual growth rate, %\n"
             "print('average annual growth:', growth.mean())\n"
             "test = g.sample(frac=0.2, random_state=1).sort_index()   # test set\n"
             "train = g.drop(test.index)                               # training set\n"
             "fc = train.reindex(g.index).ffill().shift(1).reindex(test.index)  # naive\n"
             "rmse = ((test - fc) ** 2).mean()\n"
             "print('RMSE of the naive forecast:', rmse)"),
        code("# Solution\nyoy = 100 * (g / g.shift(4) - 1)                   # error 1: growth in %, over 4 quarters\n"
             "train, test = g.iloc[:-8], g.iloc[-8:]               # error 2: split by time\n"
             "e_naive = test - train.iloc[-1]\n"
             "e_snaive = test.values - np.array([train.iloc[len(train) - 4 + (i % 4)] for i in range(8)])\n"
             "print('average annual growth, %:', round(yoy.mean(), 2))\n"
             "print('RMSE naive:', round(np.sqrt(np.mean(e_naive ** 2)), 2), ' seasonal naive:',\n"
             "      round(np.sqrt(np.mean(e_snaive ** 2)), 2))    # error 3: the square root"),
        md("**Solution discussion:** growth is a percentage over four quarters; the test set must come after the training "
           "set; RMSE needs the square root. With the corrections, a seasonal benchmark forecasts GDP within about 1%."),
    ]


if __name__ == '__main__':
    build(LECTURE, 0, 'lecture')
    try:
        build(seminar_cells(), 0, 'seminar')
    except ImportError:
        print('seminar0.py (instructor file) not found: seminar notebook not rebuilt')
