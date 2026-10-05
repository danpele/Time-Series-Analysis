"""
build_notebooks_ch11.py -- lecture notebook of Chapter 11 (TSA): foundation models for time series (self-study)
=============================================================================================================
Output: notebooks/EN/chapter11_lecture_notebook.ipynb (self-study chapter: no seminar notebook)
The code is taken from Quantlets/Ch_11/generate_all_charts.py (inspect.getsource), so the notebook stays in sync with
the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch11.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter11_lecture_notebook.ipynb
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(11)
import generate_all_charts as g   # noqa: E402
import build_quantlets as bq      # noqa: E402

CONSTS = '\n'.join(bq.CONSTS)

LECTURE = [
    md("# Time Series Analysis — Chapter 11: Foundation models for time series\n\n"
       "*Lecture notebook (self-study chapter). Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- A foundation model is pretrained once on many series and then forecasts a new series **zero-shot**, from its "
       "recent history only.\n"
       "- We use small open models that run on a CPU: **Chronos-Bolt tiny** (9 million parameters), **Chronos-Bolt small** "
       "(48 million) and **Chronos-2** (Amazon, Apache-2.0 licence), through the `chronos-forecasting` package. No paid API.\n"
       "- We compare them honestly with the seasonal naive forecast, ETS and ARIMA on Romanian and other series, with "
       "point (MASE) and probabilistic scores (weighted quantile loss, a CRPS approximation; coverage of the 80% interval).\n"
       "- The first code cell installs `chronos-forecasting` if it is missing. If the installation or the model download "
       "fails, the foundation models are skipped and the rest of the notebook still runs.\n"
       "- References: Ansari et al. (2024) Chronos, arXiv:2403.07815; Das et al. (2024) TimesFM, arXiv:2310.10688; "
       "Woo et al. (2024) Moirai, arXiv:2402.02592; Rasul et al. (2023) Lag-Llama, arXiv:2310.08278; Aksu et al. (2024) "
       "GIFT-Eval, arXiv:2410.10393; Gneiting and Raftery (2007), doi:10.1198/016214506000001437; Hyndman and Koehler "
       "(2006), doi:10.1016/j.ijforecast.2006.03.001."),
    *common_cells(bq.INSTALL),
    md("## Definitions and helpers used in the whole notebook\n\n"
       "- `get_series(key)`: the seven evaluation series (`load`, `ip`, `gdp`, `infl`, `unemp`, `eurron`, `retail`).\n"
       "- `chronos_pipeline(name)`: loads a Chronos model from Hugging Face (or returns `None`: graceful fallback).\n"
       "- `fm_quantiles(name, contexts, h)`: zero-shot quantiles at the levels 0.1, ..., 0.9.\n"
       "- `snaive_q`, `ets_q`, `arima_q`: the statistical benchmarks, with Normal quantiles.\n"
       "- `pinball`, `wql_parts`, `mase`, `crps_normal`: the scores; `backtest(key, max_origins)`: rolling-origin evaluation."),
    code(CONSTS + '\n\n\n' + src(*bq.CORE)),
    md("## 1. Is a foundation model available?\n\n- Load Chronos-Bolt small once and count its parameters."),
    code("pipe = chronos_pipeline('Chronos-Bolt small')\n"
         "print('Chronos-Bolt small loaded:', pipe is not None)\n"
         "print({name: n_parameters(name) for name in FM})"),
    md("## 2. Tokens, patches and scores\n\n"
       "- Mean scaling and quantisation (Chronos); patching (PatchTST, TimesFM, Chronos-Bolt); the pinball loss and the CRPS."),
    code(src(g.worked_examples, g.fig_series, g.fig_tokens, g.fig_patching, g.fig_scores)
         + "\n\n\nex = worked_examples()\nprint('scale s =', round(ex['s'], 3), '| scaled:', [round(z, 3) for z in ex['z']], '| tokens:', ex['tok'])\n"
           "print('attention weights:', [round(w, 3) for w in ex['att_w']], '| pinball losses:', ex['pin'])\n"
           "fig_series(save_it=False)\nprint(fig_tokens(save_it=False))\nfig_patching(save_it=False)\nfig_scores(save_it=False)"),
    md("## 3. Zero-shot forecasts\n\n- Romanian load (48 hours), Romanian industrial production (12 months), EUR/RON (20 days)."),
    code(src(g.fan, g.fig_fan_load, g.fig_fan_monthly, g.fig_fan_eurron)
         + "\n\n\nprint(fig_fan_load(save_it=False))\nprint(fig_fan_monthly(save_it=False))\nprint(fig_fan_eurron(save_it=False))"),
    md("## 4. An honest evaluation\n\n"
       "- Rolling origins, the same context and horizon for every model, scores relative to the seasonal naive forecast.\n"
       "- To keep the notebook fast we use the **last 24 origins** of each series; the slides use all origins "
       "(`run_all(max_origins=None)`, several minutes on a CPU), so your numbers differ slightly from the slides."),
    code(src(g.table_results, g.bars, g.fig_benchmark, g.fig_coverage, g.fig_horizon)
         + "\n\n\nallr, infos = run_all(max_origins=24, save_csv=False)\nR = fig_benchmark(allr, save_it=False)\n"
           "print(pd.DataFrame({k: {m: round(R[k][m]['rel_wql'], 3) for m in R[k] if not m.startswith('_')} for k in EVAL_SERIES}).T)\n"
           "print('geometric means:', {m: round(v['rel_wql'], 3) for m, v in R['_gm'].items()})\n"
           "fig_coverage(allr, save_it=False)\nprint(fig_horizon(allr, save_it=False))"),
    md("## 5. Context length, contamination and size\n\n"
       "- Accuracy against the context length (12 weekly origins of the load); origins before and after the release of "
       "the weights; accuracy against the number of parameters."),
    code(src(g.fig_context, g.fig_prepost, g.fig_size)
         + "\n\n\nprint(fig_context(save_it=False, n_origins=12))\nprint(fig_prepost(allr, save_it=False))\nprint(fig_size(allr, save_it=False))"),
    md("## 6. Exercises for self-study\n\n"
       "1. Replace Chronos-Bolt small with Chronos-Bolt tiny in `fig_fan_load`. Report the MASE and the coverage of the "
       "80% interval. Is the smaller model much worse?\n"
       "2. Add a series of your choice (for example `read_fred('INDPRO')` or a Eurostat series of another country) to "
       "`EVAL_SERIES` and run `backtest` on it. Report the relative WQL of every model.\n"
       "3. Shorten the context of the monthly series to 36 months. Does the ranking of the models change?\n"
       "4. Interpretation: in one sentence, for which of the seven series would you use a foundation model in practice, "
       "and why?"),
]

if __name__ == '__main__':
    build(LECTURE, 11, 'lecture')
