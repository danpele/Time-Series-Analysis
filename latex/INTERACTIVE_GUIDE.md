# Interactive chapter pages (TSA) — how to build chapter N

The TSA infrastructure is a port of the MFM one. TSA has no finished interactive chapter yet, so the reference
implementation is MFM Chapter 4 (read it, never edit it):
- `/Users/danpele/Documents/2026 MFM/MFM/interactive/ch4_ro.html`, `interactive/ch4.js`, `interactive/ch4_data.js`
- `/Users/danpele/Documents/2026 MFM/MFM/latex/build_interactive_ch4.py` (the chapter spec)
- `/Users/danpele/Documents/2026 MFM/MFM/Quantlets/Ch_04/export_interactive.py` (writes ch4_data.js from the chapter's own code)

TSA files (paths relative to the TSA repo root):
- `interactive/common.js` — shared helpers exposed as `window.TSAI` (page language in `window.TSA_LANG`): linear algebra
  (`dot, mv, quad, solve, chol, eigSym, cov2corr`), time series (`diff, acf, pacf` (Durbin–Levinson), `ljungBox, armaSim,
  garchSim`), `normCdf, normPpf, quantile`, seeded Normal draws `rng(seed)`, `mean, sdev, median`, formatting `nf, pct`
  (RO decimal comma), Plotly helpers `layout, AX, ttl, plot, vline, hline, CFG`, `Player` (▶ button), `range`, and the
  opt-in `predict` and `exitTicket`. **Do not edit it**; put chapter helpers in `chN.js`. If a helper belongs in common.js,
  say so in your report.
- `interactive/interactive.css` — layout of the interactive pages (do not edit).
- `latex/interactive_common.py` — builds the pages and slide images from a spec (do not edit).
- `latex/check_interactive_page.sh` — headless check (below).

## What a chapter page is
- One block per **data chart in the lecture body** (before the appendix / „Bibliografie”), in slide order, grouped by the
  lecture's sections. Skip portraits, photos, historic images and figures copied from papers. If a chart cannot be made
  interactive in a meaningful way, still give it a block where the user can at least hover and change one meaningful
  parameter (sample, series, lag, threshold); never skip silently — list skipped charts in the report with the reason.
- Left: the slide image (built automatically from the .nav file). Right: the same figure computed in the browser. **Its initial
  state reproduces the slide chart** (same data, same parameters, same numbers).
- Every block has a ▶ button (`Player`) that moves one meaningful parameter step by step (an AR coefficient, a lag order, a
  window, the GARCH persistence, the forecast horizon, the sample end, the number of simulations …). Choose the parameter
  that shows the mechanism of the slide.
- Under each block: one „Întrebare pentru sală” / "Question for the room" and its „Răspuns” / "Answer", taken from what the
  slide teaches, with the numbers of the chapter.
- RO page `interactive/chN_ro.html` and EN page `interactive/chN.html` (texts in the spec; chart text always in English).
- The web lecture (`latex/build_web_lecture.py`) picks the spec up automatically: every chart of BLOCKS is replaced in its
  frame by the live block, and the nav bar of `web/chN*.html` gets the "Interactive page" link. Rebuild the web pages after
  building the interactive page.

## Steps
1. Read the chapter's RO and EN lecture .tex (body only): `RO/Cursuri/capitolN_*.tex`, `EN/Courses/chapterN_*.tex` — take
   the exact names from `python3 latex/tsa_chapters.py` (EN/Courses also holds older decks with other names: ignore them).
   Read `Quantlets/Ch_NN/` (`generate_all_charts.py`, `chN_numbers.json`, the CSV tables).
2. Write `Quantlets/Ch_NN/export_interactive.py`: call the chapter's own functions; export only what the page needs (series,
   estimates, test results, forecasts). Re-use the numbers JSON; do not re-implement the chapter's estimation differently.
   Reading `data/` files is fine; never call a data API that needs a key. **Never run `build_quantlets.py`** (it rewrites
   every Quantlet folder of the chapter).
3. Write `interactive/chN.js` (`const { ... } = window.TSAI;`) and `latex/build_interactive_chN.py` (copy the structure of
   MFM's `build_interactive_ch4.py`: `N`, `TEX = {'ro': 'RO/Cursuri/capitolN_…', 'en': 'EN/Courses/chapterN_…'}`,
   `QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_NN'`, BLOCKS, PFX, SECTIONS, L, widgets,
   QA, TEXT_JS; `L[lang]` needs html_lang, course ('Serii de timp' / 'Time Series Analysis'), h1, file, site
   ('../index_ro.html#chapters' / '../index.html#chapters'), back, pdf_btn, ql, slide, ask, answer, note, src).
4. `cd latex && python3 build_interactive_chN.py` (writes the pages, readouts and slide images), then
   `python3 build_web_lecture.py N`.
5. Check (server: `cd <TSA repo> && python3 -m http.server 8791`; port 8765 is used by the MFM server):
   `cd interactive && ../latex/check_interactive_page.sh chN_ro.html` and the same for `chN.html`. Must print `[]` errors,
   `empty=` nothing, plots > 0, page width = window width. Then `cd web && ../latex/check_web_page.sh chN_ro.html` and
   `... 390` (see WEB_GUIDE.md). Run the checks one after the other, not in parallel (parallel runs give false widths).
6. Verify numbers: compare the page's initial readouts with `chN_numbers.json` / the slide numbers (node script that loads
   `chN_data.js` + `common.js` + `chN.js` with `window`/`document` stubs and calls the exported functions).
7. Look at it: headless Chrome screenshots, crop and inspect every block: labels must not overlap, legends below the chart,
   nothing cut.

## Rules
- Language (both pages): no AI tone. No slogan titles, no promotional intro, no punchlines, no meta remarks about the page.
  Block titles = slide titles (automatic). Readouts state results plainly, with numbers that say what they are.
- RO orthography: „sînt”, „î” inside words, „â” only in the „român” family; „distribuția Normală”; „media” (not „speranța”);
  English finance jargon stays English (volatility clustering, backtesting, drawdown …). Never „preziceți”.
- VaR convention: „VaR 1%” / "VaR 1%", never "VaR 99%"; VaR_α = −q_α.
- Charts: transparent background, legend outside at the bottom, no grey series or grey text (navy #1F2A44 for reference
  lines), the chapter's own series colours (from its `generate_all_charts.py`), English text.
- Data and numbers must match the lecture: never substitute another series, never rescale to match. A simulation shown by
  the page must default to the chapter's parameters. If something cannot be reproduced, say so in the report.
- Do not modify lecture decks, `build_chapterN.py`, seminar files, `common.js`, `interactive_common.py`, `assets/*`, or other
  chapters' files. Seminar sources (`.tex` with solutions, `build_seminar*.py`, seminar scripts in Quantlets) are private:
  never copy solutions into the pages, never link them. Student seminar PDFs may be linked.
- Do not add buttons to the course site (index.html / course-data.js). No git commit / push.

## Predictions and exit tickets
Same rules as MFM (opt-in `TSAI.predict`, `TSAI.exitTicket`; spec fields `PREDICT`, `TICKET`, `QA`):
- The question must not be answerable by reading the page; otherwise ask what ▶ will do or veil the slide/readout/charts
  (`veil: { out: true, charts: true }`).
- Ask exactly one thing; every wrong option encodes one identifiable misconception (e.g. reading a significant PACF spike
  as an MA order, a unit-root test's non-rejection as proof of a unit root, α + β < 1 read as "no volatility clustering",
  Granger causality read as economic causality); numeric distractors come from a real mistake, named in the feedback.
- The key is verified against what the page computes (node script), not against memory.
- "Not significant" never means "equal" or "no structure", only "not separated from noise by this test".
- Exit ticket: 3–5 questions from the chapter's own check-yourself slides, with their answers; every option gets a "why".
- RO: „Ce credeți?”, „Alegeți”, never „preziceți”.

## Report (final message)
Files created, number of blocks, skipped charts with reason, check output (RO and EN), numbers verified against which
results, anything left open.
