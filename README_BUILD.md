# TSA build pipeline

How the materials of *Serii de timp / Time Series Analysis* (bachelor, year 3, semester 2, Informatică economică and Cibernetică economică) are built. The pipeline is copied from the SFM course, which was adapted from MFM: same slide design, same bilingual generators, same data and notebook rules. Run every command from the repository root. The chapter rules are in `tools/HOUSE_RULES.md`.

## Layout

| Path | Content |
|---|---|
| `index.html`, `index_ro.html` | Site shell (EN, RO). Everything else comes from `assets/`. |
| `assets/site.js`, `assets/site.css` | Rendering and quiz engine (shared with the MFM and SFM sites). |
| `assets/config.js` | Google client ID (shared "MFM Quiz Login" client), Apps Script URLs, instructor e-mails. Values starting with `YOUR_` count as not configured. |
| `assets/course-data.js` | Course facts, chapters 0–15 with their links, project, AI policy, resources, bibliography. |
| `assets/quizzes/<id>.js` | Quiz bank of one chapter (`window.TSA_DATA.quizzes['<id>']`, `draw: 20`). |
| `latex/preamble.tex` | Shared Beamer preamble. It defines the logos, `\quantlet`, `\tsaquantlet`, `\colaburl`, `\nb`, `\itemsize`, `\imgcredit`, `\imgcap`, `\ifsolutions`/`\solonly`/`\propsub`, the appendix and chapter-link macros, and the vector macros of the older TSA decks. |
| `latex/tsa_chapters.py` | Chapter registry (titles as in `assets/course-data.js`) and the file-naming scheme. |
| `latex/tsa_build.py` | Common generator framework: `⟦EN‖RO⟧` text, `@{key}` numbers, the `Deck` class, legacy conversion, compilation. |
| `latex/build_chapterN.py`, `latex/build_seminarN.py` | One bilingual generator per deck (⟦EN‖RO⟧), as in SFM. |
| `latex/acronyms.py`, `_acr_scan.py`, `acronyms_extra/chN.py` | Acronym glossary, inserted after the title page. |
| `latex/appendix_links.py` | Appendix buttons and back-buttons; "Chapter N" mentions become links to the PDF on the site. |
| `data/market/*.csv`, `data/manifest.csv` | Daily market data from EODHD, saved once (91 series, copied from SFM/MFM, ending 18.09.2026). |
| `Quantlets/common/tsa_data.py` | Data loader: local `data/market` or the raw GitHub URL of this repository; BNR reference rate, FRED, Eurostat (online, no key); statsmodels data sets. |
| `Quantlets/common/tsa_style.py` | Chart style: transparent background, legend below the plot, course palette, no grey; `check_no_grey`. |
| `Quantlets/common/tsa_quantlets.py` | Quantlet builder: `Metainfo.txt` plus a self-contained Colab notebook plus charts. |
| `Quantlets/Ch_NN/` | Per rebuilt chapter: `generate_all_charts.py`, `build_quantlets.py`, `TSA_chN_*` folders. The 2025/2026 folders `Quantlets/TSA_chN` stay until their chapter is rebuilt. |
| `notebooks/tsa_notebook.py`, `notebooks/build_notebooks_chN.py` | Notebook builders. Output is English only, in `notebooks/EN/`. |
| `notebooks/add_colab_banner.py` | Adds the "Save a copy in Drive" banner and the optional Drive-save cell (`MyDrive/TSA/Chapter_N`). |
| `notebooks/split_seminar_notebooks.py`, `split_quantlet_seminars.py` | Split each seminar notebook and Quantlet into a public student version and a private instructor version. |
| `tools/HOUSE_RULES.md` | Rules for every chapter (level, language, slides, data, charts, seminars, quizzes). |
| `tools/legacy/` | The chart scripts of the 2025/2026 edition (kept for reference; they still use yfinance). |

### File names

Slugs are derived from the chapter titles. To list them all, run `python3 latex/tsa_chapters.py`.

| Material | Path |
|---|---|
| Lecture, EN | `EN/Courses/chapterN_<slug_en>.pdf` |
| Lecture, RO | `RO/Cursuri/capitolN_<slug_ro>.pdf` |
| Seminar, EN | `EN/Seminars/seminarN_<slug_en>.pdf` (instructor version: `…_solutions.pdf`, git-ignored) |
| Seminar, RO | `RO/Seminarii/seminarN_<slug_ro>_ro.pdf` (instructor version: `…_solutions.pdf`, git-ignored) |
| Notebooks | `notebooks/EN/chapterN_lecture_notebook.ipynb`, `notebooks/EN/chapterN_seminar_notebook.ipynb` |
| Site | `https://danpele.github.io/Time-Series-Analysis/<path>` |

Glossary and chapter-link processing only touches decks that sit at these paths **and** contain `\input{../../latex/preamble}`. The 2025/2026 decks in `EN/Courses`, `EN/Seminars`, `RO/Courses`, `RO/Seminars` (old names, own `preamble.tex`) stay as they are until their chapter is rebuilt. Two old EN lecture files already have the new names (`EN/Courses/chapter2_arma_models.tex`, `EN/Courses/chapter7_cointegration_vecm.tex`): the rebuilt decks will replace them.

## Chapters

| N | id | Title (RO) | Status |
|---|---|---|---|
| 0 | intro | Introducere: componente și netezire exponențială | converted to the new preamble (content of 2025/2026) |
| 1–9 | stationarity, arma, arima, seasonal, garch, var, cointegration, long-memory, ml | see `latex/tsa_chapters.py` | 2025/2026 decks linked |
| 10 | state-space | Modele în spațiul stărilor, filtrul Kalman și modele Markov switching | new, in preparation |
| 11–14 | foundation-models, spectral, lppl, mgarch | studiu individual (`selfStudy: true`) | 2025/2026 decks linked |
| 15 | review | Recapitulare și pregătire pentru examen | 2025/2026 deck linked |

## Building a chapter (commands in order)

```bash
# 1. charts, tables and numbers (Quantlets/Ch_NN)
python3 Quantlets/Ch_NN/generate_all_charts.py
python3 Quantlets/Ch_NN/seminarN.py                    # if the chapter has seminar computations
# 2. Quantlet folders (Metainfo.txt + Colab notebook + charts)
python3 Quantlets/Ch_NN/build_quantlets.py
# 3. decks EN + RO (each generator also runs latex/acronyms.py N, which runs appendix_links.py)
python3 latex/build_chapterN.py
python3 latex/build_seminarN.py                        # also writes the *_solutions.tex wrappers
# 4. compile: pdflatex twice per deck (and per _solutions wrapper); prints errors and overfull vbox
python3 latex/tsa_build.py compile N
# 5. notebooks (EN only), then execute them
python3 notebooks/build_notebooks_chN.py
jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapterN_lecture_notebook.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapterN_seminar_notebook.ipynb
# 6. Colab banner + Drive-save cell (new-pipeline notebooks and Quantlets only; idempotent)
python3 notebooks/add_colab_banner.py
# 7. seminars: student version public, full version private (../instructor/Quantlets/Ch_NN)
python3 notebooks/split_seminar_notebooks.py N         # also runs split_quantlet_seminars.py N
python3 notebooks/split_quantlet_seminars.py --write-gitignore   # private answer charts -> .gitignore block
python3 notebooks/split_quantlet_seminars.py --check   # nothing private in public folders or in git
```

Re-run step 7 after **any** `build_quantlets.py`, because that script rewrites the full seminar notebooks.

Before publishing, update the chapter's links in `assets/course-data.js` (slides, seminar, the Colab notebook, the Quantlet folder `Quantlets/Ch_NN`), extend its quiz to 24 questions, and bump the `?v=` cache keys in both `index*.html`.

### Converting an existing deck instead of rewriting it

```bash
python3 latex/tsa_build.py convert <old.tex> N en|ro [lecture|seminar]
```

The converter keeps everything after `\begin{document}`. It replaces the old `\input{preamble}` header with the shared preamble and the standard title, and writes the deck under its new name. Rebuilt chapters do not use the converter: they have their own generators (`build_chapterN.py`, `build_seminarN.py`).

## Writing a generator

`latex/tsa_build.py` has a worked example in its docstring. The rules:

- Write all text as `⟦english||română⟧`.
- Never type numbers by hand. Use `@{key}` with `Values.put(key, x, decimals)`.
- In RO decks, decimal points become commas and „de” is added after numerals of 20 and above (for example, „131 de zile”).
- Lecture helpers: `D.section`, `D.frame`, `D.chart` (chart plus Quantlet link), `D.recap`, `D.references`.
- Seminar helpers: `D.solved` (task and solution, visible to everyone), `D.proposed` (solution only with `\solutionstrue`), `D.task` (an explicit task with numbered sub-tasks and what to report), `D.frame(..., instructor_only=True)`.
- Images: `photo(file, caption, url, credit)` adds a visible credit; record the licence in `photos/CREDITS.md`. Citations are `\href` links to the DOI.
- Chapter-specific acronyms go in `latex/acronyms_extra/chN.py`. The script prints `NOT IN DICTIONARY` for any acronym that is missing.
- Each lecture ends with a short section "Contribuția posibilă a AI" / "Possible contribution of AI".

## Data

- Daily market data from EODHD, saved once in `data/market`. Do not call any API with a key.
- Convention for which price column to use:
  - Indices, FX, crypto and yields use `close`.
  - ETFs and stocks use `adjusted_close`.
- Weekend quotes and unchanged (holiday-filled) closes are dropped, except for crypto.
- Annualise with the actual observation frequency of each series.
- EUR/RON is the official BNR reference rate (`tsa_data.load_close('eurron')`). The EODHD EUR/RON series contains erroneous quotes.
- Macroeconomic series are read online without keys: `tsa_data.read_fred('UNRATE')`, `tsa_data.read_eurostat('namq_10_gdp', 'Q.CLV10_MEUR.SCA.B1GQ.RO')` (Romanian real GDP), INS TEMPO and BNR files; textbook series with `tsa_data.load_statsmodels('sunspots' | 'co2' | 'macrodata' | 'nile')`.
- New code reads data only through `tsa_data.py`. The old Quantlets and `tools/legacy` still use yfinance and pandas_datareader; they are replaced chapter by chapter.

## Private material (never committed)

- Instructor versions of seminars (`*_solutions.*`), `instructor/`, exams and their solutions.
- The unpublished "Applied Time Series Solutions" book material (ATSSB decks, solutions manual) and the Brockwell–Davis solutions.
- `apps_script_code.js` (old GitHub-OAuth script with a client secret) and `_old/` (local copies of the files removed on 5 October 2026).

## Chapter 0 (pipeline check)

```bash
python3 latex/build_chapter0.py && python3 latex/build_seminar0.py && python3 latex/tsa_build.py compile 0
```
