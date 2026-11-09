# Web lectures (TSA) — how to write the course text of chapter N and build/check its pages

Every TSA chapter 0–15 has a web lecture in two languages, generated from the lecture deck:
`web/chN_ro.html` (from `RO/Cursuri/capitolN_*.tex`) and `web/chN.html` (from `EN/Courses/chapterN_*.tex`).
Without course text, a page shows the slides converted to HTML (the current skeleton). The prose you write goes in
`latex/web_text/chN_ro.md` and `latex/web_text/chN_en.md`; the builder puts it before each frame's own content, which then
moves into a compact „Pe scurt” / "In brief" box.

Finished example of the target quality (MFM course, same builder; read, do not edit):
`/Users/danpele/Documents/2026 MFM/MFM/latex/web_text/ch4_ro.md` and `ch4_en.md`, rendered in
`/Users/danpele/Documents/2026 MFM/MFM/web/ch4_ro.html`.

## Files
- `latex/build_web_lecture.py` — the builder (its docstring is the format reference).
- `latex/refresh_web_text_keys.py` — after a deck change that needs no new prose: updates `{h=…}` and renames keys.
- `latex/check_web_page.sh` — headless check of a page.
- `web/web.css`, `web/web.js` — layout and behaviour; `web/slides/` (fallback slide images), `web/img/` (generated images).
- `web/chN_log.txt` — build log: fallback frames, text out of date, text without a matching frame, unused keys.
- The chapter→deck map is `latex/tsa_chapters.py` (`paths(n)`), the same files the site links in `assets/course-data.js`.
  `EN/Courses` also holds older decks with other names (`chapter2_arma_models.tex` is current, `chapter3_arima_models.tex`
  is not): never write text against the old ones.

## Commands
```
cd "<TSA repo>/latex"
python3 build_web_lecture.py N            # builds web/chN_ro.html and web/chN.html, prints the report
python3 build_web_lecture.py all          # chapters 0–15
python3 build_web_lecture.py N --hashes   # prints "ro section: # <title>" and "ro p. 12: ## <frame key> {h=<hash>}": copy these headings
python3 build_web_lecture.py N --keys     # prints the citation keys (@Key) of the chapter's bibliography
python3 refresh_web_text_keys.py N        # after deck edits: refresh hashes / renamed keys in both .md files
```
Check (server on port 8791, since 8765 is the MFM server):
```
cd "<TSA repo>" && python3 -m http.server 8791         # in the background
cd web && ../latex/check_web_page.sh chN_ro.html        # 1440 px
../latex/check_web_page.sh chN_ro.html 390              # phone width
```
It must print `[]` (no JS errors), `mj=done`, `merror=0`, `w=1440/1440` (or `390/390`) and an empty `wide=`. Run the checks
one after the other; parallel runs give false widths. Then look at the page yourself (headless Chrome screenshots of the
frames you wrote, at 1440 and 390 px; with `html { scroll-behavior: smooth }` scroll with `{behavior: 'instant'}`).

## Format of `latex/web_text/chN_{ro,en}.md`
```
## @intro                                 (optional, before the first #) chapter opener after the title frame
@epigraph <Key> [| <title>]               verbatim quotation (lines starting with >) from a work of the chapter's bibliography
> quotation in the original language
@portrait <file> | <name> | <credit> | <URL>   (optional) portrait left of the epigraph; file under photos/
@text
2–3 paragraphs: why the chapter's question matters
@chart <anchor or chart base name>        opening chart (a live block's anchor, or the base name of an image in charts/)
caption lines
@goals
- what the reader will be able to do (derive / estimate / test / interpret …)
@plan
- <section title exactly as in the deck> | <one-line description>

# <section title>                         text up to the first ## is the section introduction
## <frame key> {h=<hash>}                 text of that frame (2–3 paragraphs)
@after                                    (optional) paragraphs below go after the live block of the frame
## @end                                   synthesis paragraph(s) of the current section, after its last frame
```
- Section titles (`#`) and frame keys (`##`) must be the plain text the builder prints with `--hashes`. A repeated frame title gets
  ` #2`, ` #3`. The first (title) frame has key `p1`: leave it without text.
- Paragraphs are separated by blank lines. `$…$` inline math, `$$…$$` display (MathJax; the deck macros `\E`, `\Var`,
  `\Cov`, `\Corr`, `\bY`, `\bPhi`, … work). `[text](url)` links, `**bold**`, `*italic*`. `<!-- … -->` comments.
- Numbered equations: a paragraph `$$ … $$ {#eq:label}` gets the number (N.k); `@eq:label` in the text prints "(N.k)".
  RO and EN must have the same labels in the same order (the build stops otherwise).
- Citations: `@Key` → "Box și Jenkins (1970)" / "Box and Jenkins (1970)"; `[@Key]` → "(Box și Jenkins, 1970)";
  `[@Key1; @Key2]`; `[[refX]]` (the deck macro) = `@Key`. Keys: `--keys` (surnames + year, without „și”/"and",
  e.g. `@HuangPetukhina2022`, `@HyndmanEtAl2008`). Only works of the chapter's own bibliography; RO and EN must cite the
  same set of works (the build stops otherwise). An unknown key stops the build. The page ends with the list of works.
- The hash `{h=…}` records which version of the frame the text was written for; after a deck change the log says
  "text out of date": re-read the frame, adjust the text if needed, then `refresh_web_text_keys.py N`.
- After building, `web/chN_log.txt` must show no "text without a matching frame/section" and no "text out of date".

## Content rules
- **Textbook prose for every frame**: 2–3 paragraphs that teach the frame's content — definitions, the reasoning behind a
  result, how to read the chart or table, what the numbers mean. The slide content stays in the „Pe scurt” / "In brief" box
  (automatic). Graduate-textbook level; do not repeat the bullets as sentences, explain them.
- Frame titles verbatim: the `##` key is the deck title, never reworded.
- No new claims or numbers: every number comes from the slides or the chapter's results (`Quantlets/Ch_NN/chN_numbers.json`,
  CSV tables). Every number says what it is ("the Ljung–Box statistic for 10 lags is 14.2, p-value 0.16", never "14.2 (0.16)").
- No AI tone: no slogans, no punchlines, no rhetorical teasers, no "Let's dive in", no promotional intro.
- No process remarks: never "verified in Quantlet", "exactly as in the paper", "same numbers as the package", no durations
  or session notes.
- No computation-report tone ("we checked …", "we dropped …", "we ran …"): teach the concepts and the findings, with the
  data as evidence.
- Never write "not examined" / „nu a fost examinat”. If the chapter does not cover something, do not mention it.
- "Not significant" means "not separated from noise by this test", never "equal" or "no effect".
- VaR convention: "VaR 1%" / „VaR 1%” (never "VaR 99%"), with VaR_α = −q_α; "ES 2.5%".
- Romanian:
  - „sînt” (never „sunt”), „î” inside words („pînă”, „cînd”, „rămîne”), „â” only in the „român” family; „nicio/niciun”.
  - „distribuția Normală” (never „Normala” alone as a noun); „media” (never „speranța”).
  - English financial and statistical jargon stays in English (volatility clustering, backtesting, drawdown, out-of-sample,
    shrinkage, leverage effect …), no calques.
  - At the first use of a Romanian technical term, give the English term in italics in parentheses:
    „rădăcină unitară (*unit root*)”, „zgomot alb (*white noise*)”, „funcția de autocorelație parțială (*partial
    autocorrelation function*)”.
  - Question prompts: „Întrebare pentru sală” / „Ce credeți?”, never „preziceți”.
- English and Romanian texts say the same thing (same paragraphs, same numbers, same citations, same equations).

## What the converter does with the deck (for reference)
Text, lists, math, links, blocks, theorems, columns, tables (booktabs, `\multicolumn`), images (charts/, photos/, logos/),
code listings of fragile frames (as `<pre>`), `\quantlet`, `\tsaquantlet`, `\quantinar`, `\tsachlink`, `\tsachlinkt`
(links to other chapters' PDFs), `\tsaapplink` / `\tsaappback` (appendix jumps), `\imgcap`, `\imgcredit`, `\aiprompt`
are converted. A frame it cannot render faithfully (TikZ, an unknown macro or environment) is shown as its slide image
(logged as "fallback" with the reason); your text still goes above it. Fallbacks after the first build (1 Nov 2026):
ch0 „Cronologia analizei seriilor de timp” / "A timeline of time series analysis" and „Setul de antrenare și setul de test” / "Training set and test
set", ch2 „Bucla Box–Jenkins” / "The Box–Jenkins loop", ch15 „Harta cursului: Capitolele 0–14” / "The course map:
Chapters 0–14" (all TikZ diagrams). The deck's own bibliography frames are hidden and replaced by the generated list.
A link to a private file (a `.tex`, a `*_solutions` file) makes the frame fall back and is logged: seminar sources are
never published; student seminar PDFs may be linked.

## Do not
- edit the decks or `build_chapterN.py` (fix the deck through its builder, separately, if a slide is wrong; report it);
- run any `build_quantlets.py`;
- add buttons or links to the course site (`index*.html`, `assets/course-data.js`);
- commit or push.
