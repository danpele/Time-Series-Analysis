# TSA — reguli pentru fiecare capitol (stabilite de titular, 5 octombrie 2026)

Curs: „Serii de timp” / „Time Series Analysis”, licență, anul III, semestrul 2, 4 ECTS, 2 ore de curs și 1 oră de seminar pe săptămînă; programele Informatică economică și Cibernetică economică, ASE București. Titular: Prof. dr. Daniel Traian Pele. Titularul de seminar nu este încă stabilit (placeholder în `assets/config.js`, INSTRUCTORS; autor pe slide-uri: doar Pele).

Repo: `~/Documents/Teaching/TSA - Serii de timp/repo` (GitHub `danpele/Time-Series-Analysis`, site https://danpele.github.io/Time-Series-Analysis/). Cadrul tehnic este copiat din SFM; vezi `README_BUILD.md`.

## Nivel și public
- Licență, anul III: explicații complete, fără nivel de cercetare. Fiecare noțiune se definește înainte de prima folosire, cu intuiție, formulă și un exemplu numeric.
- Capitolele 11–14 sînt de studiu individual (`selfStudy: true`): materiale mai scurte, quiz, fără seminar obligatoriu.
- Manual de bază: Huang și Petukhina (2022), *Applied Time Series Analysis and Forecasting with Python*, Springer. Companion gratuit: Hyndman și Athanasopoulos, *FPP3* (OTexts). Teorie: Brockwell și Davis, Hamilton.
- Materialele nepublicate ale cărții „Applied Time Series Solutions” (ATSSB: deck-uri Keynote, manualul de soluții) NU se pun niciodată în repository și nu se publică. Exercițiile se pot adapta doar cu acordul coautorilor. Soluțiile Brockwell–Davis nu se copiază.

## Structura cursului (EN + RO)
- 16 capitole (0–15) cu `id` stabil în `assets/course-data.js`; numerotarea în subtitlu: „Chapter N” / „Capitolul N”.
- Fișiere: `EN/Courses/chapterN_<slug>.tex`, `RO/Cursuri/capitolN_<slug>.tex`, `EN/Seminars/seminarN_<slug>.tex`, `RO/Seminarii/seminarN_<slug>_ro.tex`, cu `\input{../../latex/preamble}`; pentru RO `\def\tsalang{ro}`. Numele exacte: `python3 latex/tsa_chapters.py`.
- Româna este limba principală; EN și RO sînt identice slide cu slide. Notebook-urile sînt doar în engleză.

## Curs
- 70–90 de slide-uri: o idee pe slide (cel mult 5–6 bullets), fiecare grafic pe slide-ul lui, urmat de un slide „Interpretarea …”; exemple lucrate; recapitulare la finalul fiecărei secțiuni; glosar de acronime; bibliografie în ordine alfabetică.
- Tot textul în bullets de nivel 1–2 (fără proză liberă).
- Acronimele: la prima apariție se explică în limba lor de origine, apoi se traduc; glosarul de după pagina de titlu este generat de `python3 latex/acronyms.py N`. Acronimele noi ale capitolului N se adaugă doar în `latex/acronyms_extra/chN.py`. Scriptul trebuie să raporteze 0 acronime „NOT IN DICTIONARY”. Acronimele rămîn în forma originală și în RO (ACF, PACF, ADF, KPSS, VAR, VECM).
- Orice citare este clickabilă, cu link verificat (DOI prin Crossref, titlul trebuie să coincidă). Nu se inventează referințe.
- Imagini reale (portrete: Box, Jenkins, Engle, Granger, Sims, Johansen, Kalman…; momente istorice; locuri): doar licență liberă verificată prin API-ul Wikimedia Commons, cu credit vizibil `\imgcredit{url}{text}`, trecute în `photos/CREDITS.md`.
- `\quantlet{TSA\_chN\_<name>}{https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_NN/TSA_chN_<name>}` sub fiecare grafic.
- Secțiune finală scurtă „Contribuția posibilă a AI” / „Possible contribution of AI” (1–2 slide-uri: la ce ajută AI în tema capitolului, ce trebuie verificat). Fără secțiunea de cercetare „AI for scientific discovery” de la master.

## Limba română (vezi și brief-ul RO)
- „sînt/sîntem/sînteți”; „î” în interiorul cuvintelor („â” doar în familia „român”); „nicio/niciun”; ș și ț cu virgulă dedesubt.
- Titluri și etichete: grupuri nominale, cu literă mare doar la început („Procese stochastice și staționaritate”). Fără „Ce + verb” („Ce predați”, „Ce măsurăm”). Titluri recurente: „Noțiuni necesare azi”, „Rezultatele învățării”, „Verificări necesare”, „Idei de reținut”, „Idee de proiect”, „Contribuția posibilă a AI”, „Autoevaluare”, „Exemplu rezolvat”, „Interpretarea …”.
- Fără „vs” în textul RO („față de”, „și”, „comparat cu”); fără calcuri („bazat pe” → „pe baza”, „per” → „pe”). „mers aleator” (nu „mers aleatoriu”), „nestaționaritate”, „stochastic” (nu „stocastic”), „termenul liber” (nu „interceptul”).
- „distribuția Normală” (N mare), niciodată „Normala” substantivizat; „kurtosis” (kurtosis-ul), „excesul de kurtosis”, ca în MFM (doar SFM folosește „boltire”).
- Jargonul consacrat rămîne în engleză: drawdown, volatility clustering, backtesting, bootstrap, notebook, repository. Termenii cu traducere consacrată se traduc: „zgomot alb”, „mers aleator”, „rădăcină unitară”, „funcția de autocorelație”.
- Întrebări către sală: „Întrebare pentru sală” / „Ce credeți?” + „Răspuns”; niciodată „preziceți”. O singură întrebare pe bullet.
- Numere: virgulă zecimală; „de” după numerale ≥ 20 („21 de observații”); intervale cu „;” ([1,38; 1,45]); date „2 octombrie 2026”.
- Adresare formală (dumneavoastră): „Autentificați-vă”, nu „Te rog autentifică-te”.

## VaR / ES — convenția nivelului
- „VaR 1%”, „ES 2,5%” (RO) / „VaR 1%”, „ES 2.5%” (EN); niciodată „VaR 99%”. $\mathrm{VaR}_\alpha(X) = -q_\alpha(X)$.

## Fără remarci despre procesul de lucru
- Fără „verificat în Quantlet”, „exact ca în lucrare”, „orice noțiune este definită”.
- Fără note de durată sau de ședință („2 × 50 de minute”, „Ședința 1”); se folosesc „Partea I/II”, „Etapa 1/2”.

## Date
- Serii de piață: doar din `data/market/<SYMBOL>.csv` (lista în `data/manifest.csv`), citite prin `Quantlets/common/tsa_data.py` (local sau din `https://raw.githubusercontent.com/danpele/Time-Series-Analysis/main/data/market/<SYMBOL>.csv`). Furnizorul datelor zilnice de piață: EODHD (EOD Historical Data). Fără API cu cheie.
- Surse publice fără cheie, citite online: FRED, Eurostat, INS (TEMPO), BNR (cursul de referință), seturile de date din statsmodels (sunspots, CO2, pasageri aerieni, macrodata).
- Fără yfinance și fără pandas_datareader în codul nou; codul vechi se înlocuiește capitol cu capitol.
- Nu se afirmă nicio cifră necalculată din date sau fără sursă verificată. În materiale nu se descrie cum se încarcă datele, doar numele seriei și sursa oficială.

## Grafice
- Fundal transparent; legenda întotdeauna în afara graficului, jos (`legend_outside_bottom`); paleta din `tsa_style.py`; niciodată serii sau text în gri (griul doar pentru linii de referință, benzi de încredere, grilă); etichete în engleză.
- `Quantlets/Ch_NN/generate_all_charts.py` + `build_quantlets.py` (foldere `TSA_chN_*` cu Metainfo.txt, notebook Colab autonom, copii ale graficelor).

## Seminar
- 1 oră pe săptămînă: deck de 25–35 de slide-uri pentru un seminar de 2 ore la două săptămîni.
- Părțile A (derivări scurte), B (aplicare pe date + interpretare; fiecare problemă se încheie cu o întrebare de interpretare), C (deschisă, idee de proiect).
- Probleme „[Rezolvat]” / „[Solved]” (soluții vizibile) și „[Propus]” / „[Proposed]” (fiecare cu „Model:” spre problema rezolvată analogă; soluții doar la profesor). Slide inițial „Noțiuni necesare azi”.
- Două PDF-uri din același .tex (`\ifsolutions`): studenți (pe site) și profesor (`*_solutions.tex`, gitignored).
- Linia de final: „Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar”. Studenții nu predau nimic.

## Notebook-uri
- Doar EN: `notebooks/EN/chapterN_lecture_notebook.ipynb`, `chapterN_seminar_notebook.ipynb`, executate cu 0 erori; banner Colab „Save a copy in Drive” și celula de salvare în `MyDrive/TSA` (`notebooks/add_colab_banner.py`); seminarele separate în versiunea studenților și cea completă (`split_seminar_notebooks.py N`).

## Quiz
- `assets/quizzes/<id>.js`, înregistrat ca `window.TSA_DATA.quizzes['<id>']`: 24 de întrebări EN+RO, `draw: 20`, `correct` = index 0–3, explicațiile nu numesc litere.

## Evaluare și AI
- 70% examen scris, 20% proiect de echipă (2–4 studenți, susținere orală), 10% prezență.
- AI permis și declarat în `AI_USE.md`; susținerea orală verifică înțelegerea codului și a rezultatelor.

## Verificare finală
- Fiecare deck se compilează de două ori: 0 erori, 0 „Overfull \vbox”; randarea paginilor (pymupdf) și verificare vizuală; ștergerea fișierelor auxiliare.
- Commit-uri locale cu `git add <căi>` explicite; fără push; fără linii Co-Authored-By / Generated-with.
