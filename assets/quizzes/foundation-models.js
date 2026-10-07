// ============================================================
// Chapter 11 quiz bank: Foundation models for time series (EN + RO)
// 24 questions, 20 drawn per attempt (self-study chapter).
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['foundation-models'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Zero-shot forecasting",
                "text": "What does zero-shot forecasting with a time-series foundation model mean?",
                "options": [
                    "The model is retrained from scratch on the target series before each forecast",
                    "The pretrained model forecasts a new series from its recent history (the context) without any training on that series",
                    "The model forecasts without seeing any past value of the target series",
                    "The forecast is set to zero and the model only predicts the uncertainty band"
                ],
                "correctExplanation": "Zero-shot means no parameter is estimated on the target series: the weights learnt in pretraining stay fixed and only the context of the series is passed to the model at forecast time.",
                "incorrectExplanation": "Retraining on the target series is fine-tuning (or a classical per-series model); the model always needs the context of the series; nothing forces the forecast to zero."
            },
            "ro": {
                "title": "Prognoza zero-shot",
                "text": "Ce înseamnă o prognoză zero-shot cu un foundation model pentru serii de timp?",
                "options": [
                    "Modelul este reantrenat de la zero pe seria-țintă înaintea fiecărei prognoze",
                    "Modelul pre-antrenat prognozează o serie nouă pe baza istoricului ei recent (contextul), fără nicio antrenare pe acea serie",
                    "Modelul prognozează fără să vadă nicio valoare trecută a seriei-țintă",
                    "Prognoza este fixată la zero, iar modelul estimează doar banda de incertitudine"
                ],
                "correctExplanation": "Zero-shot înseamnă că niciun parametru nu se estimează pe seria-țintă: ponderile învățate la pre-antrenare rămîn fixe, iar modelul primește doar contextul seriei în momentul prognozei.",
                "incorrectExplanation": "Reantrenarea pe seria-țintă este fine-tuning (sau un model clasic estimat pe serie); modelul are mereu nevoie de contextul seriei; nimic nu fixează prognoza la zero."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Foundation model",
                "text": "Which description fits a foundation model best?",
                "options": [
                    "An ARIMA model whose orders are chosen automatically by AIC",
                    "A model estimated separately for each series of a data set",
                    "A rule-based expert system written by forecasters",
                    "A large model pretrained once on a broad collection of data and then used for many tasks it was not trained for specifically"
                ],
                "correctExplanation": "The term (Bommasani et al., 2021) describes models trained on broad data at scale that can be adapted to a wide range of downstream tasks; for time series, one network serves thousands of different series.",
                "incorrectExplanation": "Automatic ARIMA and per-series models are still estimated on each series; an expert system is not learnt from data at all."
            },
            "ro": {
                "title": "Foundation model",
                "text": "Care descriere se potrivește cel mai bine unui foundation model?",
                "options": [
                    "Un model ARIMA ale cărui ordine sînt alese automat după AIC",
                    "Un model estimat separat pentru fiecare serie dintr-un set de date",
                    "Un sistem expert cu reguli scrise de specialiștii în prognoză",
                    "Un model mare, pre-antrenat o singură dată pe o colecție largă de date și folosit apoi pentru multe sarcini pentru care nu a fost antrenat anume"
                ],
                "correctExplanation": "Termenul (Bommasani et al., 2021) desemnează modele antrenate pe date largi, la scară mare, care pot fi adaptate pentru multe sarcini; la seriile de timp, o singură rețea servește mii de serii diferite.",
                "incorrectExplanation": "ARIMA automat și modelele pe serie se estimează tot pe fiecare serie; un sistem expert nu este învățat din date."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Mean scaling in Chronos",
                "text": "Chronos divides the context by $s = \\frac{1}{C}\\sum_{t=1}^{C}|x_t|$ before tokenisation. Why?",
                "options": [
                    "So that series measured in different units and levels fall in the same range of values and share one vocabulary of tokens",
                    "To remove the seasonal component of the series",
                    "To make the series stationary, as a unit-root test requires",
                    "To turn the series into returns"
                ],
                "correctExplanation": "After scaling, a load series in GW and an index around 100 both have values of order one; the same fixed grid of bins can then represent any series. The forecasts are multiplied back by $s$.",
                "incorrectExplanation": "Scaling by a positive constant keeps the seasonality, the trend and any unit root; it does not compute changes or returns."
            },
            "ro": {
                "title": "Scalarea prin medie în Chronos",
                "text": "Chronos împarte contextul la $s = \\frac{1}{C}\\sum_{t=1}^{C}|x_t|$ înainte de tokenizare. De ce?",
                "options": [
                    "Pentru ca serii măsurate în unități și la niveluri diferite să ajungă în același interval de valori și să folosească același vocabular de tokeni",
                    "Pentru a elimina componenta sezonieră a seriei",
                    "Pentru a face seria staționară, așa cum cere un test de rădăcină unitară",
                    "Pentru a transforma seria în randamente"
                ],
                "correctExplanation": "După scalare, o serie de consum în GW și un indice în jur de 100 au valori de ordinul unității; aceeași grilă fixă de intervale poate reprezenta orice serie. Prognozele se înmulțesc înapoi cu $s$.",
                "incorrectExplanation": "Împărțirea la o constantă pozitivă păstrează sezonalitatea, trendul și eventuala rădăcină unitară; nu calculează variații sau randamente."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Quantisation and its limit",
                "text": "Chronos maps each scaled value to one of 4093 bin centres in $[-15, 15]$. What is a consequence of this design?",
                "options": [
                    "The model can forecast only integer values",
                    "The forecasts have no uncertainty, because each token is a single number",
                    "The model needs exactly 4093 observations of context",
                    "The model cannot forecast scaled values outside $[-15, 15]$, so a series that leaves its past range by far is clipped"
                ],
                "correctExplanation": "A token is a bin of the scaled value: values beyond $\\pm 15 s$ fall into the extreme bins, so explosive growth far beyond the context cannot be represented. Sampling many token paths gives a forecast distribution.",
                "incorrectExplanation": "Bin centres are real numbers, not integers; sampling tokens yields a distribution, not a single number; the number of bins is the vocabulary size, unrelated to the context length."
            },
            "ro": {
                "title": "Cuantizarea și limita ei",
                "text": "Chronos asociază fiecare valoare scalată unuia dintre cele 4093 de centre de interval din $[-15, 15]$. Care este o consecință a acestei construcții?",
                "options": [
                    "Modelul poate prognoza doar valori întregi",
                    "Prognozele nu au incertitudine, deoarece fiecare token este un singur număr",
                    "Modelul are nevoie de exact 4093 de observații de context",
                    "Modelul nu poate prognoza valori scalate în afara intervalului $[-15, 15]$, deci o serie care iese mult din plaja ei trecută este trunchiată"
                ],
                "correctExplanation": "Un token este un interval al valorii scalate: valorile dincolo de $\\pm 15 s$ cad în intervalele extreme, deci o creștere explozivă mult peste context nu poate fi reprezentată. Eșantionarea multor traiectorii de tokeni dă o distribuție a prognozei.",
                "incorrectExplanation": "Centrele intervalelor sînt numere reale, nu întregi; eșantionarea tokenilor dă o distribuție, nu un singur număr; numărul de intervale este mărimea vocabularului, fără legătură cu lungimea contextului."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Patching",
                "text": "A context of 2048 hourly values is cut into patches of 16 values. Why does this help a Transformer?",
                "options": [
                    "Patching removes the daily cycle from the series",
                    "Each patch is forecast by a separate ARIMA model",
                    "The model sees 128 input vectors instead of 2048, and the cost of attention grows with the square of the number of inputs",
                    "Patching turns the series into a stationary one"
                ],
                "correctExplanation": "Attention compares every input with every other one, so its cost grows with the square of the sequence length; 2048 / 16 = 128 patches make long contexts affordable, and each patch also carries local shape information (Nie et al., 2023).",
                "incorrectExplanation": "Patching is a reshaping of the context: it does not remove cycles, does not fit ARIMA models and does not change the stationarity of the series."
            },
            "ro": {
                "title": "Patching",
                "text": "Un context de 2048 de valori orare este împărțit în patch-uri de cîte 16 valori. De ce ajută acest lucru un Transformer?",
                "options": [
                    "Patching-ul elimină ciclul zilnic din serie",
                    "Fiecare patch este prognozat de un model ARIMA separat",
                    "Modelul vede 128 de vectori de intrare în loc de 2048, iar costul atenției crește cu pătratul numărului de intrări",
                    "Patching-ul face seria staționară"
                ],
                "correctExplanation": "Atenția compară fiecare intrare cu toate celelalte, deci costul crește cu pătratul lungimii secvenței; 2048 / 16 = 128 de patch-uri fac contextele lungi accesibile, iar fiecare patch păstrează și forma locală a seriei (Nie et al., 2023).",
                "incorrectExplanation": "Patching-ul doar rearanjează contextul: nu elimină ciclurile, nu estimează modele ARIMA și nu schimbă staționaritatea seriei."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Attention weights",
                "text": "One query has scores $2$, $1$, $0$ against three keys (already divided by $\\sqrt{d_k}$). Which statement is correct?",
                "options": [
                    "The weights are $e^2, e^1, e^0$ divided by their sum, about 0.67, 0.24 and 0.09",
                    "The weights are 2/3, 1/3 and 0",
                    "All three keys get the same weight 1/3",
                    "The weights are 2, 1 and 0"
                ],
                "correctExplanation": "The softmax exponentiates the scores and divides by the sum: $e^2/(e^2+e+1) \\approx 0.67$, $e/(e^2+e+1) \\approx 0.24$, $1/(e^2+e+1) \\approx 0.09$; the output is the weighted average of the three values.",
                "incorrectExplanation": "Proportional weights 2/3 and 1/3 ignore the exponential; equal weights ignore the scores; raw scores do not sum to one."
            },
            "ro": {
                "title": "Ponderile atenției",
                "text": "O interogare are scorurile $2$, $1$, $0$ față de trei chei (deja împărțite la $\\sqrt{d_k}$). Care afirmație este corectă?",
                "options": [
                    "Ponderile sînt $e^2, e^1, e^0$ împărțite la suma lor, aproximativ 0,67; 0,24 și 0,09",
                    "Ponderile sînt 2/3, 1/3 și 0",
                    "Toate cele trei chei primesc aceeași pondere, 1/3",
                    "Ponderile sînt 2, 1 și 0"
                ],
                "correctExplanation": "Softmax-ul ridică la exponențială scorurile și împarte la sumă: $e^2/(e^2+e+1) \\approx 0{,}67$, $e/(e^2+e+1) \\approx 0{,}24$, $1/(e^2+e+1) \\approx 0{,}09$; ieșirea este media ponderată a celor trei valori.",
                "incorrectExplanation": "Ponderile proporționale 2/3 și 1/3 ignoră exponențiala; ponderile egale ignoră scorurile; scorurile brute nu au suma 1."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Quantile output of Chronos-Bolt",
                "text": "Chronos-Bolt returns the quantiles at levels 0.1, 0.2, ..., 0.9. A risk manager asks it for the 1% quantile. What is the problem?",
                "options": [
                    "The 1% level is outside the grid: any value reported for it is an extrapolation, not an output of the model",
                    "There is no problem: the model returns every quantile exactly",
                    "The 1% quantile equals the median for this model",
                    "The model returns only point forecasts"
                ],
                "correctExplanation": "A direct quantile head gives only the levels it was trained on; below 10% the tail must be extrapolated, and the result depends on an assumption added by the user.",
                "incorrectExplanation": "The output grid is fixed at nine levels; the median is the 50% level; the model returns quantiles, not only points."
            },
            "ro": {
                "title": "Cuantilele Chronos-Bolt",
                "text": "Chronos-Bolt întoarce cuantilele la nivelurile 0,1; 0,2; ...; 0,9. Un analist de risc îi cere cuantila de 1%. Care este problema?",
                "options": [
                    "Nivelul de 1% este în afara grilei: orice valoare raportată pentru el este o extrapolare, nu un rezultat al modelului",
                    "Nu există nicio problemă: modelul întoarce exact orice cuantilă",
                    "Pentru acest model, cuantila de 1% este egală cu mediana",
                    "Modelul întoarce doar prognoze punctuale"
                ],
                "correctExplanation": "Un strat de ieșire cu cuantile directe dă doar nivelurile pe care a fost antrenat; sub 10% coada trebuie extrapolată, iar rezultatul depinde de o ipoteză adăugată de utilizator.",
                "incorrectExplanation": "Grila de ieșire are nouă niveluri fixe; mediana este nivelul de 50%; modelul întoarce cuantile, nu doar puncte."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Pinball loss above the quantile",
                "text": "The 0.9 quantile forecast is $q = 8$ and the outcome is $y = 10$. What is the pinball loss?",
                "options": [
                    "$0.1 \\cdot (10 - 8) = 0.2$",
                    "$(10 - 8)^2 = 4$",
                    "$|10 - 8| = 2$",
                    "$0.9 \\cdot (10 - 8) = 1.8$"
                ],
                "correctExplanation": "When $y \\ge q$ the loss is $\\tau (y - q) = 0.9 \\cdot 2 = 1.8$: an outcome above a high quantile is penalised heavily, because the 0.9 quantile should be exceeded only 10% of the time.",
                "incorrectExplanation": "The weight $1 - \\tau = 0.1$ applies when the outcome is below the quantile; the squared and the absolute error are not quantile losses."
            },
            "ro": {
                "title": "Pierderea pinball deasupra cuantilei",
                "text": "Prognoza cuantilei de 0,9 este $q = 8$, iar valoarea observată este $y = 10$. Cît este pierderea pinball?",
                "options": [
                    "$0{,}1 \\cdot (10 - 8) = 0{,}2$",
                    "$(10 - 8)^2 = 4$",
                    "$|10 - 8| = 2$",
                    "$0{,}9 \\cdot (10 - 8) = 1{,}8$"
                ],
                "correctExplanation": "Cînd $y \\ge q$, pierderea este $\\tau (y - q) = 0{,}9 \\cdot 2 = 1{,}8$: o valoare peste o cuantilă mare este penalizată puternic, deoarece cuantila de 0,9 ar trebui depășită doar în 10% din cazuri.",
                "incorrectExplanation": "Ponderea $1 - \\tau = 0{,}1$ se aplică atunci cînd valoarea observată este sub cuantilă; eroarea pătratică și cea absolută nu sînt pierderi cuantile."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Pinball loss below the quantile",
                "text": "The 0.9 quantile forecast is $q = 10$ and the outcome is $y = 8$. What is the pinball loss?",
                "options": [
                    "$(1 - 0.9)(10 - 8) = 0.2$",
                    "$0.9 \\cdot (10 - 8) = 1.8$",
                    "$0$, because the outcome is inside the forecast range",
                    "$-1.8$"
                ],
                "correctExplanation": "When $y < q$ the loss is $(1 - \\tau)(q - y) = 0.1 \\cdot 2 = 0.2$: for a high quantile, outcomes below it are expected and cost little.",
                "incorrectExplanation": "The weight 0.9 applies only above the quantile; the loss is zero only if $y = q$; a loss is never negative."
            },
            "ro": {
                "title": "Pierderea pinball sub cuantilă",
                "text": "Prognoza cuantilei de 0,9 este $q = 10$, iar valoarea observată este $y = 8$. Cît este pierderea pinball?",
                "options": [
                    "$(1 - 0{,}9)(10 - 8) = 0{,}2$",
                    "$0{,}9 \\cdot (10 - 8) = 1{,}8$",
                    "$0$, deoarece valoarea observată este în intervalul prognozat",
                    "$-1{,}8$"
                ],
                "correctExplanation": "Cînd $y < q$, pierderea este $(1 - \\tau)(q - y) = 0{,}1 \\cdot 2 = 0{,}2$: pentru o cuantilă mare, valorile de sub ea sînt așteptate și costă puțin.",
                "incorrectExplanation": "Ponderea 0,9 se aplică doar deasupra cuantilei; pierderea este zero doar dacă $y = q$; o pierdere nu este niciodată negativă."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "CRPS and the absolute error",
                "text": "What does the CRPS reduce to when the forecast is a single point $\\hat y$ (a degenerate distribution)?",
                "options": [
                    "The squared error $(y - \\hat y)^2$",
                    "The log-likelihood of $y$",
                    "Zero, whatever the outcome",
                    "The absolute error $|y - \\hat y|$"
                ],
                "correctExplanation": "The CRPS integrates $(F(z) - \\mathbf{1}\\{z \\ge y\\})^2$ over $z$; for a step function at $\\hat y$ the integrand equals 1 between $\\hat y$ and $y$ and 0 elsewhere, so the CRPS is $|y - \\hat y|$. It generalises the MAE to distributions.",
                "incorrectExplanation": "The squared error corresponds to another scoring rule; the log score uses the density, which a point forecast does not have; the CRPS is zero only if $\\hat y = y$."
            },
            "ro": {
                "title": "CRPS și eroarea absolută",
                "text": "La ce se reduce CRPS cînd prognoza este un singur punct $\\hat y$ (o distribuție degenerată)?",
                "options": [
                    "La eroarea pătratică $(y - \\hat y)^2$",
                    "La logaritmul verosimilității lui $y$",
                    "La zero, oricare ar fi valoarea observată",
                    "La eroarea absolută $|y - \\hat y|$"
                ],
                "correctExplanation": "CRPS integrează $(F(z) - \\mathbf{1}\\{z \\ge y\\})^2$ după $z$; pentru o funcție treaptă în $\\hat y$, integrandul este 1 între $\\hat y$ și $y$ și 0 în rest, deci CRPS este $|y - \\hat y|$. CRPS generalizează MAE la distribuții.",
                "incorrectExplanation": "Eroarea pătratică corespunde altei reguli de scor; scorul logaritmic folosește densitatea, pe care o prognoză punctuală nu o are; CRPS este zero doar dacă $\\hat y = y$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Reading MASE",
                "text": "A model has MASE = 0.8 on a monthly series with $m = 12$. What does this mean?",
                "options": [
                    "Its forecasts are 80% correct",
                    "Its mean absolute error is 80% of the in-sample mean absolute error of the seasonal naive forecast",
                    "Its error is 0.8 units of the series",
                    "It explains 80% of the variance of the series"
                ],
                "correctExplanation": "MASE divides the MAE of the forecast by the in-sample MAE of the seasonal naive forecast $y_{t-m}$ (Hyndman and Koehler, 2006); below 1 means smaller errors than that benchmark, whatever the units.",
                "incorrectExplanation": "MASE is a ratio of errors, not a hit rate, not an error in the units of the series and not an $R^2$."
            },
            "ro": {
                "title": "Interpretarea MASE",
                "text": "Un model are MASE = 0,8 pe o serie lunară, cu $m = 12$. Ce înseamnă acest lucru?",
                "options": [
                    "Prognozele lui sînt corecte în 80% din cazuri",
                    "Eroarea lui absolută medie este 80% din eroarea absolută medie, în eșantion, a prognozei sezoniere naive",
                    "Eroarea lui este de 0,8 unități ale seriei",
                    "Explică 80% din varianța seriei"
                ],
                "correctExplanation": "MASE împarte MAE al prognozei la MAE în eșantion al prognozei sezoniere naive $y_{t-m}$ (Hyndman și Koehler, 2006); sub 1 înseamnă erori mai mici decît acest reper, indiferent de unități.",
                "incorrectExplanation": "MASE este un raport de erori, nu o rată de reușită, nu o eroare în unitățile seriei și nu un $R^2$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Coverage of an interval",
                "text": "Over 100 forecasts, the outcome falls inside the central 80% interval of a model 61 times. What do you conclude?",
                "options": [
                    "The intervals are too wide: the model is too cautious",
                    "The model is well calibrated, since 61% is close to the median",
                    "The intervals are too narrow: the model is overconfident",
                    "Coverage says nothing about the intervals"
                ],
                "correctExplanation": "A calibrated 80% interval should contain about 80% of the outcomes; 61% means that the outcomes fall outside too often, so the forecast distribution is too narrow.",
                "incorrectExplanation": "Too wide intervals would cover more than 80%; calibration compares the coverage with the nominal 80%, not with 50%; coverage is exactly the check of the intervals."
            },
            "ro": {
                "title": "Acoperirea unui interval",
                "text": "Din 100 de prognoze, valoarea observată cade de 61 de ori în intervalul central de 80% al unui model. Ce concluzionați?",
                "options": [
                    "Intervalele sînt prea largi: modelul este prea prudent",
                    "Modelul este bine calibrat, deoarece 61% este aproape de mediană",
                    "Intervalele sînt prea înguste: modelul este prea încrezător",
                    "Acoperirea nu spune nimic despre intervale"
                ],
                "correctExplanation": "Un interval de 80% calibrat ar trebui să conțină circa 80% din valorile observate; 61% înseamnă că valorile ies prea des din interval, deci distribuția prognozei este prea îngustă.",
                "incorrectExplanation": "Intervalele prea largi ar acoperi mai mult de 80%; calibrarea compară acoperirea cu nivelul nominal de 80%, nu cu 50%; acoperirea este tocmai verificarea intervalelor."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "A fair comparison",
                "text": "Which design gives a fair comparison between a foundation model and ETS?",
                "options": [
                    "Both forecast from the same origins, with the same data up to each origin and the same horizon, scored by the same metrics against the seasonal naive",
                    "The foundation model is scored on 2025 and ETS on 2015, each on its best year",
                    "ETS is tuned on the test period, the foundation model is not",
                    "Only the foundation model is compared with the seasonal naive"
                ],
                "correctExplanation": "A rolling-origin evaluation keeps the information set, the horizon and the scoring identical for all models; the seasonal naive gives a common scale (Chapter 4).",
                "incorrectExplanation": "Different test periods, tuning on the test data or a benchmark used for one model only all bias the comparison."
            },
            "ro": {
                "title": "O comparație corectă",
                "text": "Care schemă de evaluare dă o comparație corectă între un foundation model și ETS?",
                "options": [
                    "Ambele prognozează din aceleași origini, cu aceleași date pînă la fiecare origine și pe același orizont, evaluate cu aceleași măsuri, față de prognoza sezonieră naivă",
                    "Foundation model-ul este evaluat pe 2025, iar ETS pe 2015, fiecare pe anul lui cel mai bun",
                    "ETS este calibrat pe perioada de test, foundation model-ul nu",
                    "Doar foundation model-ul este comparat cu prognoza sezonieră naivă"
                ],
                "correctExplanation": "Evaluarea cu origine mobilă păstrează identice informația disponibilă, orizontul și măsurile de eroare pentru toate modelele; prognoza sezonieră naivă dă o scală comună (Capitolul 4).",
                "incorrectExplanation": "Perioadele de test diferite, calibrarea pe datele de test sau un reper folosit doar pentru un model distorsionează comparația."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Benchmark contamination",
                "text": "A foundation model scores very well on a public benchmark that was released years before the model. What is the main concern?",
                "options": [
                    "Old benchmarks are always too easy for every model",
                    "The benchmark series may be in the pretraining corpus, so the test is not out of sample",
                    "Public data cannot be used in research",
                    "The model was trained with too few parameters"
                ],
                "correctExplanation": "If the test series (or highly correlated ones) were seen during pretraining, the score measures memory, not forecasting; tests on data published after the release of the weights avoid this.",
                "incorrectExplanation": "Old benchmarks are not easy by construction (the M3 and M4 series remain hard); public data are standard; the number of parameters does not explain the issue."
            },
            "ro": {
                "title": "Contaminarea benchmark-urilor",
                "text": "Un foundation model obține un scor foarte bun pe un benchmark public lansat cu ani înaintea modelului. Care este principala îngrijorare?",
                "options": [
                    "Benchmark-urile vechi sînt întotdeauna prea ușoare pentru orice model",
                    "Seriile benchmark-ului pot fi în corpusul de pre-antrenare, deci testul nu este în afara eșantionului",
                    "Datele publice nu pot fi folosite în cercetare",
                    "Modelul a fost antrenat cu prea puțini parametri"
                ],
                "correctExplanation": "Dacă seriile de test (sau serii puternic corelate cu ele) au fost văzute la pre-antrenare, scorul măsoară memoria, nu prognoza; testele pe date publicate după lansarea ponderilor evită această problemă.",
                "incorrectExplanation": "Benchmark-urile vechi nu sînt ușoare prin construcție (seriile M3 și M4 rămîn dificile); datele publice sînt uzuale; numărul de parametri nu explică problema."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Leakage in your own pipeline",
                "text": "Which step leaks information from the test period into the forecasts?",
                "options": [
                    "Scaling each context by the mean of its own values",
                    "Using the same horizon for all models",
                    "Scaling every context by the mean of the whole sample, test period included",
                    "Forecasting from origins that move forward one month at a time"
                ],
                "correctExplanation": "A full-sample mean contains future values; every transformation must use only the data available at the forecast origin (Hewamalage et al., 2023).",
                "incorrectExplanation": "Scaling by the context only, a common horizon and a rolling origin are all correct practice."
            },
            "ro": {
                "title": "Leakage-ul în propriul cod",
                "text": "Care pas introduce în prognoze informație din perioada de test?",
                "options": [
                    "Scalarea fiecărui context cu media propriilor valori",
                    "Folosirea aceluiași orizont pentru toate modelele",
                    "Scalarea fiecărui context cu media întregului eșantion, inclusiv perioada de test",
                    "Prognoza din origini care avansează cîte o lună"
                ],
                "correctExplanation": "O medie pe tot eșantionul conține valori viitoare; orice transformare trebuie să folosească doar datele disponibile la originea prognozei (Hewamalage et al., 2023).",
                "incorrectExplanation": "Scalarea doar cu contextul, un orizont comun și originea mobilă sînt practici corecte."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Exchange rates",
                "text": "A foundation model is used to forecast the daily EUR/RON rate 20 days ahead. What should you expect?",
                "options": [
                    "It beats the random walk easily, because it has seen many exchange rates",
                    "It forecasts a strong weekly seasonality",
                    "It forecasts a steady return to the 2010 level",
                    "Its point forecasts are close to the last observed value and do not beat the random walk clearly"
                ],
                "correctExplanation": "Daily exchange rates are close to a random walk: the best point forecast is the last value, and no model can extract much more from the past alone; at best the foundation model reproduces the random walk.",
                "incorrectExplanation": "Having seen many exchange rates does not make the future predictable; there is no weekly seasonality in a daily exchange rate to exploit; mean reversion to an old level has no support."
            },
            "ro": {
                "title": "Cursurile de schimb",
                "text": "Un foundation model este folosit pentru a prognoza cursul zilnic EUR/RON pe un orizont de 20 de zile. La ce vă așteptați?",
                "options": [
                    "Bate ușor mersul aleator, deoarece a văzut multe cursuri de schimb",
                    "Prognozează o sezonalitate săptămînală puternică",
                    "Prognozează o revenire constantă la nivelul din 2010",
                    "Prognozele lui punctuale sînt apropiate de ultima valoare observată și nu bat clar mersul aleator"
                ],
                "correctExplanation": "Cursurile de schimb zilnice sînt apropiate de un mers aleator: cea mai bună prognoză punctuală este ultima valoare, iar niciun model nu poate extrage mult mai mult doar din trecut; în cel mai bun caz, foundation model-ul reproduce mersul aleator.",
                "incorrectExplanation": "Faptul că a văzut multe cursuri nu face viitorul previzibil; un curs zilnic nu are sezonalitate săptămînală de exploatat; revenirea la un nivel vechi nu are niciun fundament."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Context length",
                "text": "For hourly electricity load with a weekly cycle, why can a context of 96 hours be too short?",
                "options": [
                    "A Transformer cannot read more than 96 values",
                    "It does not contain a full week (168 hours), so the model cannot see the weekly pattern it has to repeat",
                    "A short context always makes the forecast unbiased",
                    "Load has no daily cycle, so the context does not matter"
                ],
                "correctExplanation": "A zero-shot model can only reproduce patterns present in the context: with 96 hours the weekend effect is missing; at least one or two full weeks are needed.",
                "incorrectExplanation": "Chronos-Bolt accepts up to 2048 values; a short context does not guarantee unbiasedness; load has a strong daily cycle as well."
            },
            "ro": {
                "title": "Lungimea contextului",
                "text": "Pentru consumul orar de energie electrică, care are un ciclu săptămînal, de ce poate fi prea scurt un context de 96 de ore?",
                "options": [
                    "Un Transformer nu poate citi mai mult de 96 de valori",
                    "Nu conține o săptămînă întreagă (168 de ore), deci modelul nu poate vedea tiparul săptămînal pe care trebuie să-l repete",
                    "Un context scurt face întotdeauna prognoza nedistorsionată",
                    "Consumul nu are ciclu zilnic, deci contextul nu contează"
                ],
                "correctExplanation": "Un model zero-shot poate reproduce doar tiparele prezente în context: cu 96 de ore lipsește efectul de weekend; sînt necesare cel puțin una sau două săptămîni întregi.",
                "incorrectExplanation": "Chronos-Bolt acceptă pînă la 2048 de valori; un context scurt nu garantează lipsa distorsiunii; consumul are și un ciclu zilnic puternic."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "TimeGPT",
                "text": "Why is TimeGPT, available only through a paid API, a problem for a reproducible course project?",
                "options": [
                    "It cannot forecast more than one step ahead",
                    "It is a statistical model, not a neural network",
                    "Its weights are closed and the service can change, so results cannot be reproduced or inspected, and the data leave your computer",
                    "It works only for financial data"
                ],
                "correctExplanation": "With a closed API one cannot fix the model version, inspect the training data or rerun the experiment later at no cost; open-weight models such as Chronos or TimesFM run locally (Garza et al., 2023, describe TimeGPT).",
                "incorrectExplanation": "TimeGPT produces multi-step forecasts, is a Transformer, and is marketed for many domains."
            },
            "ro": {
                "title": "TimeGPT",
                "text": "De ce este TimeGPT, disponibil doar printr-un API cu plată, o problemă pentru un proiect reproductibil?",
                "options": [
                    "Nu poate prognoza mai mult de un pas înainte",
                    "Este un model statistic, nu o rețea neuronală",
                    "Ponderile sînt închise și serviciul se poate schimba, deci rezultatele nu pot fi reproduse sau inspectate, iar datele pleacă de pe calculatorul dumneavoastră",
                    "Funcționează doar pentru date financiare"
                ],
                "correctExplanation": "Cu un API închis nu puteți fixa versiunea modelului, nu puteți inspecta datele de antrenare și nu puteți reface gratuit experimentul mai tîrziu; modelele cu ponderi deschise, precum Chronos sau TimesFM, rulează local (Garza et al., 2023, descriu TimeGPT).",
                "incorrectExplanation": "TimeGPT dă prognoze pe mai mulți pași, este un Transformer și este promovat pentru multe domenii."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Moirai",
                "text": "Which feature distinguishes Moirai (Woo et al., 2024)?",
                "options": [
                    "It is a closed model available only through an API",
                    "It forecasts only daily series",
                    "It handles any number of variables and several patch sizes by frequency, with a mixture distribution as output",
                    "It is a linear regression on lagged values"
                ],
                "correctExplanation": "Moirai flattens multivariate series into one sequence (any-variate attention), chooses the patch size by frequency and outputs a mixture of distributions; its weights are open under a non-commercial licence.",
                "incorrectExplanation": "Moirai has open weights, works across frequencies, and is a Transformer, not a linear regression."
            },
            "ro": {
                "title": "Moirai",
                "text": "Ce caracteristică distinge Moirai (Woo et al., 2024)?",
                "options": [
                    "Este un model închis, disponibil doar printr-un API",
                    "Prognozează doar serii zilnice",
                    "Acceptă orice număr de variabile și mai multe mărimi de patch după frecvență, iar ieșirea este un amestec de distribuții",
                    "Este o regresie liniară pe valorile seriei la anumite laguri"
                ],
                "correctExplanation": "Moirai aranjează seriile multivariate într-o singură secvență (atenție pentru orice număr de variabile), alege mărimea patch-ului după frecvență și dă la ieșire un amestec de distribuții; ponderile sînt deschise, cu o licență necomercială.",
                "incorrectExplanation": "Moirai are ponderi deschise, funcționează pentru mai multe frecvențe și este un Transformer, nu o regresie liniară."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Lag-Llama",
                "text": "What is the output of Lag-Llama (Rasul et al., 2023) for each future step?",
                "options": [
                    "The parameters of a Student-$t$ distribution, from which forecast samples are drawn",
                    "A single point forecast without uncertainty",
                    "A token from a vocabulary of 4093 bins",
                    "The ARIMA orders of the series"
                ],
                "correctExplanation": "Lag-Llama is a decoder-only Transformer that uses lagged values as inputs and outputs the degrees of freedom, location and scale of a Student-$t$ distribution; sampling gives forecast paths.",
                "incorrectExplanation": "It is probabilistic, it does not quantise values into bins (that is Chronos), and it does not select ARIMA orders."
            },
            "ro": {
                "title": "Lag-Llama",
                "text": "Ce dă la ieșire Lag-Llama (Rasul et al., 2023) pentru fiecare pas viitor?",
                "options": [
                    "Parametrii unei distribuții Student-$t$, din care se extrag eșantioane de prognoză",
                    "O singură prognoză punctuală, fără incertitudine",
                    "Un token dintr-un vocabular de 4093 de intervale",
                    "Ordinele ARIMA ale seriei"
                ],
                "correctExplanation": "Lag-Llama este un Transformer doar cu decodor care folosește ca intrări valorile seriei la anumite laguri și dă gradele de libertate, poziția și scala unei distribuții Student-$t$; eșantionarea dă traiectorii de prognoză.",
                "incorrectExplanation": "Este probabilist, nu cuantizează valorile în intervale (aceasta face Chronos) și nu alege ordine ARIMA."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Lessons of the M competitions",
                "text": "Which lesson of the M3 and M4 forecasting competitions is most relevant for foundation models?",
                "options": [
                    "Neural networks won every category of M3",
                    "Statistically sophisticated methods do not necessarily beat simple ones, and combinations of simple methods are hard to beat",
                    "Seasonal naive forecasts are always the best",
                    "Only the point forecast matters"
                ],
                "correctExplanation": "Makridakis and Hibon (2000) found that complex methods did not beat simple ones on average, and M4 (Makridakis et al., 2020) confirmed the strength of combinations; a new model must therefore beat simple benchmarks first.",
                "incorrectExplanation": "Neural networks did not win M3; seasonal naive is a benchmark, not always the best; M4 also scored prediction intervals."
            },
            "ro": {
                "title": "Lecțiile competițiilor M",
                "text": "Care lecție a competițiilor de prognoză M3 și M4 este cea mai relevantă pentru foundation models?",
                "options": [
                    "Rețelele neuronale au cîștigat toate categoriile din M3",
                    "Metodele statistice sofisticate nu bat neapărat metodele simple, iar combinațiile de metode simple sînt greu de învins",
                    "Prognozele sezoniere naive sînt întotdeauna cele mai bune",
                    "Contează doar prognoza punctuală"
                ],
                "correctExplanation": "Makridakis și Hibon (2000) au arătat că metodele complexe nu le bat în medie pe cele simple, iar M4 (Makridakis et al., 2020) a confirmat forța combinațiilor; un model nou trebuie deci să bată întîi reperele simple.",
                "incorrectExplanation": "Rețelele neuronale nu au cîștigat M3; prognoza sezonieră naivă este un reper, nu întotdeauna cea mai bună; M4 a evaluat și intervalele de prognoză."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Language models as forecasters",
                "text": "What did Tan et al. (2024) find about forecasters built on large language models?",
                "options": [
                    "Larger language models always forecast better",
                    "Language models beat every statistical model on every data set",
                    "Text pretraining is necessary for time-series forecasting",
                    "Removing the language-model component, or replacing it with a simple attention layer, did not degrade accuracy and often improved it"
                ],
                "correctExplanation": "Their ablations show that the language-model part adds cost but not accuracy; models pretrained on time series (Chronos, TimesFM) are a different design from reusing a text model.",
                "incorrectExplanation": "The study found no gain from the language model, so it supports none of the other statements."
            },
            "ro": {
                "title": "Modelele de limbaj ca modele de prognoză",
                "text": "Ce au arătat Tan et al. (2024) despre modelele de prognoză construite pe modele mari de limbaj?",
                "options": [
                    "Modelele de limbaj mai mari prognozează întotdeauna mai bine",
                    "Modelele de limbaj bat orice model statistic pe orice set de date",
                    "Pre-antrenarea pe text este necesară pentru prognoza seriilor de timp",
                    "Eliminarea componentei de model de limbaj, sau înlocuirea ei cu un strat simplu de atenție, nu a redus precizia și adesea a îmbunătățit-o"
                ],
                "correctExplanation": "Ablațiile lor arată că partea de model de limbaj adaugă cost, dar nu precizie; modelele pre-antrenate pe serii de timp (Chronos, TimesFM) sînt o construcție diferită de reutilizarea unui model de text.",
                "incorrectExplanation": "Studiul nu a găsit niciun cîștig din modelul de limbaj, deci nu susține niciuna dintre celelalte afirmații."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "When a foundation model is a good choice",
                "text": "In which situation is a zero-shot foundation model most useful?",
                "options": [
                    "One long financial return series must be forecast one day ahead",
                    "Thousands of short, seasonal series must be forecast quickly, with no time to build a model for each",
                    "A causal effect of a policy change must be estimated",
                    "The forecast must be explained parameter by parameter to a regulator"
                ],
                "correctExplanation": "A pretrained model needs no estimation per series, so it scales to many series and borrows patterns learnt elsewhere; it is weakest when the series is unpredictable (returns), when causality is asked, or when interpretability is required.",
                "incorrectExplanation": "Returns are close to unpredictable, causal questions need a design (Chapter 6), and a network with millions of parameters cannot be explained parameter by parameter."
            },
            "ro": {
                "title": "Cînd este util un foundation model",
                "text": "În ce situație este cel mai util un foundation model zero-shot?",
                "options": [
                    "Trebuie prognozată cu o zi înainte o singură serie lungă de randamente financiare",
                    "Trebuie prognozate rapid mii de serii scurte și sezoniere, fără timp pentru a construi un model pentru fiecare",
                    "Trebuie estimat efectul cauzal al unei schimbări de politică",
                    "Prognoza trebuie explicată unui regulator parametru cu parametru"
                ],
                "correctExplanation": "Un model pre-antrenat nu cere estimare pe fiecare serie, deci se poate aplica pe multe serii și preia tipare învățate în altă parte; este cel mai slab cînd seria este imprevizibilă (randamentele), cînd se cere cauzalitate sau cînd este necesară interpretabilitatea.",
                "incorrectExplanation": "Randamentele sînt aproape imprevizibile, întrebările cauzale cer o schemă de identificare (Capitolul 6), iar o rețea cu milioane de parametri nu poate fi explicată parametru cu parametru."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Geometric mean of relative scores",
                "text": "Why are relative scores (model error divided by seasonal naive error) averaged over series with a geometric mean?",
                "options": [
                    "The geometric mean is always larger, which favours new models",
                    "The arithmetic mean cannot be computed for ratios",
                    "Ratios of 2 and 0.5 should cancel out; the geometric mean gives 1, the arithmetic mean gives 1.25",
                    "The geometric mean removes the need for a benchmark"
                ],
                "correctExplanation": "Relative errors are multiplicative: being twice as bad on one series and twice as good on another is neutral, and only the geometric mean $\\sqrt{2 \\cdot 0.5} = 1$ says so.",
                "incorrectExplanation": "The geometric mean is never larger than the arithmetic mean; both can be computed; the ratios still need the benchmark in the denominator."
            },
            "ro": {
                "title": "Media geometrică a scorurilor relative",
                "text": "De ce se mediază scorurile relative (eroarea modelului împărțită la eroarea prognozei sezoniere naive) pe serii cu o medie geometrică?",
                "options": [
                    "Media geometrică este întotdeauna mai mare, ceea ce favorizează modelele noi",
                    "Media aritmetică nu poate fi calculată pentru rapoarte",
                    "Rapoartele 2 și 0,5 ar trebui să se compenseze; media geometrică dă 1, media aritmetică dă 1,25",
                    "Media geometrică elimină nevoia unui reper"
                ],
                "correctExplanation": "Erorile relative sînt multiplicative: a fi de două ori mai slab pe o serie și de două ori mai bun pe alta este neutru, iar doar media geometrică $\\sqrt{2 \\cdot 0{,}5} = 1$ arată acest lucru.",
                "incorrectExplanation": "Media geometrică nu este niciodată mai mare decît media aritmetică; ambele pot fi calculate; rapoartele au în continuare nevoie de reper la numitor."
            }
        }
    ]
};
