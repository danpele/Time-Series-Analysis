// ============================================================
// Chapter 4 quiz bank: Seasonality and forecasting: SARIMA, TBATS, Prophet (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['seasonal'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 2,
            "en": {
                "title": "The airline model",
                "text": "Which specification is the “airline model” of Box and Jenkins for monthly data?",
                "options": [
                    "ARIMA$(1,1,1)$",
                    "SARIMA$(1,0,0)(1,0,0)_{12}$",
                    "SARIMA$(0,1,1)(0,1,1)_{12}$",
                    "SARIMA$(2,1,0)(0,1,0)_{12}$"
                ],
                "correctExplanation": "One regular and one seasonal difference, with one regular and one seasonal MA term: $\\Delta\\Delta_{12}y_t = (1 + \\theta L)(1 + \\Theta L^{12})\\varepsilon_t$, two parameters.",
                "incorrectExplanation": "A plain ARIMA has no seasonal part; a pure seasonal AR without differences cannot follow trending seasonal data; and a model without the seasonal MA term leaves a spike at lag 12 in the residuals."
            },
            "ro": {
                "title": "Modelul airline",
                "text": "Ce specificație este „modelul airline” al lui Box și Jenkins pentru date lunare?",
                "options": [
                    "ARIMA$(1,1,1)$",
                    "SARIMA$(1,0,0)(1,0,0)_{12}$",
                    "SARIMA$(0,1,1)(0,1,1)_{12}$",
                    "SARIMA$(2,1,0)(0,1,0)_{12}$"
                ],
                "correctExplanation": "O diferență obișnuită și una sezonieră, cu un termen MA obișnuit și unul sezonier: $\\Delta\\Delta_{12}y_t = (1 + \\theta L)(1 + \\Theta L^{12})\\varepsilon_t$, doi parametri.",
                "incorrectExplanation": "Un ARIMA simplu nu are parte sezonieră; un AR sezonier pur, fără diferențe, nu poate urmări date sezoniere cu trend; iar un model fără termenul MA sezonier lasă o valoare semnificativă la decalajul 12 în reziduuri."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Free parameters",
                "text": "How many parameters, including $\\sigma^2$, does SARIMA$(0,1,1)(0,1,1)_4$ have?",
                "options": [
                    "5: MA coefficients at lags 1, 4 and 5, a constant and $\\sigma^2$",
                    "4: $\\theta$, $\\Theta$, $\\theta\\Theta$ and $\\sigma^2$",
                    "6: one for each lag from 1 to 5 and $\\sigma^2$",
                    "3: $\\theta$, $\\Theta$ and $\\sigma^2$"
                ],
                "correctExplanation": "The MA polynomial $(1 + \\theta L)(1 + \\Theta L^4)$ has a coefficient $\\theta\\Theta$ at lag 5, but it is the product of the two parameters, not a new one; with $d = D = 1$ there is no constant.",
                "incorrectExplanation": "The lag-5 coefficient is determined by $\\theta$ and $\\Theta$; a constant is dropped when $d + D \\ge 2$; and the MA polynomial has no free coefficients at lags 2 and 3."
            },
            "ro": {
                "title": "Parametri liberi",
                "text": "Cîți parametri, inclusiv $\\sigma^2$, are SARIMA$(0,1,1)(0,1,1)_4$?",
                "options": [
                    "5: coeficienții MA la decalajele 1, 4 și 5, o constantă și $\\sigma^2$",
                    "4: $\\theta$, $\\Theta$, $\\theta\\Theta$ și $\\sigma^2$",
                    "6: cîte unul pentru fiecare decalaj de la 1 la 5 și $\\sigma^2$",
                    "3: $\\theta$, $\\Theta$ și $\\sigma^2$"
                ],
                "correctExplanation": "Polinomul MA $(1 + \\theta L)(1 + \\Theta L^4)$ are coeficientul $\\theta\\Theta$ la decalajul 5, dar acesta este produsul celor doi parametri, nu unul nou; cu $d = D = 1$ nu există constantă.",
                "incorrectExplanation": "Coeficientul de la decalajul 5 este determinat de $\\theta$ și $\\Theta$; constanta lipsește cînd $d + D \\ge 2$; iar polinomul MA nu are coeficienți liberi la decalajele 2 și 3."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Satellites in the ACF",
                "text": "The ACF of $\\Delta\\Delta_{12}y_t$ has spikes at lags 1 and 12 and two small spikes at lags 11 and 13. What do the small spikes show?",
                "options": [
                    "The product $\\rho_1\\rho_{12}$ of the multiplicative airline model",
                    "A second seasonal period of 11 months",
                    "That the series needs a third difference",
                    "Sampling noise that should be ignored in every case"
                ],
                "correctExplanation": "For $(1 + \\theta L)(1 + \\Theta L^{12})\\varepsilon_t$, $\\rho_{11} = \\rho_{13} = \\rho_1\\rho_{12}$: the “satellites” are the fingerprint of the multiplicative structure.",
                "incorrectExplanation": "No 11-month cycle is involved; spikes at fixed lags do not signal a missing unit root; and the satellites are predicted by the model, so they are information, not noise."
            },
            "ro": {
                "title": "Sateliții din ACF",
                "text": "ACF a lui $\\Delta\\Delta_{12}y_t$ are valori semnificative la decalajele 1 și 12 și două valori mici la decalajele 11 și 13. Ce arată valorile mici?",
                "options": [
                    "Produsul $\\rho_1\\rho_{12}$ al modelului airline multiplicativ",
                    "O a doua perioadă sezonieră, de 11 luni",
                    "Că seria are nevoie de o a treia diferență",
                    "Un zgomot de eșantionare care trebuie ignorat în orice situație"
                ],
                "correctExplanation": "Pentru $(1 + \\theta L)(1 + \\Theta L^{12})\\varepsilon_t$, $\\rho_{11} = \\rho_{13} = \\rho_1\\rho_{12}$: „sateliții” sînt amprenta structurii multiplicative.",
                "incorrectExplanation": "Nu există un ciclu de 11 luni; valorile la decalaje fixe nu semnalează o rădăcină unitară lipsă; iar sateliții sînt prevăzuți de model, deci sînt informație, nu zgomot."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Seasonal MA or seasonal AR?",
                "text": "After differencing, the ACF has a single significant value at lag 12 and the PACF decays at lags 12, 24, 36. Which seasonal term is suggested?",
                "options": [
                    "A seasonal AR(1), $\\Phi(L^{12}) = 1 - \\Phi L^{12}$",
                    "A seasonal MA(1), $\\Theta(L^{12}) = 1 + \\Theta L^{12}$",
                    "A regular MA(12)",
                    "No seasonal term, because one spike is not enough"
                ],
                "correctExplanation": "The rules of Chapter 2 at the seasonal lags: an ACF that cuts off after lag $s$ with a decaying seasonal PACF means a seasonal MA(1).",
                "incorrectExplanation": "A seasonal AR(1) gives the opposite pattern (ACF decaying at 12, 24, 36, PACF cutting off); an MA(12) would spend 12 parameters on one spike; and a significant seasonal spike must be modelled."
            },
            "ro": {
                "title": "MA sezonier sau AR sezonier?",
                "text": "După diferențiere, ACF are o singură valoare semnificativă la decalajul 12, iar PACF descrește la decalajele 12, 24, 36. Ce termen sezonier este sugerat?",
                "options": [
                    "Un AR(1) sezonier, $\\Phi(L^{12}) = 1 - \\Phi L^{12}$",
                    "Un MA(1) sezonier, $\\Theta(L^{12}) = 1 + \\Theta L^{12}$",
                    "Un MA(12) obișnuit",
                    "Niciun termen sezonier, pentru că o singură valoare nu este suficientă"
                ],
                "correctExplanation": "Regulile din Capitolul 2, la decalajele sezoniere: o ACF care se anulează după decalajul $s$, cu PACF sezonieră descrescătoare, indică un MA(1) sezonier.",
                "incorrectExplanation": "Un AR(1) sezonier dă tiparul opus (ACF descrescătoare la 12, 24, 36, PACF care se anulează); un MA(12) ar folosi 12 parametri pentru o singură valoare; iar o valoare sezonieră semnificativă trebuie modelată."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Deterministic or stochastic seasonality",
                "text": "In the seasonal random walk $y_t = y_{t-4} + \\varepsilon_t$, which statement is true?",
                "options": [
                    "The seasonal means are constant, so seasonal dummies remove the seasonality",
                    "The variance of $y_t$ is constant over time",
                    "Each quarter follows its own random walk, so the seasonal pattern can drift",
                    "The ACF of $y_t$ is zero at lag 4"
                ],
                "correctExplanation": "Quarters 1, 2, 3 and 4 form four separate random walks: the seasonal pattern changes permanently with every shock, and the right filter is $\\Delta_4$.",
                "incorrectExplanation": "Constant seasonal means describe deterministic seasonality; the variance of a random walk grows with $t$; and the ACF at lag 4 is close to 1, not 0."
            },
            "ro": {
                "title": "Sezonalitate deterministă sau stochastică",
                "text": "Pentru mersul aleator sezonier $y_t = y_{t-4} + \\varepsilon_t$, ce afirmație este adevărată?",
                "options": [
                    "Mediile sezoniere sînt constante, deci variabilele dummy sezoniere elimină sezonalitatea",
                    "Varianța lui $y_t$ este constantă în timp",
                    "Fiecare trimestru urmează propriul mers aleator, deci tiparul sezonier se poate deplasa",
                    "ACF a lui $y_t$ este zero la decalajul 4"
                ],
                "correctExplanation": "Trimestrele 1, 2, 3 și 4 formează patru mersuri aleatoare separate: tiparul sezonier se schimbă permanent cu fiecare șoc, iar filtrul potrivit este $\\Delta_4$.",
                "incorrectExplanation": "Mediile sezoniere constante descriu sezonalitatea deterministă; varianța unui mers aleator crește cu $t$; iar ACF la decalajul 4 este aproape de 1, nu de 0."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The HEGY test",
                "text": "What does the HEGY test add to the Dickey–Fuller test?",
                "options": [
                    "It tests for a structural break at an unknown date",
                    "It replaces the $t$-test by a Ljung–Box test",
                    "It tests stationarity as the null, like KPSS",
                    "It tests separately for unit roots at the zero frequency and at the seasonal frequencies"
                ],
                "correctExplanation": "For quarterly data, one regression tests the roots $1$ ($\\pi_1$), $-1$ ($\\pi_2$) and $\\pm i$ ($\\pi_3 = \\pi_4 = 0$): it tells which parts of $1 - L^4$ are needed.",
                "incorrectExplanation": "Break tests are Perron and Zivot–Andrews (Chapter 3); HEGY uses $t$ and $F$ tests on the $\\pi$ coefficients; and its nulls are unit roots, as in Dickey–Fuller (Canova–Hansen is the stationarity-type test)."
            },
            "ro": {
                "title": "Testul HEGY",
                "text": "Ce adaugă testul HEGY față de testul Dickey–Fuller?",
                "options": [
                    "Testează o ruptură structurală la o dată necunoscută",
                    "Înlocuiește testul $t$ cu un test Ljung–Box",
                    "Testează staționaritatea ca ipoteză nulă, ca KPSS",
                    "Testează separat rădăcinile unitare la frecvența zero și la frecvențele sezoniere"
                ],
                "correctExplanation": "Pentru date trimestriale, o singură regresie testează rădăcinile $1$ ($\\pi_1$), $-1$ ($\\pi_2$) și $\\pm i$ ($\\pi_3 = \\pi_4 = 0$): arată ce părți din $1 - L^4$ sînt necesare.",
                "incorrectExplanation": "Testele de ruptură sînt Perron și Zivot–Andrews (Capitolul 3); HEGY folosește teste $t$ și $F$ pentru coeficienții $\\pi$; iar ipotezele lui nule sînt rădăcini unitare, ca la Dickey–Fuller (Canova–Hansen este testul de tip staționaritate)."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The Canova–Hansen test",
                "text": "What is the null hypothesis of the Canova–Hansen test?",
                "options": [
                    "A stable (deterministic) seasonal pattern",
                    "A seasonal unit root at every seasonal frequency",
                    "No seasonality at all",
                    "A unit root at the zero frequency only"
                ],
                "correctExplanation": "Canova–Hansen is the seasonal analogue of KPSS: the null is stable seasonality, and rejecting it means the pattern changes over time.",
                "incorrectExplanation": "Seasonal unit roots are the null of HEGY and OCSB; the test assumes seasonality is present and asks whether it is stable; and the zero-frequency root is the subject of ADF and KPSS."
            },
            "ro": {
                "title": "Testul Canova–Hansen",
                "text": "Care este ipoteza nulă a testului Canova–Hansen?",
                "options": [
                    "Un tipar sezonier stabil (determinist)",
                    "O rădăcină unitară sezonieră la fiecare frecvență sezonieră",
                    "Absența oricărei sezonalități",
                    "O rădăcină unitară doar la frecvența zero"
                ],
                "correctExplanation": "Canova–Hansen este analogul sezonier al testului KPSS: ipoteza nulă este sezonalitatea stabilă, iar respingerea ei înseamnă că tiparul se schimbă în timp.",
                "incorrectExplanation": "Rădăcinile unitare sezoniere sînt ipoteza nulă la HEGY și OCSB; testul presupune că sezonalitatea există și întreabă dacă este stabilă; iar rădăcina de la frecvența zero este subiectul testelor ADF și KPSS."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Seasonal over-differencing",
                "text": "After $\\Delta_{12}$, a SARIMA gives $\\hat\\Theta = -0.99$. What does this suggest?",
                "options": [
                    "A strong stochastic seasonality that needs a second seasonal difference",
                    "The seasonal pattern is almost fixed: $\\Delta_{12}$ over-differences and seasonal dummies would do",
                    "The model is perfect and should be kept as it is",
                    "The regular difference should be removed"
                ],
                "correctExplanation": "$(1 - L^{12})$ on the left and $(1 - 0.99L^{12})$ on the right almost cancel: a near-unit root in the seasonal MA is the symptom of over-differencing a deterministic pattern.",
                "incorrectExplanation": "A second seasonal difference would make the problem worse; a non-invertible MA root is a warning, not a success; and the seasonal MA root concerns $D$, not $d$."
            },
            "ro": {
                "title": "Supradiferențierea sezonieră",
                "text": "După $\\Delta_{12}$, un SARIMA dă $\\hat\\Theta = -0{,}99$. Ce sugerează acest rezultat?",
                "options": [
                    "O sezonalitate stochastică puternică, care cere o a doua diferență sezonieră",
                    "Tiparul sezonier este aproape fix: $\\Delta_{12}$ supradiferențiază, iar variabilele dummy sezoniere ar fi suficiente",
                    "Modelul este perfect și trebuie păstrat așa",
                    "Diferența obișnuită trebuie eliminată"
                ],
                "correctExplanation": "$(1 - L^{12})$ în stînga și $(1 - 0{,}99L^{12})$ în dreapta aproape se simplifică: o rădăcină aproape unitară în MA sezonier este simptomul supradiferențierii unui tipar determinist.",
                "incorrectExplanation": "O a doua diferență sezonieră ar agrava problema; o rădăcină MA neinvertibilă este un avertisment, nu un succes; iar rădăcina MA sezonieră privește $D$, nu $d$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Comparing SARIMA models",
                "text": "Can AICc be used to choose between SARIMA$(1,1,1)(0,1,1)_{12}$ and SARIMA$(1,0,1)(0,1,1)_{12}$?",
                "options": [
                    "Yes, AICc always compares any two models",
                    "Yes, but only with BIC instead of AICc",
                    "No: the regular difference changes the data on which the likelihood is computed",
                    "No: AICc cannot be used for seasonal models"
                ],
                "correctExplanation": "Information criteria compare likelihoods of the same data; with different $d$ (or $D$), the differenced series differ. The orders of differencing are chosen first, by tests and plots.",
                "incorrectExplanation": "BIC has the same problem as AICc; and both criteria are fine for seasonal models, as long as $d$ and $D$ are the same."
            },
            "ro": {
                "title": "Compararea modelelor SARIMA",
                "text": "Se poate folosi AICc pentru a alege între SARIMA$(1,1,1)(0,1,1)_{12}$ și SARIMA$(1,0,1)(0,1,1)_{12}$?",
                "options": [
                    "Da, AICc compară întotdeauna oricare două modele",
                    "Da, dar doar cu BIC în locul lui AICc",
                    "Nu: diferența obișnuită schimbă datele pe care se calculează verosimilitatea",
                    "Nu: AICc nu poate fi folosit pentru modele sezoniere"
                ],
                "correctExplanation": "Criteriile informaționale compară verosimilități pentru aceleași date; cu $d$ (sau $D$) diferit, seriile diferențiate diferă. Ordinele de diferențiere se aleg mai întîi, prin teste și grafice.",
                "incorrectExplanation": "BIC are aceeași problemă ca AICc; iar ambele criterii pot fi folosite pentru modele sezoniere, cu condiția ca $d$ și $D$ să fie aceleași."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Reading official data",
                "text": "Unadjusted Romanian GDP falls by about a third from the fourth quarter to the first quarter. Which series answers the question “did the economy shrink?”?",
                "options": [
                    "The unadjusted (NSA) series, quarter on quarter",
                    "The unadjusted series, in levels",
                    "The calendar adjusted (CA) series, quarter on quarter",
                    "The seasonally and calendar adjusted (SCA) series, quarter on quarter"
                ],
                "correctExplanation": "The first quarter is always the lowest; only the seasonally and calendar adjusted series shows the change that is not due to the season or the calendar.",
                "incorrectExplanation": "Unadjusted quarter-on-quarter changes are dominated by the seasonal pattern; levels show the same pattern; and calendar adjustment alone leaves the seasonal swing in place."
            },
            "ro": {
                "title": "Citirea datelor oficiale",
                "text": "PIB-ul neajustat al României scade cu aproximativ o treime din trimestrul 4 în trimestrul 1. Ce serie răspunde la întrebarea „s-a contractat economia?”?",
                "options": [
                    "Seria neajustată (NSA), față de trimestrul anterior",
                    "Seria neajustată, în niveluri",
                    "Seria ajustată doar pentru efectele de calendar (CA), față de trimestrul anterior",
                    "Seria ajustată sezonier și pentru efectele de calendar (SCA), față de trimestrul anterior"
                ],
                "correctExplanation": "Primul trimestru este întotdeauna cel mai slab; doar seria ajustată sezonier și pentru efectele de calendar arată modificarea care nu se datorează sezonului sau calendarului.",
                "incorrectExplanation": "Modificările neajustate față de trimestrul anterior sînt dominate de tiparul sezonier; nivelurile arată același tipar; iar ajustarea doar pentru calendar lasă oscilația sezonieră neatinsă."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "X-13ARIMA-SEATS",
                "text": "What is the role of the regARIMA step in X-13ARIMA-SEATS?",
                "options": [
                    "To remove outliers and calendar effects and to extend the series with forecasts before the seasonal filters are applied",
                    "To estimate the final seasonal factors with a neural network",
                    "To replace the seasonal adjustment by an ARIMA forecast",
                    "To compute the year-on-year growth rates published by statistical offices"
                ],
                "correctExplanation": "The regression with ARIMA errors handles outliers, working days and Easter, and its forecasts extend the series so that the symmetric moving averages also work near the end: fewer revisions.",
                "incorrectExplanation": "X-13 uses moving-average filters (X-11) or a model-based decomposition (SEATS), not neural networks; the forecasts only support the adjustment; and growth rates are computed afterwards by users."
            },
            "ro": {
                "title": "X-13ARIMA-SEATS",
                "text": "Care este rolul etapei regARIMA din X-13ARIMA-SEATS?",
                "options": [
                    "Elimină valorile aberante și efectele de calendar și prelungește seria cu prognoze înainte de aplicarea filtrelor sezoniere",
                    "Estimează factorii sezonieri finali cu o rețea neuronală",
                    "Înlocuiește ajustarea sezonieră cu o prognoză ARIMA",
                    "Calculează ratele de creștere anuale publicate de institutele de statistică"
                ],
                "correctExplanation": "Regresia cu erori ARIMA tratează valorile aberante, zilele lucrătoare și Paștele, iar prognozele ei prelungesc seria, astfel încît mediile mobile simetrice funcționează și la capăt: mai puține revizuiri.",
                "incorrectExplanation": "X-13 folosește filtre de medie mobilă (X-11) sau o descompunere pe baza unui model (SEATS), nu rețele neuronale; prognozele doar sprijină ajustarea; iar ratele de creștere le calculează ulterior utilizatorii."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Orthodox Easter",
                "text": "Why can a fixed monthly seasonal pattern not capture the effect of Orthodox Easter on food sales?",
                "options": [
                    "Because Easter has no effect on sales",
                    "Because Easter moves between April and May from year to year",
                    "Because Easter always falls in March",
                    "Because seasonal patterns cannot have more than one peak"
                ],
                "correctExplanation": "Orthodox Easter fell on 5 May 2024, 20 April 2025 and 12 April 2026: the extra shopping moves between months, so it needs a regressor such as the share of the days before Easter in each month.",
                "incorrectExplanation": "The data show a clear Easter effect in food retail; Orthodox Easter falls between 4 April and 8 May, never in March; and seasonal patterns can have any shape."
            },
            "ro": {
                "title": "Paștele ortodox",
                "text": "De ce nu poate un tipar sezonier lunar fix să surprindă efectul Paștelui ortodox asupra vînzărilor de alimente?",
                "options": [
                    "Pentru că Paștele nu are niciun efect asupra vînzărilor",
                    "Pentru că Paștele se mută între aprilie și mai de la un an la altul",
                    "Pentru că Paștele cade întotdeauna în martie",
                    "Pentru că tiparele sezoniere nu pot avea mai mult de un vîrf"
                ],
                "correctExplanation": "Paștele ortodox a căzut pe 5 mai 2024, 20 aprilie 2025 și 12 aprilie 2026: cumpărăturile suplimentare se mută între luni, deci este nevoie de un regresor, de exemplu ponderea zilelor dinaintea Paștelui în fiecare lună.",
                "incorrectExplanation": "Datele arată un efect clar al Paștelui în vînzările de alimente; Paștele ortodox cade între 4 aprilie și 8 mai, niciodată în martie; iar tiparele sezoniere pot avea orice formă."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Working days",
                "text": "Why can the number of working days be used as a regressor in forecasts, unlike most explanatory variables?",
                "options": [
                    "Because it is always the same in every month",
                    "Because it is estimated by the model",
                    "Because it is known in advance from the calendar",
                    "Because it is uncorrelated with production"
                ],
                "correctExplanation": "Future working days and holidays are known years ahead, so the regression can be used to forecast without forecasting the regressor first.",
                "incorrectExplanation": "The number of working days varies from month to month (that is why it matters); it is counted, not estimated; and it is strongly correlated with production (about 2.5% per day for Romanian industry)."
            },
            "ro": {
                "title": "Zilele lucrătoare",
                "text": "De ce poate fi folosit numărul zilelor lucrătoare ca regresor în prognoze, spre deosebire de cele mai multe variabile explicative?",
                "options": [
                    "Pentru că este același în fiecare lună",
                    "Pentru că este estimat de model",
                    "Pentru că este cunoscut dinainte, din calendar",
                    "Pentru că este necorelat cu producția"
                ],
                "correctExplanation": "Zilele lucrătoare și sărbătorile viitoare sînt cunoscute cu ani înainte, deci regresia poate fi folosită pentru prognoză fără a prognoza mai întîi regresorul.",
                "incorrectExplanation": "Numărul zilelor lucrătoare variază de la o lună la alta (de aceea contează); este numărat, nu estimat; și este puternic corelat cu producția (aproximativ 2,5% pe zi pentru industria României)."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Fourier terms",
                "text": "How many coefficients does a Fourier representation with $K = 3$ harmonics of a weekly period ($m = 7$) add?",
                "options": [
                    "3",
                    "7",
                    "12",
                    "6"
                ],
                "correctExplanation": "Each harmonic $k$ adds a sine and a cosine, $\\sin(2\\pi kt/m)$ and $\\cos(2\\pi kt/m)$: $2K = 6$ coefficients.",
                "incorrectExplanation": "Each harmonic needs two terms, not one; seven would be one dummy per day, one more than needed with a constant; and 12 corresponds to $K = 6$."
            },
            "ro": {
                "title": "Termenii Fourier",
                "text": "Cîți coeficienți adaugă o reprezentare Fourier cu $K = 3$ armonici pentru o perioadă săptămînală ($m = 7$)?",
                "options": [
                    "3",
                    "7",
                    "12",
                    "6"
                ],
                "correctExplanation": "Fiecare armonică $k$ adaugă un sinus și un cosinus, $\\sin(2\\pi kt/m)$ și $\\cos(2\\pi kt/m)$: $2K = 6$ coeficienți.",
                "incorrectExplanation": "Fiecare armonică are nevoie de doi termeni, nu de unul; șapte ar însemna cîte o variabilă dummy pentru fiecare zi, cu una mai mult decît este necesar cînd există constantă; iar 12 corespunde lui $K = 6$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Non-integer periods",
                "text": "Which approach can model an annual cycle in daily data, where the period is 365.25 days?",
                "options": [
                    "Fourier terms with $m = 365.25$",
                    "A seasonal difference $\\Delta_{365.25}$",
                    "A SARIMA with $s = 365.25$",
                    "Seasonal dummies for each of the 365.25 days"
                ],
                "correctExplanation": "$\\sin(2\\pi kt/365.25)$ and $\\cos(2\\pi kt/365.25)$ are defined for any period; DHR, TBATS and Prophet all use them.",
                "incorrectExplanation": "A lag operator needs an integer number of periods, so neither $\\Delta_{365.25}$ nor a SARIMA with $s = 365.25$ exists; and dummies need an integer number of seasons (and 365 of them would be far too many)."
            },
            "ro": {
                "title": "Perioade neîntregi",
                "text": "Ce abordare poate modela un ciclu anual în date zilnice, unde perioada este de 365,25 de zile?",
                "options": [
                    "Termenii Fourier cu $m = 365{,}25$",
                    "O diferență sezonieră $\\Delta_{365{,}25}$",
                    "Un SARIMA cu $s = 365{,}25$",
                    "Variabile dummy sezoniere pentru fiecare dintre cele 365,25 de zile"
                ],
                "correctExplanation": "$\\sin(2\\pi kt/365{,}25)$ și $\\cos(2\\pi kt/365{,}25)$ sînt definite pentru orice perioadă; DHR, TBATS și Prophet le folosesc toate.",
                "incorrectExplanation": "Un operator lag are nevoie de un număr întreg de perioade, deci nici $\\Delta_{365{,}25}$, nici un SARIMA cu $s = 365{,}25$ nu există; iar variabilele dummy cer un număr întreg de sezoane (și 365 ar fi mult prea multe)."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Multiple seasonality",
                "text": "Hourly electricity load has daily, weekly and annual cycles. What is the main limitation of SARIMA here?",
                "options": [
                    "It cannot be estimated by maximum likelihood",
                    "It has a single integer seasonal period $s$",
                    "It cannot produce prediction intervals",
                    "It needs the series to be stationary without differencing"
                ],
                "correctExplanation": "SARIMA has one seasonal polynomial in $L^s$; here $s = 24$, $168$ and about $8766$ act at the same time, and a polynomial in $L^{168}$ is slow and unstable. MSTL, DHR, TBATS and Prophet handle several periods.",
                "incorrectExplanation": "SARIMA is estimated by maximum likelihood and gives intervals; and it differences non-stationary series with $d$ and $D$."
            },
            "ro": {
                "title": "Sezonalitatea multiplă",
                "text": "Consumul orar de electricitate are cicluri zilnice, săptămînale și anuale. Care este principala limită a modelului SARIMA aici?",
                "options": [
                    "Nu poate fi estimat prin verosimilitate maximă",
                    "Are o singură perioadă sezonieră întreagă $s$",
                    "Nu poate produce intervale de prognoză",
                    "Cere ca seria să fie staționară fără diferențiere"
                ],
                "correctExplanation": "SARIMA are un singur polinom sezonier în $L^s$; aici acționează simultan $s = 24$, $168$ și aproximativ $8766$, iar un polinom în $L^{168}$ este lent și instabil. MSTL, DHR, TBATS și Prophet tratează mai multe perioade.",
                "incorrectExplanation": "SARIMA se estimează prin verosimilitate maximă și dă intervale; iar seriile nestaționare sînt diferențiate prin $d$ și $D$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "TBATS",
                "text": "Which statement about TBATS is correct?",
                "options": [
                    "It models holidays through a list of dates",
                    "It is a regression with a piecewise-linear trend and changepoints",
                    "It is exponential smoothing with trigonometric seasonal states, an optional Box–Cox transformation and ARMA errors",
                    "It requires integer seasonal periods"
                ],
                "correctExplanation": "T(rigonometric seasonality), B(ox–Cox), A(RMA errors), T(rend), S(easonal components): an exponential smoothing state-space model that allows several, even non-integer, periods.",
                "incorrectExplanation": "TBATS has no regressors, so moving holidays such as Easter are not modelled; the piecewise-linear trend with changepoints is Prophet; and its trigonometric seasonality accepts periods such as 365.25."
            },
            "ro": {
                "title": "TBATS",
                "text": "Ce afirmație despre TBATS este corectă?",
                "options": [
                    "Modelează sărbătorile pe baza unei liste de date",
                    "Este o regresie cu trend liniar pe porțiuni și puncte de schimbare",
                    "Este o netezire exponențială cu stări sezoniere trigonometrice, o transformare Box–Cox opțională și erori ARMA",
                    "Cere perioade sezoniere întregi"
                ],
                "correctExplanation": "T (sezonalitate trigonometrică), B (Box–Cox), A (erori ARMA), T (trend), S (componente sezoniere): un model de netezire exponențială în spațiul stărilor care permite mai multe perioade, chiar neîntregi.",
                "incorrectExplanation": "TBATS nu are regresori, deci sărbătorile mobile, precum Paștele, nu sînt modelate; trendul liniar pe porțiuni cu puncte de schimbare este al lui Prophet; iar sezonalitatea trigonometrică acceptă perioade ca 365,25."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Prophet",
                "text": "Which decomposition does Prophet estimate?",
                "options": [
                    "$y_t = \\phi y_{t-1} + \\varepsilon_t$ with seasonal dummies",
                    "$\\phi(L)\\Phi(L^s)y_t = \\theta(L)\\Theta(L^s)\\varepsilon_t$",
                    "$y_t = \\ell_{t-1} + b_{t-1} + s_{t-m} + \\varepsilon_t$ with smoothing equations",
                    "$y(t) = g(t) + s(t) + h(t) + \\varepsilon_t$: trend, Fourier seasonality and holiday effects"
                ],
                "correctExplanation": "Prophet is a regression in time: a trend with changepoints $g(t)$, Fourier seasonal terms $s(t)$ and holiday effects $h(t)$; it has no autoregressive part.",
                "incorrectExplanation": "The first is an AR(1) with dummies; the second is SARIMA; the third is Holt–Winters exponential smoothing."
            },
            "ro": {
                "title": "Prophet",
                "text": "Ce descompunere estimează Prophet?",
                "options": [
                    "$y_t = \\phi y_{t-1} + \\varepsilon_t$ cu variabile dummy sezoniere",
                    "$\\phi(L)\\Phi(L^s)y_t = \\theta(L)\\Theta(L^s)\\varepsilon_t$",
                    "$y_t = \\ell_{t-1} + b_{t-1} + s_{t-m} + \\varepsilon_t$ cu ecuații de netezire",
                    "$y(t) = g(t) + s(t) + h(t) + \\varepsilon_t$: trend, sezonalitate Fourier și efecte de sărbătoare"
                ],
                "correctExplanation": "Prophet este o regresie în timp: un trend cu puncte de schimbare $g(t)$, termeni sezonieri Fourier $s(t)$ și efecte de sărbătoare $h(t)$; nu are parte autoregresivă.",
                "incorrectExplanation": "Prima este un AR(1) cu variabile dummy; a doua este SARIMA; a treia este netezirea exponențială Holt–Winters."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Time-series cross-validation",
                "text": "In time-series cross-validation with a rolling origin, how is the model refitted at each origin?",
                "options": [
                    "On the data up to the origin only, before forecasting the next $h$ values",
                    "On all data, including the values to be forecast",
                    "On a random 80% of the observations",
                    "Only once, on the first training sample, and never again"
                ],
                "correctExplanation": "The forecast must use only information available at the origin; the origin then moves forward and the model is refitted. This mimics real forecasting.",
                "incorrectExplanation": "Using future values leaks information; random folds break the time order; and a model fitted once is a simple train/test split, not cross-validation over many origins."
            },
            "ro": {
                "title": "Validarea încrucișată pentru serii de timp",
                "text": "În validarea încrucișată pentru serii de timp cu origine mobilă, cum se reestimează modelul la fiecare origine?",
                "options": [
                    "Doar pe datele pînă la origine, înainte de a prognoza următoarele $h$ valori",
                    "Pe toate datele, inclusiv valorile care trebuie prognozate",
                    "Pe 80% dintre observații, alese aleator",
                    "O singură dată, pe primul eșantion de antrenare, și apoi niciodată"
                ],
                "correctExplanation": "Prognoza trebuie să folosească doar informația disponibilă la origine; apoi originea se mută înainte, iar modelul se reestimează. Așa se imită prognoza reală.",
                "incorrectExplanation": "Folosirea valorilor viitoare introduce informație din viitor; grupurile aleatoare rup ordinea în timp; iar un model estimat o singură dată este o simplă împărțire în antrenare și test, nu o validare pe multe origini."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "MASE",
                "text": "A method has MASE = 0.89 on daily load, with the weekly seasonal naive method as the scale. What does this mean?",
                "options": [
                    "Its errors are 0.89% of the load",
                    "Its MAE is 11% smaller than the in-sample MAE of the weekly seasonal naive method",
                    "It is 89% accurate",
                    "It is worse than the seasonal naive method"
                ],
                "correctExplanation": "MASE = MAE / (in-sample MAE of the seasonal naive method); a value below 1 means smaller errors than that benchmark.",
                "incorrectExplanation": "MASE is a ratio of absolute errors, not a percentage of the load or an “accuracy”; and a value above 1, not below, would mean worse than the benchmark."
            },
            "ro": {
                "title": "MASE",
                "text": "O metodă are MASE = 0,89 pentru consumul zilnic, cu metoda sezonieră naivă săptămînală ca scală. Ce înseamnă?",
                "options": [
                    "Erorile ei sînt 0,89% din consum",
                    "MAE a ei este cu 11% mai mică decît MAE în eșantion a metodei sezoniere naive săptămînale",
                    "Are o acuratețe de 89%",
                    "Este mai slabă decît metoda sezonieră naivă"
                ],
                "correctExplanation": "MASE = MAE / (MAE în eșantion a metodei sezoniere naive); o valoare sub 1 înseamnă erori mai mici decît ale acestui reper.",
                "incorrectExplanation": "MASE este un raport de erori absolute, nu un procent din consum sau o „acuratețe”; iar o valoare peste 1, nu sub 1, ar însemna o metodă mai slabă decît reperul."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Reading a Diebold–Mariano test",
                "text": "With $d_t = L(e_{1t}) - L(e_{2t})$, a DM test gives a statistic of $-3.2$ with a p-value of $0.004$. What follows?",
                "options": [
                    "Forecast 2 is significantly more accurate",
                    "The two forecasts are equally accurate",
                    "Forecast 1 is significantly more accurate",
                    "Forecast 1 is unbiased"
                ],
                "correctExplanation": "A negative mean of $d_t$ means smaller losses for forecast 1, and the p-value rejects equal accuracy at 5%.",
                "incorrectExplanation": "The sign shows which forecast wins: negative favours forecast 1; equal accuracy is rejected; and the test compares accuracy, not bias."
            },
            "ro": {
                "title": "Citirea unui test Diebold–Mariano",
                "text": "Cu $d_t = L(e_{1t}) - L(e_{2t})$, un test DM dă o statistică de $-3{,}2$ cu valoarea p $0{,}004$. Ce rezultă?",
                "options": [
                    "Prognoza 2 este semnificativ mai precisă",
                    "Cele două prognoze sînt la fel de precise",
                    "Prognoza 1 este semnificativ mai precisă",
                    "Prognoza 1 este nedeplasată"
                ],
                "correctExplanation": "O medie negativă a lui $d_t$ înseamnă pierderi mai mici pentru prognoza 1, iar valoarea p respinge acuratețea egală la 5%.",
                "incorrectExplanation": "Semnul arată care prognoză cîștigă: negativ o favorizează pe prima; acuratețea egală se respinge; iar testul compară acuratețea, nu deplasarea."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Why combine forecasts",
                "text": "Two unbiased forecasts have error variances $\\sigma_1^2 = \\sigma_2^2 = 1$ and error correlation $\\rho = 0.5$. What is the MSE of their simple average?",
                "options": [
                    "0.75",
                    "1",
                    "0.5",
                    "1.5"
                ],
                "correctExplanation": "$\\mathrm{MSE}(0.5) = (\\sigma_1^2 + \\sigma_2^2 + 2\\rho\\sigma_1\\sigma_2)/4 = (1 + 1 + 1)/4 = 0.75$: lower than each forecast alone, because the errors partly cancel.",
                "incorrectExplanation": "An MSE of 1 would mean perfectly correlated errors; 0.5 would need uncorrelated errors ($\\rho = 0$); and the average of two forecasts can never be worse than the worse of the two."
            },
            "ro": {
                "title": "De ce combinăm prognozele",
                "text": "Două prognoze nedeplasate au variațiile erorilor $\\sigma_1^2 = \\sigma_2^2 = 1$ și corelația erorilor $\\rho = 0{,}5$. Cît este MSE al mediei lor simple?",
                "options": [
                    "0,75",
                    "1",
                    "0,5",
                    "1,5"
                ],
                "correctExplanation": "$\\mathrm{MSE}(0{,}5) = (\\sigma_1^2 + \\sigma_2^2 + 2\\rho\\sigma_1\\sigma_2)/4 = (1 + 1 + 1)/4 = 0{,}75$: mai mic decît al fiecărei prognoze, pentru că erorile se compensează parțial.",
                "incorrectExplanation": "Un MSE de 1 ar însemna erori perfect corelate; 0,5 ar cere erori necorelate ($\\rho = 0$); iar media a două prognoze nu poate fi niciodată mai slabă decît cea mai slabă dintre ele."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The forecast combination puzzle",
                "text": "What is the “forecast combination puzzle”?",
                "options": [
                    "Combinations are always worse than the best single forecast",
                    "Simple equal weights often beat weights estimated to be optimal",
                    "Combining forecasts makes prediction intervals impossible to compute",
                    "Only forecasts from the same model class can be combined"
                ],
                "correctExplanation": "Optimal weights must be estimated, and their sampling error often eats the theoretical gain: the simple average is hard to beat (Stock and Watson, 2004).",
                "incorrectExplanation": "Combinations are often better than most of their members and sometimes better than all; intervals can be computed (for example by simulation); and forecasts from any methods can be averaged."
            },
            "ro": {
                "title": "Paradoxul combinării prognozelor",
                "text": "Ce este „paradoxul combinării prognozelor”?",
                "options": [
                    "Combinațiile sînt întotdeauna mai slabe decît cea mai bună prognoză individuală",
                    "Ponderile egale simple bat adesea ponderile estimate ca optime",
                    "Combinarea prognozelor face imposibil calculul intervalelor de prognoză",
                    "Se pot combina doar prognoze din aceeași clasă de modele"
                ],
                "correctExplanation": "Ponderile optime trebuie estimate, iar eroarea lor de estimare consumă adesea cîștigul teoretic: media simplă este greu de bătut (Stock și Watson, 2004).",
                "incorrectExplanation": "Combinațiile sînt adesea mai bune decît majoritatea componentelor și uneori decît toate; intervalele se pot calcula (de exemplu prin simulare); iar se pot face medii ale prognozelor din orice metode."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Holidays in electricity load",
                "text": "In the cross-validation of Romanian daily load, the dynamic harmonic regression and Prophet beat TBATS and SARIMA. What is the main reason?",
                "options": [
                    "They use more data",
                    "They estimate the weekly cycle with more harmonics",
                    "They are refitted more often",
                    "They know the dates of public holidays and Orthodox Easter"
                ],
                "correctExplanation": "Holidays are the largest deviations from the weekly pattern; a model with holiday regressors lowers the forecast on the right days, while weekly models treat a holiday on a weekday as a normal working day.",
                "incorrectExplanation": "All methods use the same data and the same origins; and the weekly cycle is easy for every method, so the difference comes from the calendar."
            },
            "ro": {
                "title": "Sărbătorile în consumul de electricitate",
                "text": "În validarea încrucișată pentru consumul zilnic din România, regresia armonică dinamică și Prophet bat TBATS și SARIMA. Care este motivul principal?",
                "options": [
                    "Folosesc mai multe date",
                    "Estimează ciclul săptămînal cu mai multe armonici",
                    "Sînt reestimate mai des",
                    "Cunosc datele sărbătorilor legale și ale Paștelui ortodox"
                ],
                "correctExplanation": "Sărbătorile sînt cele mai mari abateri de la tiparul săptămînal; un model cu regresori de sărbătoare coboară prognoza în zilele corecte, în timp ce modelele săptămînale tratează o sărbătoare dintr-o zi lucrătoare ca pe o zi obișnuită.",
                "incorrectExplanation": "Toate metodele folosesc aceleași date și aceleași origini; iar ciclul săptămînal este ușor pentru orice metodă, deci diferența vine din calendar."
            }
        }
    ]
};
