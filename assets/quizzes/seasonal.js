// ============================================================
// Chapter 4 quiz bank: Seasonality and forecasting: SARIMA, TBATS, Prophet (EN + RO)
// 40 questions ported from the 2025/2026 site; 20 drawn per attempt.
// The bank grows to 24 questions when the chapter is rebuilt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['seasonal'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Seasonal differencing",
                "text": "For a monthly series, what does the seasonal difference $\\Delta_{12} Y_t = (1-L^{12})Y_t$ compute?",
                "options": [
                    "The 12-month moving average of $Y_t$",
                    "The change from the same month one year earlier, $Y_t - Y_{t-12}$",
                    "The change from the previous month, $Y_t - Y_{t-1}$",
                    "The sum of the last 12 observations"
                ],
                "correctExplanation": "$(1-L^{12})Y_t = Y_t - Y_{t-12}$: each observation is compared with the same month of the previous year, which removes a stable or slowly evolving annual pattern (a seasonal unit root).",
                "incorrectExplanation": "The operator $L^{12}$ shifts the series back 12 periods, so $(1-L^{12})$ produces a year-on-year difference, not a month-on-month difference ($1-L$), a moving average or a rolling sum."
            },
            "ro": {
                "title": "Diferențierea sezonieră",
                "text": "Pentru o serie lunară, ce calculează diferența sezonieră $\\Delta_{12} Y_t = (1-L^{12})Y_t$?",
                "options": [
                    "Media mobilă pe 12 luni a lui $Y_t$",
                    "Variația față de aceeași lună a anului precedent, $Y_t - Y_{t-12}$",
                    "Variația față de luna precedentă, $Y_t - Y_{t-1}$",
                    "Suma ultimelor 12 observații"
                ],
                "correctExplanation": "$(1-L^{12})Y_t = Y_t - Y_{t-12}$: fiecare observație este comparată cu aceeași lună a anului precedent, ceea ce elimină un tipar anual stabil sau care evoluează lent (o rădăcină unitară sezonieră).",
                "incorrectExplanation": "Operatorul $L^{12}$ deplasează seria cu 12 perioade înapoi, deci $(1-L^{12})$ produce o diferență an la an, nu o diferență lună la lună ($1-L$), o medie mobilă sau o sumă mobilă."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Seasonal period in SARIMA notation",
                "text": "A quarterly GDP series is modelled as SARIMA$(p,d,q)\\times(P,D,Q)_s$. Which value of $s$ is appropriate and what does it denote?",
                "options": [
                    "$s = 12$, the number of months in a year",
                    "$s = 7$, the maximum lag order of the model",
                    "$s = 4$, the number of observations in one seasonal cycle",
                    "$s = 52$, the minimum sample size in years"
                ],
                "correctExplanation": "The subscript $s$ is the seasonal period, the number of observations per seasonal cycle: $s=4$ for quarterly data with an annual cycle ($s=12$ monthly, $s=7$ daily with a weekly cycle, $s=52$ weekly). Together with $p,d,q,P,D,Q$ it is one of the seven values that specify a SARIMA model.",
                "incorrectExplanation": "$s$ is neither a lag order nor a sample-size requirement; it counts observations per cycle. $s=12$ would be right for monthly data, not quarterly data."
            },
            "ro": {
                "title": "Perioada sezonieră în notația SARIMA",
                "text": "O serie trimestrială a PIB este modelată ca SARIMA$(p,d,q)\\times(P,D,Q)_s$. Ce valoare a lui $s$ este potrivită și ce reprezintă ea?",
                "options": [
                    "$s = 12$, numărul de luni dintr-un an",
                    "$s = 7$, ordinul maxim al lag-urilor din model",
                    "$s = 4$, numărul de observații dintr-un ciclu sezonier",
                    "$s = 52$, volumul minim al eșantionului, în ani"
                ],
                "correctExplanation": "Indicele $s$ este perioada sezonieră, adică numărul de observații dintr-un ciclu sezonier: $s=4$ pentru date trimestriale cu ciclu anual ($s=12$ pentru date lunare, $s=7$ pentru date zilnice cu ciclu săptămînal, $s=52$ pentru date săptămînale). Împreună cu $p,d,q,P,D,Q$, este una dintre cele șapte valori care specifică un model SARIMA.",
                "incorrectExplanation": "$s$ nu este un ordin al lag-urilor și nici o cerință privind volumul eșantionului; el numără observațiile dintr-un ciclu. Valoarea $s=12$ ar fi corectă pentru date lunare, nu pentru date trimestriale."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Airline model",
                "text": "The 'airline model' introduced by Box and Jenkins (1970) for monthly airline passenger numbers is which specification?",
                "options": [
                    "SARIMA$(1,1,0)\\times(1,1,0)_{12}$, with four parameters",
                    "SARIMA$(1,1,1)\\times(1,1,1)_{12}$, with four parameters",
                    "SARIMA$(0,1,1)\\times(0,1,1)_{12}$, with two parameters",
                    "SARIMA$(0,1,0)\\times(0,1,0)_{12}$, with no parameters"
                ],
                "correctExplanation": "The airline model is $(1-L)(1-L^{12})Y_t = (1+\\theta_1 L)(1+\\Theta_1 L^{12})\\varepsilon_t$ (usually fitted to log passengers): regular and seasonal differencing plus one regular and one seasonal MA term. Only $\\theta_1$ and $\\Theta_1$ are estimated (besides $\\sigma^2$), yet it fits many monthly series well.",
                "incorrectExplanation": "The AR versions and the full $(1,1,1)\\times(1,1,1)_{12}$ model are not the airline model, and the counts attached to them are therefore irrelevant. SARIMA$(0,1,0)\\times(0,1,0)_{12}$ is a seasonal random walk with no MA terms. The airline model is the parsimonious double-differenced MA specification with two parameters."
            },
            "ro": {
                "title": "Modelul airline",
                "text": "Modelul „airline”, introdus de Box și Jenkins (1970) pentru numărul lunar de pasageri ai companiilor aeriene, are ce specificație?",
                "options": [
                    "SARIMA$(1,1,0)\\times(1,1,0)_{12}$, cu patru parametri",
                    "SARIMA$(1,1,1)\\times(1,1,1)_{12}$, cu patru parametri",
                    "SARIMA$(0,1,1)\\times(0,1,1)_{12}$, cu doi parametri",
                    "SARIMA$(0,1,0)\\times(0,1,0)_{12}$, fără parametri"
                ],
                "correctExplanation": "Modelul airline este $(1-L)(1-L^{12})Y_t = (1+\\theta_1 L)(1+\\Theta_1 L^{12})\\varepsilon_t$ (estimat de obicei pe logaritmul numărului de pasageri): diferențiere obișnuită și sezonieră, plus un termen MA obișnuit și unul sezonier. Se estimează doar $\\theta_1$ și $\\Theta_1$ (pe lîngă $\\sigma^2$), totuși modelul descrie bine multe serii lunare.",
                "incorrectExplanation": "Variantele AR și modelul complet $(1,1,1)\\times(1,1,1)_{12}$ nu sînt modelul airline. SARIMA$(0,1,0)\\times(0,1,0)_{12}$ este un mers aleator sezonier, fără termeni MA. Modelul airline este specificația MA parcimonioasă, cu dublă diferențiere și doi parametri."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Seasonal ACF pattern",
                "text": "In a monthly series, the ACF has significant spikes at lags 12, 24 and 36 that decay only slowly. What does this suggest?",
                "options": [
                    "The series is already stationary and needs no transformation",
                    "The series follows an AR(36) process",
                    "The series has stochastic seasonality and needs seasonal differencing",
                    "The series has only an ordinary unit root at frequency zero"
                ],
                "correctExplanation": "Large ACF values at multiples of $s$ (12, 24, 36, ...) that die out slowly are the seasonal analogue of the slowly decaying ACF of a random walk: they point to a seasonal unit root, which is removed by $(1-L^{12})$.",
                "incorrectExplanation": "A stationary series would show quickly decaying seasonal autocorrelations. An AR(36) is not implied by spikes at seasonal lags. A unit root at frequency zero shows up as slow decay at all lags 1, 2, 3, ..., not specifically at the seasonal lags."
            },
            "ro": {
                "title": "Tipare sezoniere în ACF",
                "text": "Într-o serie lunară, ACF are vîrfuri semnificative la lag-urile 12, 24 și 36, care descresc lent. Ce sugerează acest lucru?",
                "options": [
                    "Seria este deja staționară și nu necesită transformări",
                    "Seria urmează un proces AR(36)",
                    "Seria are sezonalitate stochastică și necesită diferențiere sezonieră",
                    "Seria are doar o rădăcină unitară obișnuită, la frecvența zero"
                ],
                "correctExplanation": "Valori mari ale ACF la multipli de $s$ (12, 24, 36, ...), care se sting lent, sînt analogul sezonier al ACF lent descrescătoare a unui mers aleator: indică o rădăcină unitară sezonieră, eliminată prin $(1-L^{12})$.",
                "incorrectExplanation": "O serie staționară ar avea autocorelații sezoniere care descresc rapid. Vîrfurile la lag-urile sezoniere nu implică un proces AR(36). O rădăcină unitară la frecvența zero apare ca o descreștere lentă la toate lag-urile 1, 2, 3, ..., nu doar la cele sezoniere."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Multiplicative SARIMA structure",
                "text": "In SARIMA, the regular and seasonal polynomials are multiplied. Expanding $(1-\\phi L)(1-\\Phi L^{12})Y_t$ for SARIMA$(1,0,0)\\times(1,0,0)_{12}$ produces which interaction term?",
                "options": [
                    "$\\phi\\Phi Y_{t-13}$",
                    "$\\phi\\Phi Y_{t-11}$",
                    "$(\\phi+\\Phi) Y_{t-12}$",
                    "$\\phi\\Phi Y_{t-24}$"
                ],
                "correctExplanation": "$(1-\\phi L)(1-\\Phi L^{12}) = 1 - \\phi L - \\Phi L^{12} + \\phi\\Phi L^{13}$, so the cross term is $\\phi\\Phi Y_{t-13}$. This is what 'multiplicative' means: the polynomials $\\phi(L)\\Phi(L^s)$ and $\\theta(L)\\Theta(L^s)$ are multiplied, creating lags $s\\pm$ regular lags with restricted coefficients.",
                "incorrectExplanation": "Multiplying $L$ by $L^{12}$ gives $L^{1+12}=L^{13}$, not $L^{11}$ or $L^{24}$, and the coefficient is the product $\\phi\\Phi$, not a sum. 'Multiplicative' here refers to the lag polynomials, not to seasonal amplitude growing with the level."
            },
            "ro": {
                "title": "Structura multiplicativă SARIMA",
                "text": "În SARIMA, polinoamele obișnuite și cele sezoniere se înmulțesc. Dezvoltînd $(1-\\phi L)(1-\\Phi L^{12})Y_t$ pentru SARIMA$(1,0,0)\\times(1,0,0)_{12}$, ce termen de interacțiune obținem?",
                "options": [
                    "$\\phi\\Phi Y_{t-13}$",
                    "$\\phi\\Phi Y_{t-11}$",
                    "$(\\phi+\\Phi) Y_{t-12}$",
                    "$\\phi\\Phi Y_{t-24}$"
                ],
                "correctExplanation": "$(1-\\phi L)(1-\\Phi L^{12}) = 1 - \\phi L - \\Phi L^{12} + \\phi\\Phi L^{13}$, deci termenul încrucișat este $\\phi\\Phi Y_{t-13}$. Acesta este sensul cuvîntului „multiplicativ”: polinoamele $\\phi(L)\\Phi(L^s)$ și $\\theta(L)\\Theta(L^s)$ se înmulțesc și generează lag-uri de forma $s\\pm$ lag-uri obișnuite, cu coeficienți restricționați.",
                "incorrectExplanation": "Produsul dintre $L$ și $L^{12}$ este $L^{1+12}=L^{13}$, nu $L^{11}$ sau $L^{24}$, iar coeficientul este produsul $\\phi\\Phi$, nu o sumă. „Multiplicativ” se referă aici la polinoamele de lag, nu la o amplitudine sezonieră care crește odată cu nivelul."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Regular and seasonal differencing",
                "text": "When should both regular ($d=1$) and seasonal ($D=1$) differencing be applied?",
                "options": [
                    "When the series has only a trend",
                    "When the series has only seasonality",
                    "When the series has both a stochastic trend and stochastic seasonality",
                    "Never, because the two operators cancel each other"
                ],
                "correctExplanation": "$(1-L)(1-L^s)$ is needed when the series has a unit root at frequency zero (stochastic trend) and a seasonal unit root (stochastic seasonality), as in the airline model.",
                "incorrectExplanation": "A trend alone calls for $d=1$ only, and stochastic seasonality alone for $D=1$ only; applying an unneeded difference over-differences the series. The two operators do not cancel: their product is $1 - L - L^s + L^{s+1}$."
            },
            "ro": {
                "title": "Diferențiere obișnuită și sezonieră",
                "text": "Cînd trebuie aplicate atît diferențierea obișnuită ($d=1$), cît și cea sezonieră ($D=1$)?",
                "options": [
                    "Cînd seria are doar un trend",
                    "Cînd seria are doar sezonalitate",
                    "Cînd seria are atît un trend stochastic, cît și sezonalitate stochastică",
                    "Niciodată, deoarece cei doi operatori se anulează reciproc"
                ],
                "correctExplanation": "$(1-L)(1-L^s)$ este necesar cînd seria are o rădăcină unitară la frecvența zero (trend stochastic) și o rădăcină unitară sezonieră (sezonalitate stochastică), ca în modelul airline.",
                "incorrectExplanation": "Un trend singur cere doar $d=1$, iar sezonalitatea stochastică singură doar $D=1$; o diferență inutilă duce la supradiferențiere. Cei doi operatori nu se anulează: produsul lor este $1 - L - L^s + L^{s+1}$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Additive and multiplicative seasonality",
                "text": "The seasonal amplitude of a series grows together with the level of its trend. Which type of seasonality is present and which transformation is recommended before fitting SARIMA?",
                "options": [
                    "Additive seasonality; no transformation is needed",
                    "Multiplicative seasonality; a log or Box-Cox transformation",
                    "Deterministic seasonality; seasonal dummy variables",
                    "Additive seasonality; second-order differencing"
                ],
                "correctExplanation": "An amplitude proportional to the level means $Y_t = T_t \\cdot S_t \\cdot \\varepsilon_t$. Taking logs gives $\\log Y_t = \\log T_t + \\log S_t + \\log\\varepsilon_t$, which is additive with stable variance (the series must be strictly positive).",
                "incorrectExplanation": "Additive seasonality has a constant amplitude regardless of the level. Dummies address whether the pattern is fixed, not whether it scales with the level. Higher-order differencing does not stabilise a variance that grows with the level."
            },
            "ro": {
                "title": "Sezonalitate aditivă și multiplicativă",
                "text": "Amplitudinea sezonieră a unei serii crește odată cu nivelul trendului. Ce tip de sezonalitate este prezent și ce transformare se recomandă înainte de estimarea unui model SARIMA?",
                "options": [
                    "Sezonalitate aditivă; nu este necesară nicio transformare",
                    "Sezonalitate multiplicativă; transformarea logaritmică sau Box-Cox",
                    "Sezonalitate deterministă; variabile dummy sezoniere",
                    "Sezonalitate aditivă; diferențiere de ordinul doi"
                ],
                "correctExplanation": "O amplitudine proporțională cu nivelul înseamnă $Y_t = T_t \\cdot S_t \\cdot \\varepsilon_t$. Prin logaritmare, $\\log Y_t = \\log T_t + \\log S_t + \\log\\varepsilon_t$, adică o structură aditivă, cu varianță stabilă (seria trebuie să fie strict pozitivă).",
                "incorrectExplanation": "Sezonalitatea aditivă are o amplitudine constantă, indiferent de nivel. Variabilele dummy privesc caracterul fix al tiparului, nu proporționalitatea cu nivelul. Diferențierea de ordin mai mare nu stabilizează o varianță care crește cu nivelul."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Combining regular and seasonal differences",
                "text": "For monthly data, $(1-L)(1-L^{12})Y_t$ equals:",
                "options": [
                    "$Y_t - Y_{t-1} - Y_{t-12} + Y_{t-13}$",
                    "$Y_t - 2Y_{t-1} + Y_{t-12}$",
                    "$Y_t - Y_{t-12}$",
                    "$Y_t - Y_{t-1} + Y_{t-12} - Y_{t-13}$"
                ],
                "correctExplanation": "$(1-L)(1-L^{12}) = 1 - L - L^{12} + L^{13}$, so $(1-L)(1-L^{12})Y_t = Y_t - Y_{t-1} - Y_{t-12} + Y_{t-13}$: the monthly change this year minus the same monthly change last year.",
                "incorrectExplanation": "The product contains the cross term $+L^{13}$ and both $-L$ and $-L^{12}$; dropping the regular difference leaves only $Y_t - Y_{t-12}$, and flipping signs or squaring $(1-L)$ gives the other expressions."
            },
            "ro": {
                "title": "Combinarea diferențelor obișnuite și sezoniere",
                "text": "Pentru date lunare, $(1-L)(1-L^{12})Y_t$ este egal cu:",
                "options": [
                    "$Y_t - Y_{t-1} - Y_{t-12} + Y_{t-13}$",
                    "$Y_t - 2Y_{t-1} + Y_{t-12}$",
                    "$Y_t - Y_{t-12}$",
                    "$Y_t - Y_{t-1} + Y_{t-12} - Y_{t-13}$"
                ],
                "correctExplanation": "$(1-L)(1-L^{12}) = 1 - L - L^{12} + L^{13}$, deci $(1-L)(1-L^{12})Y_t = Y_t - Y_{t-1} - Y_{t-12} + Y_{t-13}$: variația lunară din acest an minus aceeași variație lunară de anul trecut.",
                "incorrectExplanation": "Produsul conține termenul încrucișat $+L^{13}$ și ambii termeni $-L$ și $-L^{12}$; fără diferența obișnuită rămîne doar $Y_t - Y_{t-12}$, iar inversarea semnelor sau ridicarea la pătrat a lui $(1-L)$ duce la celelalte expresii."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Deterministic and stochastic seasonality",
                "text": "How is deterministic seasonality treated, compared with stochastic seasonality?",
                "options": [
                    "Both are treated by seasonal differencing",
                    "Deterministic by a Box-Cox transformation, stochastic by differencing",
                    "Deterministic by seasonal dummy variables, stochastic by seasonal differencing $(1-L^s)$",
                    "Deterministic by differencing, stochastic by seasonal dummy variables"
                ],
                "correctExplanation": "Deterministic seasonality is a fixed pattern repeating every year, captured by $s-1$ dummies (or Fourier terms). Stochastic seasonality evolves over time (this December depends on last December plus a shock); it contains a seasonal unit root and is removed by $(1-L^s)$.",
                "incorrectExplanation": "Differencing a deterministic seasonal pattern over-differences it (it creates a non-invertible seasonal MA), while dummies cannot capture a pattern that drifts. Box-Cox addresses variance, not the nature of seasonality."
            },
            "ro": {
                "title": "Sezonalitate deterministă și stochastică",
                "text": "Cum se tratează sezonalitatea deterministă, comparativ cu cea stochastică?",
                "options": [
                    "Ambele se tratează prin diferențiere sezonieră",
                    "Cea deterministă prin transformarea Box-Cox, cea stochastică prin diferențiere",
                    "Cea deterministă prin variabile dummy sezoniere, cea stochastică prin diferențiere sezonieră $(1-L^s)$",
                    "Cea deterministă prin diferențiere, cea stochastică prin variabile dummy sezoniere"
                ],
                "correctExplanation": "Sezonalitatea deterministă este un tipar fix, care se repetă în fiecare an, captat prin $s-1$ variabile dummy (sau termeni Fourier). Sezonalitatea stochastică evoluează în timp (decembrie din acest an depinde de decembrie din anul trecut plus un șoc); ea conține o rădăcină unitară sezonieră și se elimină prin $(1-L^s)$.",
                "incorrectExplanation": "Diferențierea unui tipar sezonier determinist duce la supradiferențiere (apare un MA sezonier neinversabil), iar variabilele dummy nu pot capta un tipar care se modifică în timp. Transformarea Box-Cox privește varianța, nu natura sezonalității."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Identification from ACF and PACF",
                "text": "After applying $\\Delta\\Delta_{12}$, the ACF has significant spikes only at lag 1 and lag 12 (with small side spikes at 11 and 13), and the PACF decays gradually at both the regular and seasonal lags. Which model is suggested?",
                "options": [
                    "SARIMA$(1,1,0)\\times(1,1,0)_{12}$",
                    "SARIMA$(1,1,1)\\times(1,1,1)_{12}$",
                    "SARIMA$(0,1,0)\\times(0,1,0)_{12}$",
                    "SARIMA$(0,1,1)\\times(0,1,1)_{12}$"
                ],
                "correctExplanation": "An ACF that cuts off after lag 1 ($q=1$) and after seasonal lag 12 ($Q=1$), with a decaying PACF, is the signature of MA(1) times seasonal MA(1): the airline model. The side spikes at 11 and 13 come from the multiplicative cross term.",
                "incorrectExplanation": "AR components would make the PACF cut off and the ACF decay, the reverse of what is seen. A model with no ARMA terms would leave the significant spikes at lags 1 and 12 unexplained, and the full (1,1,1) specification is not suggested by a clean ACF cut-off."
            },
            "ro": {
                "title": "Identificarea modelului din ACF și PACF",
                "text": "După aplicarea $\\Delta\\Delta_{12}$, ACF are vîrfuri semnificative doar la lag-ul 1 și la lag-ul 12 (cu vîrfuri laterale mici la 11 și 13), iar PACF descrește treptat atît la lag-urile obișnuite, cît și la cele sezoniere. Ce model este sugerat?",
                "options": [
                    "SARIMA$(1,1,0)\\times(1,1,0)_{12}$",
                    "SARIMA$(1,1,1)\\times(1,1,1)_{12}$",
                    "SARIMA$(0,1,0)\\times(0,1,0)_{12}$",
                    "SARIMA$(0,1,1)\\times(0,1,1)_{12}$"
                ],
                "correctExplanation": "O ACF care se anulează după lag-ul 1 ($q=1$) și după lag-ul sezonier 12 ($Q=1$), împreună cu o PACF descrescătoare, este semnătura produsului MA(1) × MA(1) sezonier: modelul airline. Vîrfurile laterale de la 11 și 13 provin din termenul încrucișat multiplicativ.",
                "incorrectExplanation": "Componentele AR ar face ca PACF să se anuleze și ACF să descrească, invers față de ceea ce se observă. Un model fără termeni ARMA nu ar explica vîrfurile semnificative de la lag-urile 1 și 12, iar specificația completă (1,1,1) nu este sugerată de o anulare netă a ACF."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The HEGY test",
                "text": "What distinguishes the HEGY test from the classical ADF test?",
                "options": [
                    "HEGY tests only for a unit root at frequency zero",
                    "HEGY is a nonparametric test that does not need any distributional assumption",
                    "HEGY tests for unit roots at frequency zero and at each seasonal frequency",
                    "HEGY can be applied only to quarterly data, not to monthly data"
                ],
                "correctExplanation": "HEGY (Hylleberg, Engle, Granger and Yoo, 1990) factorises $1-L^s$ and tests for a unit root at frequency zero and separately at each seasonal frequency (for quarterly data: $\\pi$ and $\\pi/2$), under the null of a unit root.",
                "incorrectExplanation": "The ADF test looks only at frequency zero; HEGY adds the seasonal frequencies. It is a regression-based (parametric) test like ADF, and it was extended to monthly data by Franses (1990) and Beaulieu and Miron (1993)."
            },
            "ro": {
                "title": "Testul HEGY",
                "text": "Ce diferențiază testul HEGY de testul ADF clasic?",
                "options": [
                    "HEGY testează doar o rădăcină unitară la frecvența zero",
                    "HEGY este un test neparametric, care nu necesită nicio ipoteză privind distribuția",
                    "HEGY testează rădăcini unitare la frecvența zero și la fiecare frecvență sezonieră",
                    "HEGY poate fi aplicat doar pe date trimestriale, nu și pe date lunare"
                ],
                "correctExplanation": "HEGY (Hylleberg, Engle, Granger și Yoo, 1990) factorizează $1-L^s$ și testează existența unei rădăcini unitare la frecvența zero și, separat, la fiecare frecvență sezonieră (pentru date trimestriale: $\\pi$ și $\\pi/2$), sub ipoteza nulă a rădăcinii unitare.",
                "incorrectExplanation": "Testul ADF privește doar frecvența zero; HEGY adaugă frecvențele sezoniere. Este un test pe bază de regresie (parametric), ca ADF, și a fost extins la date lunare de Franses (1990) și de Beaulieu și Miron (1993)."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Box-Cox transformation",
                "text": "In the Box-Cox family $y^{(\\lambda)} = (y^\\lambda - 1)/\\lambda$, which special case corresponds to $\\lambda = 0$?",
                "options": [
                    "The square-root transformation",
                    "No transformation (the series is unchanged up to a shift)",
                    "The log transformation",
                    "The inverse transformation"
                ],
                "correctExplanation": "As $\\lambda \\to 0$, $(y^\\lambda - 1)/\\lambda \\to \\log y$. Other cases: $\\lambda = 0.5$ is (a rescaled) square root, $\\lambda = 1$ leaves the series unchanged up to a shift, $\\lambda = -1$ is (a rescaled) inverse.",
                "incorrectExplanation": "Square root, identity and inverse correspond to $\\lambda = 0.5$, $1$ and $-1$. Only the limit $\\lambda \\to 0$ gives the logarithm."
            },
            "ro": {
                "title": "Transformarea Box-Cox",
                "text": "În familia Box-Cox $y^{(\\lambda)} = (y^\\lambda - 1)/\\lambda$, ce caz particular corespunde lui $\\lambda = 0$?",
                "options": [
                    "Transformarea rădăcină pătrată",
                    "Nicio transformare (seria rămîne neschimbată, pînă la o translație)",
                    "Transformarea logaritmică",
                    "Transformarea inversă"
                ],
                "correctExplanation": "Cînd $\\lambda \\to 0$, $(y^\\lambda - 1)/\\lambda \\to \\log y$. Alte cazuri: $\\lambda = 0{,}5$ dă (pînă la o scalare) rădăcina pătrată, $\\lambda = 1$ lasă seria neschimbată pînă la o translație, $\\lambda = -1$ dă (pînă la o scalare) inversa.",
                "incorrectExplanation": "Rădăcina pătrată, identitatea și inversa corespund valorilor $\\lambda = 0{,}5$, $1$ și $-1$. Doar limita $\\lambda \\to 0$ dă logaritmul."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "STL decomposition",
                "text": "Which of the following is NOT an advantage of STL (Seasonal-Trend decomposition using Loess) over classical decomposition?",
                "options": [
                    "It allows the seasonal component to change over time",
                    "It can be made robust to outliers",
                    "It produces forecasts directly, without an additional model",
                    "It works with any seasonal period $s > 1$"
                ],
                "correctExplanation": "STL splits the series into trend, seasonal and remainder components, but it is a decomposition method, not a forecasting model: forecasting requires, for example, a seasonal naive forecast of the seasonal component plus an ETS or ARIMA model for the seasonally adjusted series.",
                "incorrectExplanation": "Time-varying seasonality, an optional robust fitting step and any seasonal period are genuine advantages of STL over classical decomposition; direct forecasting is not something STL provides."
            },
            "ro": {
                "title": "Descompunerea STL",
                "text": "Care dintre următoarele NU este un avantaj al descompunerii STL (Seasonal-Trend decomposition using Loess) față de descompunerea clasică?",
                "options": [
                    "Permite ca componenta sezonieră să se modifice în timp",
                    "Poate fi făcută robustă la valori extreme",
                    "Generează direct prognoze, fără un model suplimentar",
                    "Funcționează cu orice perioadă sezonieră $s > 1$"
                ],
                "correctExplanation": "STL descompune seria în trend, componentă sezonieră și rest, dar este o metodă de descompunere, nu un model de prognoză: prognoza necesită, de exemplu, o prognoză seasonal naive pentru componenta sezonieră și un model ETS sau ARIMA pentru seria ajustată sezonier.",
                "incorrectExplanation": "Sezonalitatea variabilă în timp, pasul opțional de estimare robustă și orice perioadă sezonieră sînt avantaje reale ale STL față de descompunerea clasică; prognoza directă nu este oferită de STL."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The MASE metric",
                "text": "What does a MASE (Mean Absolute Scaled Error) value greater than 1 mean for a seasonal series?",
                "options": [
                    "The model is better than the seasonal naive benchmark",
                    "The model is equivalent to the seasonal naive benchmark",
                    "The model has a percentage error of 100%",
                    "The model performs worse than the in-sample one-step seasonal naive forecast"
                ],
                "correctExplanation": "MASE divides the model's MAE by the in-sample MAE of the one-step seasonal naive forecast $Y_t - Y_{t-s}$ (Hyndman and Koehler, 2006). MASE > 1 means the model's errors are larger, on average, than those of this simple benchmark.",
                "incorrectExplanation": "MASE < 1 means better than the benchmark and MASE = 1 means equal. MASE is a scaled absolute error, not a percentage error, so 1 does not mean 100%."
            },
            "ro": {
                "title": "Indicatorul MASE",
                "text": "Ce înseamnă o valoare MASE (Mean Absolute Scaled Error) mai mare decît 1 pentru o serie sezonieră?",
                "options": [
                    "Modelul este mai bun decît reperul seasonal naive",
                    "Modelul este echivalent cu reperul seasonal naive",
                    "Modelul are o eroare procentuală de 100%",
                    "Modelul are erori mai mari decît prognoza seasonal naive cu un pas, calculată în eșantion"
                ],
                "correctExplanation": "MASE împarte MAE al modelului la MAE în eșantion al prognozei seasonal naive cu un pas, $Y_t - Y_{t-s}$ (Hyndman și Koehler, 2006). MASE > 1 înseamnă că erorile modelului sînt, în medie, mai mari decît cele ale acestui reper simplu.",
                "incorrectExplanation": "MASE < 1 înseamnă un model mai bun decît reperul, iar MASE = 1 înseamnă performanță egală. MASE este o eroare absolută scalată, nu o eroare procentuală, deci valoarea 1 nu înseamnă 100%."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The Canova-Hansen test",
                "text": "What is the null hypothesis of the Canova-Hansen test, and how does it relate to HEGY?",
                "options": [
                    "$H_0$: a seasonal unit root is present (as in HEGY)",
                    "$H_0$: there is no seasonality of any kind",
                    "$H_0$: there is no seasonal unit root, i.e. the seasonal pattern is stable (the opposite of HEGY)",
                    "$H_0$: there is a unit root at frequency zero (as in ADF)"
                ],
                "correctExplanation": "Canova and Hansen (1995) test the null of a stable (deterministic or stationary) seasonal pattern against a seasonal unit root. HEGY has a seasonal unit root as the null, so the roles are reversed, as for KPSS and ADF at frequency zero.",
                "incorrectExplanation": "The null is neither 'no seasonality' (deterministic seasonality is allowed under the null) nor a unit root; Canova-Hansen concerns the seasonal frequencies, not frequency zero."
            },
            "ro": {
                "title": "Testul Canova-Hansen",
                "text": "Care este ipoteza nulă a testului Canova-Hansen și cum se raportează ea la testul HEGY?",
                "options": [
                    "$H_0$: există o rădăcină unitară sezonieră (ca la HEGY)",
                    "$H_0$: nu există sezonalitate de niciun tip",
                    "$H_0$: nu există o rădăcină unitară sezonieră, adică tiparul sezonier este stabil (invers față de HEGY)",
                    "$H_0$: există o rădăcină unitară la frecvența zero (ca la ADF)"
                ],
                "correctExplanation": "Canova și Hansen (1995) testează ipoteza nulă a unui tipar sezonier stabil (determinist sau staționar) față de alternativa unei rădăcini unitare sezoniere. La HEGY, ipoteza nulă este rădăcina unitară sezonieră, deci rolurile sînt inversate, ca la KPSS față de ADF la frecvența zero.",
                "incorrectExplanation": "Ipoteza nulă nu este „fără sezonalitate” (sezonalitatea deterministă este permisă sub ipoteza nulă) și nici o rădăcină unitară; testul Canova-Hansen privește frecvențele sezoniere, nu frecvența zero."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Seasonal naive forecast",
                "text": "What is the seasonal naive forecast for a monthly series ($s = 12$), for horizons $h \\le s$?",
                "options": [
                    "$\\hat{Y}_{T+h} = Y_T$ (the last observation)",
                    "$\\hat{Y}_{T+h} = \\bar{Y}$ (the sample mean)",
                    "$\\hat{Y}_{T+h} = Y_{T+h-s}$ (the value in the same month of the last observed year)",
                    "$\\hat{Y}_{T+h} = Y_T + h\\hat{\\beta}$ (linear trend extrapolation)"
                ],
                "correctExplanation": "The seasonal naive forecast repeats the last observed seasonal cycle: next March equals last March. For longer horizons the cycle is repeated, $\\hat{Y}_{T+h} = Y_{T+h-s(k+1)}$ with $k = \\lfloor (h-1)/s \\rfloor$. It is the standard benchmark (and the scaling of MASE) for seasonal series.",
                "incorrectExplanation": "The last observation is the plain naive forecast, the sample mean is the mean forecast, and the trend extrapolation is the drift method; none of them uses the seasonal pattern."
            },
            "ro": {
                "title": "Prognoza seasonal naive",
                "text": "Care este prognoza seasonal naive pentru o serie lunară ($s = 12$), pentru orizonturi $h \\le s$?",
                "options": [
                    "$\\hat{Y}_{T+h} = Y_T$ (ultima observație)",
                    "$\\hat{Y}_{T+h} = \\bar{Y}$ (media eșantionului)",
                    "$\\hat{Y}_{T+h} = Y_{T+h-s}$ (valoarea din aceeași lună a ultimului an observat)",
                    "$\\hat{Y}_{T+h} = Y_T + h\\hat{\\beta}$ (extrapolarea liniară a trendului)"
                ],
                "correctExplanation": "Prognoza seasonal naive repetă ultimul ciclu sezonier observat: luna martie viitoare este egală cu luna martie trecută. Pentru orizonturi mai lungi ciclul se repetă, $\\hat{Y}_{T+h} = Y_{T+h-s(k+1)}$, cu $k = \\lfloor (h-1)/s \\rfloor$. Este reperul standard (și numitorul MASE) pentru serii sezoniere.",
                "incorrectExplanation": "Ultima observație este prognoza naive simplă, media eșantionului este prognoza prin medie, iar extrapolarea trendului este metoda drift; niciuna nu folosește tiparul sezonier."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Seasonal over-differencing",
                "text": "What is the main symptom of seasonal over-differencing?",
                "options": [
                    "The ACF decays slowly at all lags",
                    "The ACF at the seasonal lag $s$ is close to $-0.5$",
                    "The PACF has significant spikes at every lag",
                    "The variance of the series grows exponentially"
                ],
                "correctExplanation": "Differencing a series that has no seasonal unit root creates a seasonal MA root $\\Theta \\approx -1$; for $(1-L^s)\\varepsilon_t$ the autocorrelation at lag $s$ is exactly $-0.5$, so a strongly negative ACF near $-0.5$ at lag $s$ signals over-differencing.",
                "incorrectExplanation": "Slow ACF decay indicates under-differencing (a remaining unit root), not over-differencing. Over-differencing does not create spikes at all PACF lags, and it increases the variance moderately rather than exponentially."
            },
            "ro": {
                "title": "Supradiferențierea sezonieră",
                "text": "Care este principalul simptom al supradiferențierii sezoniere?",
                "options": [
                    "ACF descrește lent la toate lag-urile",
                    "ACF la lag-ul sezonier $s$ este apropiată de $-0{,}5$",
                    "PACF are vîrfuri semnificative la toate lag-urile",
                    "Varianța seriei crește exponențial"
                ],
                "correctExplanation": "Diferențierea unei serii fără rădăcină unitară sezonieră creează o rădăcină MA sezonieră $\\Theta \\approx -1$; pentru $(1-L^s)\\varepsilon_t$, autocorelația la lag-ul $s$ este exact $-0{,}5$, deci o ACF puternic negativă, apropiată de $-0{,}5$ la lag-ul $s$, semnalează supradiferențierea.",
                "incorrectExplanation": "Descreșterea lentă a ACF indică o diferențiere insuficientă (o rădăcină unitară rămasă), nu supradiferențiere. Supradiferențierea nu creează vîrfuri la toate lag-urile PACF și crește varianța moderat, nu exponențial."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Diagnostic checking of SARIMA",
                "text": "When checking a fitted SARIMA model for monthly data, the residual ACF has a significant spike at lag 12. What does this indicate?",
                "options": [
                    "The model is correctly specified and captures all seasonal structure",
                    "The residuals follow the Normal distribution",
                    "The seasonal component is inadequately modelled and the specification needs revision",
                    "The original series has no seasonality"
                ],
                "correctExplanation": "Residual autocorrelation at the seasonal lag means seasonal dependence is left in the errors: add or revise SAR/SMA terms (or reconsider $D$), then re-check the residual ACF and the Ljung-Box test at seasonal lags.",
                "incorrectExplanation": "A well-specified model leaves white-noise residuals, with no spike at lag 12. The ACF says nothing about normality. A remaining seasonal spike shows that the original series is seasonal, not the opposite."
            },
            "ro": {
                "title": "Validarea unui model SARIMA",
                "text": "La validarea unui model SARIMA estimat pe date lunare, ACF a reziduurilor are un vîrf semnificativ la lag-ul 12. Ce indică acest lucru?",
                "options": [
                    "Modelul este corect specificat și captează toată structura sezonieră",
                    "Reziduurile urmează distribuția Normală",
                    "Componenta sezonieră este modelată inadecvat, iar specificația trebuie revizuită",
                    "Seria inițială nu prezintă sezonalitate"
                ],
                "correctExplanation": "Autocorelația reziduurilor la lag-ul sezonier înseamnă că o parte din dependența sezonieră a rămas în erori: se adaugă sau se modifică termenii SAR/SMA (ori se reconsideră $D$), apoi se verifică din nou ACF a reziduurilor și testul Ljung-Box la lag-urile sezoniere.",
                "incorrectExplanation": "Un model bine specificat lasă reziduuri de tip zgomot alb, fără vîrf la lag-ul 12. ACF nu spune nimic despre normalitate. Un vîrf sezonier rămas arată că seria inițială este sezonieră, nu invers."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Information criterion in auto.arima",
                "text": "In automatic SARIMA selection with the Hyndman-Khandakar algorithm (auto.arima), which criterion is used by default to compare candidate models?",
                "options": [
                    "BIC (Bayesian Information Criterion)",
                    "AICc (AIC corrected for small samples)",
                    "RMSE (Root Mean Squared Error)",
                    "MAPE (Mean Absolute Percentage Error)"
                ],
                "correctExplanation": "auto.arima chooses $d$ and $D$ with unit-root and seasonal-strength tests, then searches over the ARMA orders minimising $\\text{AICc} = \\text{AIC} + \\frac{2k(k+1)}{T-k-1}$, which corrects the small-sample bias of AIC.",
                "incorrectExplanation": "BIC is available as an option but is not the default. RMSE and MAPE are accuracy measures, usually computed on a test set; they do not penalise the number of parameters and are not used by the algorithm to rank models."
            },
            "ro": {
                "title": "Criteriul informațional din auto.arima",
                "text": "În selecția automată a unui model SARIMA prin algoritmul Hyndman-Khandakar (auto.arima), ce criteriu se folosește implicit pentru compararea modelelor candidate?",
                "options": [
                    "BIC (Bayesian Information Criterion)",
                    "AICc (AIC corectat pentru eșantioane mici)",
                    "RMSE (Root Mean Squared Error)",
                    "MAPE (Mean Absolute Percentage Error)"
                ],
                "correctExplanation": "auto.arima alege $d$ și $D$ cu ajutorul unor teste de rădăcină unitară și de intensitate a sezonalității, apoi caută ordinele ARMA care minimizează $\\text{AICc} = \\text{AIC} + \\frac{2k(k+1)}{T-k-1}$, criteriu care corectează deplasarea AIC în eșantioane mici.",
                "incorrectExplanation": "BIC este disponibil ca opțiune, dar nu este implicit. RMSE și MAPE sînt măsuri de acuratețe, calculate de obicei pe un set de test; ele nu penalizează numărul de parametri și nu sînt folosite de algoritm pentru ordonarea modelelor."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Multiple seasonality",
                "text": "Why is a standard SARIMA model not suited to hourly electricity demand?",
                "options": [
                    "SARIMA can only be used for monthly data",
                    "SARIMA allows only one seasonal period $s$",
                    "SARIMA cannot include a trend",
                    "SARIMA requires normally distributed data"
                ],
                "correctExplanation": "Hourly demand has daily (24), weekly (168) and annual (about 8766) cycles. A SARIMA has a single seasonal period, and a period as long as 168 or 8766 is impractical to estimate; models such as TBATS, Prophet or regression with Fourier terms and ARMA errors handle several periods.",
                "incorrectExplanation": "SARIMA works at any frequency, handles trends through differencing and drift, and does not need normally distributed data for estimation. The limitation is the single seasonal period."
            },
            "ro": {
                "title": "Sezonalitate multiplă",
                "text": "De ce un model SARIMA standard nu este potrivit pentru consumul orar de energie electrică?",
                "options": [
                    "SARIMA poate fi folosit doar pentru date lunare",
                    "SARIMA permite o singură perioadă sezonieră $s$",
                    "SARIMA nu poate include un trend",
                    "SARIMA necesită date cu distribuția Normală"
                ],
                "correctExplanation": "Consumul orar are cicluri zilnice (24), săptămînale (168) și anuale (aproximativ 8766). Un model SARIMA are o singură perioadă sezonieră, iar o perioadă de 168 sau 8766 este greu de estimat în practică; modele precum TBATS, Prophet sau regresia cu termeni Fourier și erori ARMA tratează mai multe perioade.",
                "incorrectExplanation": "SARIMA funcționează la orice frecvență, tratează trendul prin diferențiere și drift și nu necesită date cu distribuția Normală pentru estimare. Limita reală este perioada sezonieră unică."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The TBATS acronym",
                "text": "What does TBATS stand for?",
                "options": [
                    "Trend, Baseline, ARMA, Transform, Seasonal",
                    "Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend, Seasonal components",
                    "Time-Based Automatic Time Series",
                    "Temporal Bayesian Adaptive Trend System"
                ],
                "correctExplanation": "TBATS (De Livera, Hyndman and Snyder, 2011): Trigonometric (Fourier) seasonal terms, Box-Cox transformation, ARMA errors, (damped) Trend and multiple Seasonal components, all in an exponential smoothing state space model.",
                "incorrectExplanation": "The other expansions sound plausible but are invented; in particular TBATS is not a Bayesian model. Each letter refers to a component of the model."
            },
            "ro": {
                "title": "Acronimul TBATS",
                "text": "Ce înseamnă TBATS?",
                "options": [
                    "Trend, Baseline, ARMA, Transform, Seasonal",
                    "Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend, Seasonal components",
                    "Time-Based Automatic Time Series",
                    "Temporal Bayesian Adaptive Trend System"
                ],
                "correctExplanation": "TBATS (De Livera, Hyndman și Snyder, 2011): termeni sezonieri trigonometrici (Fourier), transformarea Box-Cox, erori ARMA, trend (eventual amortizat) și mai multe componente sezoniere, toate într-un model de nivelare exponențială în spațiul stărilor.",
                "incorrectExplanation": "Celelalte variante par plauzibile, dar sînt inventate; în particular, TBATS nu este un model bayesian. Fiecare literă desemnează o componentă a modelului."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Number of Fourier harmonics",
                "text": "Increasing the number of Fourier harmonics $K$ for one seasonal period (in TBATS or a regression with Fourier terms):",
                "options": [
                    "Always improves forecast accuracy",
                    "Allows more flexible seasonal shapes, at the risk of overfitting",
                    "Reduces the complexity of the model",
                    "Removes the need for a Box-Cox transformation"
                ],
                "correctExplanation": "Each harmonic adds a sine-cosine pair, so a larger $K$ captures sharper seasonal shapes (at most $K = m/2$). Too many harmonics give jagged seasonal patterns and poor out-of-sample accuracy, so $K$ is chosen by AIC or cross-validation.",
                "incorrectExplanation": "More harmonics mean more parameters, not fewer, and extra flexibility can worsen forecasts through overfitting. Fourier terms describe the seasonal shape; they do not stabilise the variance, which is the role of Box-Cox."
            },
            "ro": {
                "title": "Numărul de armonici Fourier",
                "text": "Creșterea numărului de armonici Fourier $K$ pentru o perioadă sezonieră (în TBATS sau într-o regresie cu termeni Fourier):",
                "options": [
                    "Îmbunătățește întotdeauna acuratețea prognozei",
                    "Permite forme sezoniere mai flexibile, cu riscul supraajustării (overfitting)",
                    "Reduce complexitatea modelului",
                    "Elimină necesitatea transformării Box-Cox"
                ],
                "correctExplanation": "Fiecare armonică adaugă o pereche sinus-cosinus, deci un $K$ mai mare captează forme sezoniere mai accentuate (cel mult $K = m/2$). Prea multe armonici produc tipare sezoniere zimțate și o acuratețe slabă în afara eșantionului, de aceea $K$ se alege prin AIC sau validare încrucișată.",
                "incorrectExplanation": "Mai multe armonici înseamnă mai mulți parametri, nu mai puțini, iar flexibilitatea suplimentară poate înrăutăți prognozele prin supraajustare. Termenii Fourier descriu forma sezonieră; ei nu stabilizează varianța, acesta fiind rolul transformării Box-Cox."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Prophet decomposition",
                "text": "Prophet decomposes a time series into which components, and why is this useful in practice?",
                "options": [
                    "AR, MA and seasonal components, which give exact likelihood inference",
                    "Trend, seasonality, holidays and error, which can be plotted and explained separately",
                    "Mean, variance and autocorrelation, which summarise the dependence structure",
                    "Level, slope and curvature, which describe the yield curve"
                ],
                "correctExplanation": "Prophet is a curve-fitting model $y(t) = g(t) + s(t) + h(t) + \\varepsilon_t$ (trend, seasonality, holiday effects, error). Each component can be plotted, so analysts can explain which part drives a forecast, a key advantage over black-box ML models.",
                "incorrectExplanation": "Prophet has no AR or MA terms; mean, variance and autocorrelation are summary statistics, not model components; level, slope and curvature are the Nelson-Siegel factors of the yield curve."
            },
            "ro": {
                "title": "Descompunerea Prophet",
                "text": "În ce componente descompune Prophet o serie de timp și de ce este utilă această descompunere în practică?",
                "options": [
                    "Componente AR, MA și sezoniere, care permit inferență pe baza verosimilității exacte",
                    "Trend, sezonalitate, sărbători și eroare, care pot fi reprezentate grafic și explicate separat",
                    "Medie, varianță și autocorelație, care rezumă structura de dependență",
                    "Nivel, pantă și curbură, care descriu curba randamentelor"
                ],
                "correctExplanation": "Prophet este un model de ajustare a curbelor, $y(t) = g(t) + s(t) + h(t) + \\varepsilon_t$ (trend, sezonalitate, efectele sărbătorilor, eroare). Fiecare componentă poate fi reprezentată grafic, astfel încît analiștii pot explica ce parte determină prognoza, un avantaj important față de modelele ML de tip cutie neagră.",
                "incorrectExplanation": "Prophet nu are termeni AR sau MA; media, varianța și autocorelația sînt statistici descriptive, nu componente ale modelului; nivelul, panta și curbura sînt factorii Nelson-Siegel ai curbei randamentelor."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Choosing between Prophet and TBATS",
                "text": "Hourly energy demand shows daily, weekly and annual cycles plus strong holiday effects. When is Prophet preferable to TBATS?",
                "options": [
                    "When fully automatic model selection is the priority",
                    "When known holidays, special events and changepoints must be built into the model",
                    "When the most parsimonious model is required",
                    "When the data have no trend"
                ],
                "correctExplanation": "Both models handle multiple seasonality. Prophet makes it easy to add domain knowledge (holiday calendars, events, known changepoints, extra regressors), whereas TBATS is the more automatic choice. In practice both are compared by rolling-origin cross-validation.",
                "incorrectExplanation": "Automatic selection is TBATS's strength, not Prophet's. Neither model is chosen for parsimony, and the absence of a trend does not favour Prophet, whose main feature is a flexible trend."
            },
            "ro": {
                "title": "Alegerea între Prophet și TBATS",
                "text": "Consumul orar de energie prezintă cicluri zilnice, săptămînale și anuale, plus efecte puternice ale sărbătorilor. Cînd este preferabil Prophet față de TBATS?",
                "options": [
                    "Cînd prioritatea este selecția complet automată a modelului",
                    "Cînd sărbătorile cunoscute, evenimentele speciale și punctele de schimbare trebuie incluse în model",
                    "Cînd este necesar cel mai parcimonios model",
                    "Cînd datele nu au trend"
                ],
                "correctExplanation": "Ambele modele tratează sezonalitatea multiplă. Prophet permite includerea ușoară a cunoștințelor din domeniu (calendare de sărbători, evenimente, changepoints cunoscute, regresori suplimentari), pe cînd TBATS este alegerea mai automată. În practică, cele două se compară prin validare încrucișată cu origine mobilă.",
                "incorrectExplanation": "Selecția automată este punctul forte al TBATS, nu al Prophet. Niciunul dintre modele nu se alege pentru parcimonie, iar lipsa trendului nu favorizează Prophet, a cărui caracteristică principală este tocmai un trend flexibil."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Seasonality mode in Prophet",
                "text": "In retail sales, December sales are about three times the monthly average, and the December peak grows as sales grow. Which seasonality mode is more appropriate in Prophet?",
                "options": [
                    "Additive: the seasonal effect is a fixed amount, independent of the level",
                    "Multiplicative: the seasonal effect scales with the level of the series",
                    "It does not matter: Prophet chooses the mode automatically",
                    "Neither: such a pattern can only be modelled with ARIMA"
                ],
                "correctExplanation": "When seasonal effects are proportional to the level ('three times', not '+1000 units'), use seasonality_mode='multiplicative', so that $y(t) = g(t)\\,(1 + s(t)) + \\varepsilon_t$.",
                "incorrectExplanation": "An additive effect would add the same amount every December even as the level grows. Prophet does not choose the mode itself (the default is additive), and it handles proportional seasonality without ARIMA."
            },
            "ro": {
                "title": "Modul de sezonalitate în Prophet",
                "text": "În comerțul cu amănuntul, vînzările din decembrie sînt de aproximativ trei ori mai mari decît media lunară, iar vîrful din decembrie crește odată cu vînzările. Ce mod de sezonalitate este mai potrivit în Prophet?",
                "options": [
                    "Aditiv: efectul sezonier este o valoare fixă, independentă de nivel",
                    "Multiplicativ: efectul sezonier se scalează cu nivelul seriei",
                    "Nu contează: Prophet alege automat modul",
                    "Niciunul: un astfel de tipar poate fi modelat doar cu ARIMA"
                ],
                "correctExplanation": "Cînd efectele sezoniere sînt proporționale cu nivelul („de trei ori”, nu „+1000 de unități”), se folosește seasonality_mode='multiplicative', astfel încît $y(t) = g(t)\\,(1 + s(t)) + \\varepsilon_t$.",
                "incorrectExplanation": "Un efect aditiv ar adăuga aceeași valoare în fiecare decembrie, chiar dacă nivelul crește. Prophet nu alege singur modul (implicit este cel aditiv) și tratează sezonalitatea proporțională fără ARIMA."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Changepoints in Prophet",
                "text": "What are changepoints in Prophet?",
                "options": [
                    "Points where the seasonal pattern or period changes",
                    "Points where the growth rate (slope) of the trend changes",
                    "Points where the variance of the errors changes",
                    "Points where outliers are detected and removed"
                ],
                "correctExplanation": "Changepoints make the trend piecewise linear (or piecewise logistic): at each changepoint the slope can change. By default Prophet places 25 potential changepoints in the first 80% of the history and shrinks most rate changes towards zero; users can also specify them.",
                "incorrectExplanation": "Changepoints act only on the trend $g(t)$: they do not change the seasonal period, the error variance or remove outliers."
            },
            "ro": {
                "title": "Changepoints în Prophet",
                "text": "Ce sînt changepoints în Prophet?",
                "options": [
                    "Momente în care tiparul sau perioada sezonieră se schimbă",
                    "Momente în care rata de creștere (panta) a trendului se schimbă",
                    "Momente în care varianța erorilor se schimbă",
                    "Momente în care valorile extreme sînt detectate și eliminate"
                ],
                "correctExplanation": "Changepoints fac trendul liniar pe porțiuni (sau logistic pe porțiuni): în fiecare changepoint panta se poate modifica. Implicit, Prophet plasează 25 de changepoints potențiale în primele 80% din istoric și restrînge spre zero majoritatea variațiilor de pantă; utilizatorul le poate și specifica.",
                "incorrectExplanation": "Changepoints acționează doar asupra trendului $g(t)$: nu schimbă perioada sezonieră, nu modifică varianța erorilor și nu elimină valori extreme."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Parsimony with a single seasonality",
                "text": "Daily call-centre volumes show only a weekly seasonal pattern. Which model is the most appropriate starting point?",
                "options": [
                    "TBATS, because it is designed for multiple seasonality",
                    "Prophet, because it handles any seasonality well",
                    "A standard SARIMA with $s = 7$, because it is simpler and sufficient",
                    "An LSTM network, because it is the most flexible"
                ],
                "correctExplanation": "Parsimony: with a single, short seasonal period ($s=7$), SARIMA (or ETS) is enough and well understood. More complex models should be adopted only if they beat it in out-of-sample evaluation.",
                "incorrectExplanation": "TBATS and Prophet add flexibility that is not needed for one short period, and an LSTM needs much more data and tuning; extra flexibility often hurts forecasts when the structure is simple."
            },
            "ro": {
                "title": "Parcimonie în cazul unei singure sezonalități",
                "text": "Volumul zilnic de apeluri al unui call center prezintă doar un tipar sezonier săptămînal. Ce model este cel mai potrivit punct de plecare?",
                "options": [
                    "TBATS, deoarece este conceput pentru sezonalitate multiplă",
                    "Prophet, deoarece tratează bine orice sezonalitate",
                    "Un model SARIMA standard cu $s = 7$, deoarece este mai simplu și suficient",
                    "O rețea LSTM, deoarece este cea mai flexibilă"
                ],
                "correctExplanation": "Principiul parcimoniei: pentru o singură perioadă sezonieră scurtă ($s=7$), SARIMA (sau ETS) este suficient și bine înțeles. Modelele mai complexe se adoptă doar dacă îl depășesc în evaluarea în afara eșantionului.",
                "incorrectExplanation": "TBATS și Prophet adaugă o flexibilitate inutilă pentru o singură perioadă scurtă, iar o rețea LSTM cere mult mai multe date și reglaje; flexibilitatea suplimentară strică adesea prognozele cînd structura este simplă."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Prophet prediction intervals",
                "text": "With its default settings (MAP estimation), how does Prophet produce prediction intervals?",
                "options": [
                    "From analytic ARMA formulas based on the $\\psi$-weights",
                    "By simulating future trend changes, with the frequency and size of past changepoints, plus observation noise",
                    "By bootstrap resampling of historical forecast errors",
                    "By adding a fixed percentage (for example, plus or minus 10%) to the point forecast"
                ],
                "correctExplanation": "Prophet assumes that future changepoints occur as often and are as large as in the history; it simulates many future trend paths and adds observation noise. The intervals therefore widen quickly with the horizon, and trend uncertainty dominates long-horizon forecasts. Seasonal parameter uncertainty is included only with full MCMC sampling (mcmc_samples > 0).",
                "incorrectExplanation": "Prophet has no ARMA structure, does not bootstrap past errors and does not use fixed-width bands; the default intervals come from simulation of trend changes and noise, not from sampling the full posterior of all parameters."
            },
            "ro": {
                "title": "Intervalele de prognoză în Prophet",
                "text": "Cu setările implicite (estimare MAP), cum construiește Prophet intervalele de prognoză?",
                "options": [
                    "Prin formule analitice ARMA, pe baza ponderilor $\\psi$",
                    "Prin simularea unor schimbări viitoare ale trendului, cu frecvența și mărimea changepoints din trecut, la care se adaugă zgomotul de observație",
                    "Prin reeșantionare bootstrap a erorilor de prognoză din trecut",
                    "Prin adăugarea unui procent fix (de exemplu, plus sau minus 10%) la prognoza punctuală"
                ],
                "correctExplanation": "Prophet presupune că changepoints viitoare apar la fel de des și sînt la fel de mari ca în istoric; simulează multe traiectorii viitoare ale trendului și adaugă zgomotul de observație. De aceea intervalele se lărgesc rapid cu orizontul, iar incertitudinea trendului domină prognozele pe termen lung. Incertitudinea parametrilor sezonieri este inclusă doar cu eșantionare MCMC completă (mcmc_samples > 0).",
                "incorrectExplanation": "Prophet nu are structură ARMA, nu aplică bootstrap pe erorile din trecut și nu folosește benzi de lățime fixă; intervalele implicite provin din simularea schimbărilor de trend și a zgomotului, nu din eșantionarea distribuției a posteriori a tuturor parametrilor."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Box-Cox in TBATS",
                "text": "What is the purpose of the Box-Cox transformation in TBATS?",
                "options": [
                    "To remove the trend",
                    "To stabilise a variance that changes with the level of the series",
                    "To remove seasonality",
                    "To speed up computation"
                ],
                "correctExplanation": "$y^{(\\lambda)} = (y^\\lambda - 1)/\\lambda$ (with $\\lambda = 0$ giving the log) makes the variance roughly constant when it grows with the level; TBATS estimates $\\lambda$ or decides by AIC whether to use the transformation at all.",
                "incorrectExplanation": "Trend and seasonality are handled by the trend and trigonometric seasonal components, not by Box-Cox, and the transformation adds an estimation step rather than saving time."
            },
            "ro": {
                "title": "Transformarea Box-Cox în TBATS",
                "text": "Care este rolul transformării Box-Cox în TBATS?",
                "options": [
                    "Eliminarea trendului",
                    "Stabilizarea unei varianțe care se modifică odată cu nivelul seriei",
                    "Eliminarea sezonalității",
                    "Accelerarea calculelor"
                ],
                "correctExplanation": "$y^{(\\lambda)} = (y^\\lambda - 1)/\\lambda$ (pentru $\\lambda = 0$ se obține logaritmul) face varianța aproximativ constantă cînd aceasta crește cu nivelul; TBATS estimează $\\lambda$ sau decide pe baza AIC dacă folosește transformarea.",
                "incorrectExplanation": "Trendul și sezonalitatea sînt tratate de componenta de trend și de componentele sezoniere trigonometrice, nu de transformarea Box-Cox, iar transformarea adaugă un pas de estimare, nu economisește timp."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Trend types in Prophet",
                "text": "Prophet's default trend is piecewise linear. When should logistic growth be used instead?",
                "options": [
                    "When the data are seasonal",
                    "When there is a natural upper limit (a saturating capacity)",
                    "When the data contain missing values",
                    "When faster fitting is needed"
                ],
                "correctExplanation": "Logistic growth $g(t) = C/(1 + \\exp(-k(t-m)))$ saturates at the capacity $C$ (market size, adoption ceiling, population limit); in Prophet it must be requested explicitly (growth='logistic') together with a cap column.",
                "incorrectExplanation": "Seasonality and missing values are handled by other parts of Prophet regardless of the trend type, and logistic growth is not faster to fit; it is chosen only when the series cannot grow beyond a known limit."
            },
            "ro": {
                "title": "Tipuri de trend în Prophet",
                "text": "Trendul implicit din Prophet este liniar pe porțiuni. Cînd trebuie folosită în schimb creșterea logistică?",
                "options": [
                    "Cînd datele sînt sezoniere",
                    "Cînd există o limită superioară naturală (o capacitate de saturare)",
                    "Cînd datele conțin valori lipsă",
                    "Cînd este necesară o estimare mai rapidă"
                ],
                "correctExplanation": "Creșterea logistică $g(t) = C/(1 + \\exp(-k(t-m)))$ se saturează la capacitatea $C$ (dimensiunea pieței, plafonul de adopție, o limită a populației); în Prophet trebuie cerută explicit (growth='logistic'), împreună cu o coloană cap.",
                "incorrectExplanation": "Sezonalitatea și valorile lipsă sînt tratate de alte părți ale modelului Prophet, indiferent de tipul trendului, iar creșterea logistică nu se estimează mai rapid; ea se alege doar cînd seria nu poate depăși o limită cunoscută."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Fourier terms and parameter count",
                "text": "For weekly seasonality ($m = 7$) with $K = 3$ Fourier harmonics, how many seasonal coefficients are added?",
                "options": [
                    "3",
                    "6",
                    "7",
                    "14"
                ],
                "correctExplanation": "Each harmonic $k$ contributes $\\sin(2\\pi k t/m)$ and $\\cos(2\\pi k t/m)$, each with its own coefficient: $2K = 6$. Since $K = 3 \\le m/2$, this is also the most flexible weekly pattern, equivalent to 6 day-of-week dummies.",
                "incorrectExplanation": "Counting one coefficient per harmonic forgets the sine-cosine pair; 7 or 14 would exceed the $m - 1 = 6$ free parameters any weekly pattern can have."
            },
            "ro": {
                "title": "Termeni Fourier și numărul de parametri",
                "text": "Pentru o sezonalitate săptămînală ($m = 7$) cu $K = 3$ armonici Fourier, cîți coeficienți sezonieri se adaugă?",
                "options": [
                    "3",
                    "6",
                    "7",
                    "14"
                ],
                "correctExplanation": "Fiecare armonică $k$ contribuie cu $\\sin(2\\pi k t/m)$ și $\\cos(2\\pi k t/m)$, fiecare cu propriul coeficient: $2K = 6$. Deoarece $K = 3 \\le m/2$, acesta este și cel mai flexibil tipar săptămînal, echivalent cu 6 variabile dummy pentru zilele săptămînii.",
                "incorrectExplanation": "Un singur coeficient pentru fiecare armonică ignoră perechea sinus-cosinus; 7 sau 14 ar depăși cei $m - 1 = 6$ parametri liberi pe care îi poate avea orice tipar săptămînal."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Missing data in Prophet",
                "text": "How does Prophet handle missing values in the series?",
                "options": [
                    "It requires imputation before fitting",
                    "It simply fits on the observed dates, so missing values drop out of the likelihood",
                    "It interpolates them linearly before fitting",
                    "It fails if any value is missing"
                ],
                "correctExplanation": "Prophet treats time as a regressor (a curve-fitting model), so the rows with missing values are just absent from the fit; irregular timestamps are handled in the same way, and the model still predicts at the missing dates.",
                "incorrectExplanation": "No imputation or interpolation step is needed and missing values do not cause an error. This differs from models whose recursions need equally spaced data, although SARIMA in state space form can also handle gaps through the Kalman filter."
            },
            "ro": {
                "title": "Date lipsă în Prophet",
                "text": "Cum tratează Prophet valorile lipsă din serie?",
                "options": [
                    "Necesită imputarea lor înainte de estimare",
                    "Estimează pur și simplu pe datele observate, deci valorile lipsă nu intră în verosimilitate",
                    "Le interpolează liniar înainte de estimare",
                    "Eșuează dacă lipsește vreo valoare"
                ],
                "correctExplanation": "Prophet tratează timpul ca regresor (este un model de ajustare a curbelor), deci rîndurile cu valori lipsă pur și simplu nu intră în estimare; momentele de timp neregulate sînt tratate la fel, iar modelul poate prognoza și la datele lipsă.",
                "incorrectExplanation": "Nu este necesară imputarea sau interpolarea, iar valorile lipsă nu produc o eroare. Diferența față de modelele ale căror recursii cer date echidistante este reală, deși SARIMA în spațiul stărilor poate trata și el golurile prin filtrul Kalman."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Holiday effects in Prophet",
                "text": "How does Prophet handle holiday effects?",
                "options": [
                    "It detects holidays automatically from the data",
                    "The user supplies the holiday dates (optionally with windows of days around them) and Prophet estimates their effects",
                    "Holidays are modelled through extra Fourier terms at holiday frequencies",
                    "The user must specify the size of each holiday effect"
                ],
                "correctExplanation": "The user provides a holiday data frame (or a built-in country calendar) with optional lower and upper windows; each holiday becomes an indicator regressor whose coefficient is estimated, with a prior that shrinks small effects. By default the effects are additive, like the seasonality.",
                "incorrectExplanation": "Prophet does not discover holidays by itself, holidays are indicator regressors rather than Fourier terms, and their magnitudes are estimated from the data, not fixed by the user."
            },
            "ro": {
                "title": "Efectele sărbătorilor în Prophet",
                "text": "Cum tratează Prophet efectele sărbătorilor?",
                "options": [
                    "Detectează automat sărbătorile din date",
                    "Utilizatorul furnizează datele sărbătorilor (opțional cu o fereastră de zile în jurul lor), iar Prophet estimează efectele",
                    "Sărbătorile sînt modelate prin termeni Fourier suplimentari, la frecvențele sărbătorilor",
                    "Utilizatorul trebuie să specifice mărimea fiecărui efect"
                ],
                "correctExplanation": "Utilizatorul furnizează un tabel cu sărbători (sau un calendar național predefinit), cu ferestre opționale înainte și după; fiecare sărbătoare devine un regresor indicator al cărui coeficient este estimat, cu o distribuție a priori care restrînge efectele mici. Implicit, efectele sînt aditive, ca sezonalitatea.",
                "incorrectExplanation": "Prophet nu descoperă singur sărbătorile, acestea sînt regresori indicatori, nu termeni Fourier, iar mărimea efectelor este estimată din date, nu fixată de utilizator."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Model selection in TBATS",
                "text": "How does TBATS choose its configuration (Box-Cox or not, trend and damping, ARMA orders, number of harmonics), and what does this imply for computing time?",
                "options": [
                    "By cross-validation on a hold-out set, which makes it fast",
                    "By fitting many candidate configurations and minimising AIC, which can make it slow on long series with long seasonal periods",
                    "By spectral analysis of the periodogram, with no estimation needed",
                    "By fixed default values, so no selection takes place"
                ],
                "correctExplanation": "TBATS fits alternative models (with and without Box-Cox, trend, damping and ARMA errors, with different $K$ for each period) and keeps the one with the lowest AIC; $\\lambda$ is estimated as well. The many maximum-likelihood fits make it slow on long high-frequency series, where Prophet is usually faster.",
                "incorrectExplanation": "TBATS does not use cross-validation, the periodogram or fixed defaults to pick its components; its automatic search over configurations is what makes it convenient but computationally heavy."
            },
            "ro": {
                "title": "Selecția modelului în TBATS",
                "text": "Cum își alege TBATS configurația (cu sau fără Box-Cox, trend și amortizare, ordinele ARMA, numărul de armonici) și ce implică aceasta pentru timpul de calcul?",
                "options": [
                    "Prin validare încrucișată pe un set de test, ceea ce îl face rapid",
                    "Prin estimarea multor configurații candidate și minimizarea AIC, ceea ce îl poate face lent pe serii lungi cu perioade sezoniere lungi",
                    "Prin analiza spectrală a periodogramei, fără a fi necesară estimarea",
                    "Prin valori implicite fixe, deci nu are loc nicio selecție"
                ],
                "correctExplanation": "TBATS estimează modele alternative (cu și fără Box-Cox, trend, amortizare și erori ARMA, cu valori diferite ale lui $K$ pentru fiecare perioadă) și îl păstrează pe cel cu AIC minim; $\\lambda$ este și el estimat. Numeroasele estimări de verosimilitate maximă îl fac lent pe serii lungi de frecvență înaltă, unde Prophet este de obicei mai rapid.",
                "incorrectExplanation": "TBATS nu folosește validarea încrucișată, periodograma sau valori fixe pentru a-și alege componentele; căutarea automată printre configurații îl face comod, dar costisitor din punct de vedere computațional."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Cross-validation in Prophet",
                "text": "Prophet's built-in cross_validation function uses:",
                "options": [
                    "Standard k-fold cross-validation with random folds",
                    "Rolling-origin cross-validation: simulated historical forecasts from successive cutoffs",
                    "Leave-one-out cross-validation",
                    "A single random hold-out sample"
                ],
                "correctExplanation": "cross_validation fits the model on the data up to each cutoff and forecasts the next horizon (parameters initial, period and horizon), always training on the past and testing on the future; performance_metrics then summarises the errors by horizon.",
                "incorrectExplanation": "Random folds, leave-one-out and random hold-outs mix past and future observations, which leaks information and gives over-optimistic accuracy for time series."
            },
            "ro": {
                "title": "Validarea încrucișată în Prophet",
                "text": "Funcția cross_validation din Prophet folosește:",
                "options": [
                    "Validare încrucișată k-fold standard, cu partiții aleatoare",
                    "Validare încrucișată cu origine mobilă: prognoze istorice simulate din momente de tăiere succesive",
                    "Validare încrucișată leave-one-out",
                    "Un singur eșantion de test ales aleator"
                ],
                "correctExplanation": "cross_validation estimează modelul pe datele pînă la fiecare moment de tăiere și prognozează orizontul următor (parametrii initial, period și horizon), antrenînd mereu pe trecut și testînd pe viitor; performance_metrics rezumă apoi erorile în funcție de orizont.",
                "incorrectExplanation": "Partițiile aleatoare, leave-one-out și eșantioanele de test aleatoare amestecă observații din trecut și din viitor, ceea ce scurge informație și dă o acuratețe prea optimistă pentru seriile de timp."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Changepoint prior scale",
                "text": "In Prophet, increasing changepoint_prior_scale makes the trend:",
                "options": [
                    "Smoother and more stable",
                    "More flexible and more responsive to changes",
                    "Strictly linear, without changepoints",
                    "Logistic instead of linear"
                ],
                "correctExplanation": "changepoint_prior_scale (default 0.05) is the scale of the Laplace prior on the slope changes. A larger value allows larger rate changes, so the trend follows the data more closely (risking overfitting and wider intervals); a smaller value gives a smoother trend.",
                "incorrectExplanation": "A smaller value, not a larger one, smooths the trend. The parameter does not switch off changepoints and does not change the growth type, which is set by the growth argument."
            },
            "ro": {
                "title": "Parametrul changepoint_prior_scale",
                "text": "În Prophet, creșterea valorii changepoint_prior_scale face trendul:",
                "options": [
                    "Mai neted și mai stabil",
                    "Mai flexibil și mai sensibil la schimbări",
                    "Strict liniar, fără changepoints",
                    "Logistic în loc de liniar"
                ],
                "correctExplanation": "changepoint_prior_scale (implicit 0,05) este scala distribuției a priori Laplace pentru variațiile pantei. O valoare mai mare permite schimbări de pantă mai mari, deci trendul urmărește mai îndeaproape datele (cu risc de supraajustare și intervale mai largi); o valoare mai mică dă un trend mai neted.",
                "incorrectExplanation": "O valoare mai mică, nu una mai mare, netezește trendul. Parametrul nu elimină changepoints și nu schimbă tipul de creștere, stabilit prin argumentul growth."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Yearly seasonality in Prophet",
                "text": "How does Prophet model yearly seasonality by default?",
                "options": [
                    "With seasonal dummy variables",
                    "With a Fourier series of order 10 (10 sine-cosine pairs)",
                    "With a spline function",
                    "With exponential smoothing"
                ],
                "correctExplanation": "Prophet represents each seasonality by a truncated Fourier series $\\sum_{k=1}^{K}[a_k\\cos(2\\pi k t/P) + b_k\\sin(2\\pi k t/P)]$; the default order is $K = 10$ for yearly ($P = 365.25$), 3 for weekly and 4 for daily seasonality.",
                "incorrectExplanation": "Dummies would require 364 parameters for daily data with yearly seasonality and cannot handle a non-integer period; Prophet uses neither splines nor exponential smoothing for seasonality."
            },
            "ro": {
                "title": "Sezonalitatea anuală în Prophet",
                "text": "Cum modelează implicit Prophet sezonalitatea anuală?",
                "options": [
                    "Prin variabile dummy sezoniere",
                    "Printr-o serie Fourier de ordin 10 (10 perechi sinus-cosinus)",
                    "Printr-o funcție spline",
                    "Prin nivelare exponențială"
                ],
                "correctExplanation": "Prophet reprezintă fiecare sezonalitate printr-o serie Fourier trunchiată, $\\sum_{k=1}^{K}[a_k\\cos(2\\pi k t/P) + b_k\\sin(2\\pi k t/P)]$; ordinul implicit este $K = 10$ pentru sezonalitatea anuală ($P = 365{,}25$), 3 pentru cea săptămînală și 4 pentru cea zilnică.",
                "incorrectExplanation": "Variabilele dummy ar cere 364 de parametri pentru date zilnice cu sezonalitate anuală și nu pot trata o perioadă neîntreagă; Prophet nu folosește pentru sezonalitate nici funcții spline, nici nivelare exponențială."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ARMA errors in TBATS",
                "text": "What role does the ARMA component play in TBATS?",
                "options": [
                    "It models the trend",
                    "It captures autocorrelation left in the errors after trend and seasonality",
                    "It detects changepoints",
                    "It normalises the data"
                ],
                "correctExplanation": "After the level, trend and trigonometric seasonal states, the error $d_t$ may still be autocorrelated; TBATS models it as ARMA$(p,q)$ (orders chosen automatically), which improves short-horizon forecasts.",
                "incorrectExplanation": "The trend has its own (damped) component, TBATS has no changepoints (that is a Prophet feature), and variance stabilisation is done by Box-Cox, not by ARMA."
            },
            "ro": {
                "title": "Erorile ARMA în TBATS",
                "text": "Ce rol are componenta ARMA în TBATS?",
                "options": [
                    "Modelează trendul",
                    "Captează autocorelația rămasă în erori după trend și sezonalitate",
                    "Detectează changepoints",
                    "Normalizează datele"
                ],
                "correctExplanation": "După stările de nivel, trend și sezonalitate trigonometrică, eroarea $d_t$ poate fi încă autocorelată; TBATS o modelează ca ARMA$(p,q)$ (cu ordine alese automat), ceea ce îmbunătățește prognozele pe orizonturi scurte.",
                "incorrectExplanation": "Trendul are propria componentă (eventual amortizată), TBATS nu are changepoints (acestea sînt specifice Prophet), iar stabilizarea varianței se face prin Box-Cox, nu prin ARMA."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Seasonal periods in TBATS",
                "text": "How many seasonal periods can TBATS handle at the same time?",
                "options": [
                    "Exactly two",
                    "At most five",
                    "In principle any number, limited in practice by the data and computing time; non-integer periods are allowed",
                    "Only one, like SARIMA"
                ],
                "correctExplanation": "Each seasonal period gets its own set of trigonometric states, so TBATS can combine several periods (for example 24, 168 and 8766 hours) and even non-integer periods such as 52.18 weeks per year. The practical limit is the number of parameters and the estimation time.",
                "incorrectExplanation": "There is no fixed limit of two or five periods, and handling more than one period is precisely what distinguishes TBATS from SARIMA."
            },
            "ro": {
                "title": "Perioade sezoniere în TBATS",
                "text": "Cîte perioade sezoniere poate trata simultan TBATS?",
                "options": [
                    "Exact două",
                    "Cel mult cinci",
                    "În principiu oricîte, limitat în practică de date și de timpul de calcul; sînt permise și perioade neîntregi",
                    "Doar una, ca SARIMA"
                ],
                "correctExplanation": "Fiecare perioadă sezonieră are propriul set de stări trigonometrice, deci TBATS poate combina mai multe perioade (de exemplu 24, 168 și 8766 de ore) și chiar perioade neîntregi, precum 52,18 săptămîni pe an. Limita practică este dată de numărul de parametri și de timpul de estimare.",
                "incorrectExplanation": "Nu există o limită fixă de două sau cinci perioade, iar tratarea mai multor perioade este tocmai ceea ce deosebește TBATS de SARIMA."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Custom seasonality in Prophet",
                "text": "How is a custom seasonality (for example, a monthly cycle of 30.5 days) added to a Prophet model?",
                "options": [
                    "By adding 30 dummy variables to the data frame",
                    "With add_seasonality(name, period=30.5, fourier_order=K), which adds a Fourier series with that period",
                    "It is not possible: Prophet supports only daily, weekly and yearly seasonality",
                    "By differencing the series at lag 30 before fitting"
                ],
                "correctExplanation": "add_seasonality accepts any period, including non-integer ones, and a Fourier order that controls its flexibility; it can also be conditional (for example, a weekly pattern that differs in term time and holidays).",
                "incorrectExplanation": "Prophet does not use dummies or differencing for seasonality, and it is not limited to the three built-in seasonalities."
            },
            "ro": {
                "title": "Sezonalitate personalizată în Prophet",
                "text": "Cum se adaugă unui model Prophet o sezonalitate personalizată (de exemplu, un ciclu lunar de 30,5 zile)?",
                "options": [
                    "Prin adăugarea a 30 de variabile dummy în tabelul de date",
                    "Cu add_seasonality(name, period=30.5, fourier_order=K), care adaugă o serie Fourier cu perioada respectivă",
                    "Nu este posibil: Prophet acceptă doar sezonalitate zilnică, săptămînală și anuală",
                    "Prin diferențierea seriei la lag-ul 30 înainte de estimare"
                ],
                "correctExplanation": "add_seasonality acceptă orice perioadă, inclusiv neîntreagă, și un ordin Fourier care îi controlează flexibilitatea; sezonalitatea poate fi și condiționată (de exemplu, un tipar săptămînal diferit în timpul semestrului și în vacanță).",
                "incorrectExplanation": "Prophet nu folosește variabile dummy sau diferențiere pentru sezonalitate și nu este limitat la cele trei sezonalități predefinite."
            }
        }
    ]
};
