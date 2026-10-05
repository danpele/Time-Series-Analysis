// ============================================================
// Chapter 3 quiz bank: Unit roots and ARIMA models (EN + RO)
// 31 questions ported from the 2025/2026 site; 20 drawn per attempt.
// The bank grows to 24 questions when the chapter is rebuilt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['arima'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 3,
            "en": {
                "title": "Consequences of nonstationarity",
                "text": "Which of the following is NOT a consequence of nonstationarity in a time series?",
                "options": [
                    "An OLS regression between nonstationary series can be spurious",
                    "Sample moments (mean, variance, ACF) no longer estimate constant population parameters",
                    "Standard $t$ and $F$ inference can become invalid",
                    "The autocorrelation function (ACF) decays quickly to zero"
                ],
                "correctExplanation": "A quickly decaying ACF is a property of STATIONARY series. For a nonstationary (e.g. I(1)) series the sample ACF decays very slowly.",
                "incorrectExplanation": "Spurious regression, sample moments that do not settle down and invalid standard inference are genuine consequences of nonstationarity. A fast-decaying ACF points the other way: it indicates stationarity."
            },
            "ro": {
                "title": "Consecințele nestaționarității",
                "text": "Care dintre următoarele NU este o consecință a nestaționarității unei serii de timp?",
                "options": [
                    "Regresia OLS între serii nestaționare poate fi o regresie falsă",
                    "Momentele de selecție (medie, varianță, ACF) nu mai estimează parametri constanți ai populației",
                    "Inferența statistică standard (testele $t$ și $F$) poate deveni invalidă",
                    "Funcția de autocorelație (ACF) scade rapid către zero"
                ],
                "correctExplanation": "O ACF care scade rapid este o proprietate a seriilor STAȚIONARE. La o serie nestaționară (de exemplu I(1)), ACF-ul de selecție scade foarte lent.",
                "incorrectExplanation": "Regresia falsă, momentele de selecție care nu se stabilizează și inferența standard invalidă sînt consecințe reale ale nestaționarității. O ACF care scade rapid indică, dimpotrivă, staționaritate."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Random walk variance",
                "text": "For a random walk $Y_t = Y_{t-1} + \\varepsilon_t$ with $Y_0 = 0$ and $\\varepsilon_t \\sim WN(0, \\sigma^2)$, what is $\\text{Var}(Y_t)$?",
                "options": [
                    "$\\sigma^2$ (constant)",
                    "$\\sigma^{2t}$ (grows exponentially)",
                    "$t \\cdot \\sigma^2$ (grows linearly in time)",
                    "$\\sigma^2 / t$ (decreases in time)"
                ],
                "correctExplanation": "$Y_t = \\sum_{i=1}^t \\varepsilon_i$, so $\\text{Var}(Y_t) = t \\cdot \\sigma^2$. The variance grows linearly with $t$, hence the process is nonstationary.",
                "incorrectExplanation": "The shocks are uncorrelated, so their variances add up: $t$ terms of variance $\\sigma^2$ give $t\\sigma^2$, neither constant nor exponential nor decreasing. A constant variance such as $\\sigma^2/(1-\\phi^2)$ exists only for a stationary AR(1) with $|\\phi|<1$."
            },
            "ro": {
                "title": "Varianța mersului aleator",
                "text": "Pentru un mers aleator $Y_t = Y_{t-1} + \\varepsilon_t$ cu $Y_0 = 0$ și $\\varepsilon_t \\sim WN(0, \\sigma^2)$, cît este varianța $\\text{Var}(Y_t)$?",
                "options": [
                    "$\\sigma^2$ (constantă)",
                    "$\\sigma^{2t}$ (crește exponențial)",
                    "$t \\cdot \\sigma^2$ (crește liniar în timp)",
                    "$\\sigma^2 / t$ (scade în timp)"
                ],
                "correctExplanation": "$Y_t = \\sum_{i=1}^t \\varepsilon_i$, deci $\\text{Var}(Y_t) = t \\cdot \\sigma^2$. Varianța crește liniar cu $t$, deci procesul este nestaționar.",
                "incorrectExplanation": "Șocurile sînt necorelate, deci varianțele lor se adună: $t$ termeni de varianță $\\sigma^2$ dau $t\\sigma^2$, nici constantă, nici exponențială, nici descrescătoare. O varianță constantă, de tipul $\\sigma^2/(1-\\phi^2)$, există doar pentru un AR(1) staționar cu $|\\phi|<1$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Deterministic and stochastic trend",
                "text": "What is the main difference between a deterministic trend and a stochastic trend (unit root)?",
                "options": [
                    "A deterministic trend requires differencing, a stochastic trend requires regression on time",
                    "Shocks have temporary effects around a deterministic trend and permanent effects in a stochastic trend",
                    "A stochastic trend has constant variance, a deterministic trend does not",
                    "There is no practical difference between the two"
                ],
                "correctExplanation": "Trend-stationary series return to the trend line after a shock (temporary effect). With a unit root the shock is accumulated into the level and never dies out (permanent effect), as in a random walk.",
                "incorrectExplanation": "The prescriptions are reversed: a deterministic trend is removed by regressing on time (detrending), a stochastic trend by differencing. A stochastic trend has a variance that grows with time, and confusing the two leads to wrong forecasts and wrong inference."
            },
            "ro": {
                "title": "Trend determinist și trend stochastic",
                "text": "Care este diferența principală dintre un trend determinist și un trend stochastic (rădăcină unitară)?",
                "options": [
                    "Trendul determinist necesită diferențiere, iar cel stochastic necesită regresie pe timp",
                    "Șocurile au efecte temporare în jurul unui trend determinist și efecte permanente într-un trend stochastic",
                    "Trendul stochastic are varianță constantă, iar cel determinist nu",
                    "Nu există nicio diferență practică între cele două"
                ],
                "correctExplanation": "O serie staționară în jurul unui trend revine la linia trendului după un șoc (efect temporar). Cu rădăcină unitară, șocul se acumulează în nivel și nu se stinge niciodată (efect permanent), ca la mersul aleator.",
                "incorrectExplanation": "Rețetele sînt inversate: trendul determinist se elimină prin regresie pe timp, iar trendul stochastic prin diferențiere. Trendul stochastic are o varianță care crește în timp, iar confuzia dintre cele două duce la prognoze și inferențe greșite."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Integrated processes",
                "text": "If a time series $Y_t$ is $I(2)$, what does this mean?",
                "options": [
                    "The series is stationary",
                    "The first difference $\\Delta Y_t$ is stationary",
                    "The series must be differenced twice to become stationary",
                    "The series has two autoregressive parameters"
                ],
                "correctExplanation": "$Y_t \\sim I(d)$ means that $d$ differences are needed to reach stationarity. For $I(2)$, $\\Delta^2 Y_t$ is stationary, while $\\Delta Y_t$ is not.",
                "incorrectExplanation": "A stationary series is $I(0)$, and a series whose first difference is stationary is $I(1)$. The order of integration counts differences, not AR lags: an AR(2) can perfectly well be $I(0)$."
            },
            "ro": {
                "title": "Procese integrate",
                "text": "Dacă o serie de timp $Y_t$ este $I(2)$, ce înseamnă acest lucru?",
                "options": [
                    "Seria este staționară",
                    "Prima diferență $\\Delta Y_t$ este staționară",
                    "Seria trebuie diferențiată de două ori pentru a deveni staționară",
                    "Seria are doi parametri autoregresivi"
                ],
                "correctExplanation": "$Y_t \\sim I(d)$ înseamnă că sînt necesare $d$ diferențieri pentru a obține staționaritate. Pentru $I(2)$, $\\Delta^2 Y_t$ este staționară, dar $\\Delta Y_t$ nu este.",
                "incorrectExplanation": "O serie staționară este $I(0)$, iar o serie a cărei primă diferență este staționară este $I(1)$. Ordinul de integrare numără diferențieri, nu lag-uri AR: un AR(2) poate fi foarte bine $I(0)$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Order of integration in practice",
                "text": "How is the order of integration $d$ of a time series determined in practice?",
                "options": [
                    "By counting the significant spikes in the ACF",
                    "By applying unit root tests (ADF/KPSS) repeatedly, differencing until the series is stationary",
                    "By computing the sample variance",
                    "By estimating every possible ARIMA model and keeping the largest $R^2$"
                ],
                "correctExplanation": "Test the original series with ADF/KPSS; if it is nonstationary, difference it and test again, until the tests indicate stationarity. The number of differences taken is $d$.",
                "incorrectExplanation": "Counting ACF spikes identifies the MA order $q$, not $d$; a sample variance alone says nothing about unit roots; and $R^2$ comparisons across differenced and undifferenced series are meaningless. The order $d$ comes from sequential unit root testing."
            },
            "ro": {
                "title": "Ordinul de integrare în practică",
                "text": "Cum se determină în practică ordinul de integrare $d$ al unei serii de timp?",
                "options": [
                    "Prin numărarea vîrfurilor semnificative din ACF",
                    "Prin aplicarea repetată a testelor de rădăcină unitară (ADF/KPSS), diferențiind pînă cînd seria devine staționară",
                    "Prin calcularea varianței de selecție",
                    "Prin estimarea tuturor modelelor ARIMA posibile și păstrarea celui cu $R^2$ maxim"
                ],
                "correctExplanation": "Se aplică ADF/KPSS seriei inițiale; dacă este nestaționară, se diferențiază și se testează din nou, pînă cînd testele indică staționaritate. Numărul de diferențieri efectuate este $d$.",
                "incorrectExplanation": "Numărul vîrfurilor din ACF identifică ordinul MA $q$, nu $d$; varianța de selecție singură nu spune nimic despre rădăcini unitare; iar compararea $R^2$ între serii diferențiate și nediferențiate nu are sens. Ordinul $d$ rezultă din testarea secvențială a rădăcinii unitare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Typical order of integration of macro data",
                "text": "What is the typical order of integration of most macroeconomic series in levels (GDP, consumption, price indices)?",
                "options": [
                    "$I(0)$: stationary in levels",
                    "$I(1)$: stationary after one difference",
                    "$I(2)$ as a rule, for every macroeconomic series",
                    "$I(3)$ or higher"
                ],
                "correctExplanation": "Most macroeconomic series in levels (or logs) behave like $I(1)$: their growth rates are stationary. Some price levels in high-inflation periods may be close to $I(2)$, but $I(1)$ is the typical case.",
                "incorrectExplanation": "Levels of GDP or prices trend and wander, so they are not $I(0)$. $I(2)$ appears only occasionally (e.g. price levels when inflation itself is persistent), and orders of three or more are practically never found."
            },
            "ro": {
                "title": "Ordinul de integrare tipic al datelor macroeconomice",
                "text": "Care este ordinul de integrare tipic al majorității seriilor macroeconomice în niveluri (PIB, consum, indici de prețuri)?",
                "options": [
                    "$I(0)$: staționară în niveluri",
                    "$I(1)$: staționară după o diferențiere",
                    "$I(2)$, ca regulă, pentru orice serie macroeconomică",
                    "$I(3)$ sau mai mare"
                ],
                "correctExplanation": "Majoritatea seriilor macroeconomice în niveluri (sau în logaritmi) se comportă ca $I(1)$: ratele lor de creștere sînt staționare. Unele niveluri de prețuri din perioade cu inflație ridicată pot fi apropiate de $I(2)$, dar cazul tipic este $I(1)$.",
                "incorrectExplanation": "Nivelurile PIB-ului sau ale prețurilor au trend și rătăcesc, deci nu sînt $I(0)$. $I(2)$ apare doar ocazional (de exemplu nivelul prețurilor cînd inflația însăși este persistentă), iar ordine de trei sau mai mult practic nu se întîlnesc."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ACF of an I(1) series",
                "text": "What is characteristic of the sample ACF of an I(1) series?",
                "options": [
                    "It cuts off sharply after lag 1",
                    "It decays very slowly, staying close to 1 for many lags",
                    "It shows no significant lags",
                    "It alternates between positive and negative values"
                ],
                "correctExplanation": "A very slow, almost linear decay of the sample ACF from values near 1 is the classic visual sign of a unit root.",
                "incorrectExplanation": "A cut-off after lag 1 suggests an MA(1); no significant lags suggests white noise; alternating signs suggest an AR(1) with negative coefficient. None of these is the signature of an I(1) series."
            },
            "ro": {
                "title": "ACF-ul unei serii I(1)",
                "text": "Ce caracterizează ACF-ul de selecție al unei serii I(1)?",
                "options": [
                    "Se anulează brusc după lag-ul 1",
                    "Scade foarte lent, rămînînd aproape de 1 pentru multe lag-uri",
                    "Nu are niciun lag semnificativ",
                    "Alternează între valori pozitive și negative"
                ],
                "correctExplanation": "O scădere foarte lentă, aproape liniară, a ACF-ului de selecție de la valori apropiate de 1 este semnul vizual clasic al unei rădăcini unitare.",
                "incorrectExplanation": "Anularea după lag-ul 1 sugerează un MA(1); lipsa lag-urilor semnificative sugerează zgomot alb; alternanța semnelor sugerează un AR(1) cu coeficient negativ. Niciuna nu este semnătura unei serii I(1)."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Second difference",
                "text": "What is $(1-L)^2 Y_t$ written out?",
                "options": [
                    "$Y_t - Y_{t-1}$",
                    "$Y_t - 2Y_{t-1} + Y_{t-2}$",
                    "$Y_t + 2Y_{t-1} + Y_{t-2}$",
                    "$Y_t - Y_{t-2}$"
                ],
                "correctExplanation": "$(1-L)^2 = 1 - 2L + L^2$, so $\\Delta^2 Y_t = Y_t - 2Y_{t-1} + Y_{t-2}$.",
                "incorrectExplanation": "$Y_t - Y_{t-1}$ is only the first difference, $Y_t - Y_{t-2}$ is the lag-2 difference $(1-L^2)Y_t$, and the plus signs come from $(1+L)^2$. Squaring $(1-L)$ gives the middle term $-2L$."
            },
            "ro": {
                "title": "A doua diferență",
                "text": "Cum se scrie desfășurat $(1-L)^2 Y_t$?",
                "options": [
                    "$Y_t - Y_{t-1}$",
                    "$Y_t - 2Y_{t-1} + Y_{t-2}$",
                    "$Y_t + 2Y_{t-1} + Y_{t-2}$",
                    "$Y_t - Y_{t-2}$"
                ],
                "correctExplanation": "$(1-L)^2 = 1 - 2L + L^2$, deci $\\Delta^2 Y_t = Y_t - 2Y_{t-1} + Y_{t-2}$.",
                "incorrectExplanation": "$Y_t - Y_{t-1}$ este doar prima diferență, $Y_t - Y_{t-2}$ este diferența la lag 2, $(1-L^2)Y_t$, iar semnele plus provin din $(1+L)^2$. Ridicarea la pătrat a lui $(1-L)$ dă termenul din mijloc $-2L$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Overdifferencing",
                "text": "What happens if an $I(1)$ random walk is differenced twice (overdifferencing)?",
                "options": [
                    "We obtain pure white noise",
                    "We obtain a trend-stationary process",
                    "We obtain an MA(1) with $\\theta_1 = -1$, on the boundary of non-invertibility",
                    "We obtain a stationary AR(2) process"
                ],
                "correctExplanation": "$\\Delta Y_t = \\varepsilon_t$, so $\\Delta^2 Y_t = \\varepsilon_t - \\varepsilon_{t-1}$: an MA(1) with a unit MA root ($\\theta_1 = -1$). It has lag-1 autocorrelation $-0.5$, i.e. artificial negative autocorrelation, and it degrades estimation and forecasts.",
                "incorrectExplanation": "White noise is obtained after ONE difference; the second difference adds the term $-\\varepsilon_{t-1}$. The result has no trend and no AR structure; its hallmark is a lag-1 autocorrelation close to $-0.5$ and an MA root on the unit circle."
            },
            "ro": {
                "title": "Supradiferențierea",
                "text": "Ce se întîmplă dacă un mers aleator $I(1)$ este diferențiat de două ori (supradiferențiere)?",
                "options": [
                    "Obținem un zgomot alb pur",
                    "Obținem un proces staționar în jurul unui trend",
                    "Obținem un MA(1) cu $\\theta_1 = -1$, la limita neinvertibilității",
                    "Obținem un proces AR(2) staționar"
                ],
                "correctExplanation": "$\\Delta Y_t = \\varepsilon_t$, deci $\\Delta^2 Y_t = \\varepsilon_t - \\varepsilon_{t-1}$: un MA(1) cu rădăcină MA unitară ($\\theta_1 = -1$). Autocorelația la lag 1 este $-0{,}5$, adică o autocorelație negativă artificială, care degradează estimarea și prognozele.",
                "incorrectExplanation": "Zgomotul alb se obține după O SINGURĂ diferențiere; a doua diferență adaugă termenul $-\\varepsilon_{t-1}$. Rezultatul nu are trend și nici structură AR; semnul distinctiv este o autocorelație la lag 1 apropiată de $-0{,}5$ și o rădăcină MA pe cercul unitate."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ADF test hypotheses",
                "text": "What are the hypotheses of the Augmented Dickey-Fuller (ADF) test?",
                "options": [
                    "$H_0$: the series is stationary; $H_1$: the series has a unit root",
                    "$H_0$: the series has a unit root; $H_1$: the series is stationary",
                    "$H_0$: the mean is zero; $H_1$: the mean is nonzero",
                    "$H_0$: the variance is constant; $H_1$: the variance changes over time"
                ],
                "correctExplanation": "In $\\Delta Y_t = \\alpha + \\gamma Y_{t-1} + \\sum_j \\delta_j \\Delta Y_{t-j} + \\varepsilon_t$, ADF tests $H_0: \\gamma = 0$ (unit root) against $H_1: \\gamma < 0$ (stationarity). Rejecting $H_0$ is evidence of stationarity.",
                "incorrectExplanation": "Stationarity as the null is the KPSS design, the reverse of ADF. ADF says nothing directly about the level of the mean or about changing variance (the latter is the domain of ARCH tests)."
            },
            "ro": {
                "title": "Ipotezele testului ADF",
                "text": "Care sînt ipotezele testului Augmented Dickey-Fuller (ADF)?",
                "options": [
                    "$H_0$: seria este staționară; $H_1$: seria are rădăcină unitară",
                    "$H_0$: seria are rădăcină unitară; $H_1$: seria este staționară",
                    "$H_0$: media este zero; $H_1$: media este nenulă",
                    "$H_0$: varianța este constantă; $H_1$: varianța se modifică în timp"
                ],
                "correctExplanation": "În $\\Delta Y_t = \\alpha + \\gamma Y_{t-1} + \\sum_j \\delta_j \\Delta Y_{t-j} + \\varepsilon_t$, ADF testează $H_0: \\gamma = 0$ (rădăcină unitară) față de $H_1: \\gamma < 0$ (staționaritate). Respingerea lui $H_0$ indică staționaritate.",
                "incorrectExplanation": "Staționaritatea ca ipoteză nulă este construcția testului KPSS, inversa ADF. ADF nu spune nimic direct despre nivelul mediei sau despre varianța variabilă în timp (aceasta din urmă ține de testele ARCH)."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ADF rejection rule",
                "text": "In the ADF test, $H_0$ (unit root) is rejected when:",
                "options": [
                    "The test statistic is above the critical value (closer to zero)",
                    "The test statistic is below the critical value (more negative)",
                    "The $p$-value is above 0.05",
                    "The test statistic exceeds 1.96 in absolute value"
                ],
                "correctExplanation": "ADF is a left-tailed test with Dickey-Fuller critical values (e.g. about $-2.86$ at 5% with a constant): the unit root is rejected only when the statistic is sufficiently negative.",
                "incorrectExplanation": "A statistic closer to zero than the critical value means we do not reject; a large $p$-value also means no rejection. The Normal value 1.96 does not apply: under the unit-root null the statistic follows the non-standard Dickey-Fuller distribution, with more negative critical values."
            },
            "ro": {
                "title": "Regula de respingere în testul ADF",
                "text": "În testul ADF, $H_0$ (rădăcină unitară) este respinsă atunci cînd:",
                "options": [
                    "Statistica de test este peste valoarea critică (mai aproape de zero)",
                    "Statistica de test este sub valoarea critică (mai negativă)",
                    "$p$-valoarea este mai mare decît 0,05",
                    "Statistica de test depășește 1,96 în valoare absolută"
                ],
                "correctExplanation": "ADF este un test unilateral la stînga, cu valori critice Dickey-Fuller (de exemplu aproximativ $-2{,}86$ la 5% cu constantă): rădăcina unitară este respinsă doar cînd statistica este suficient de negativă.",
                "incorrectExplanation": "O statistică mai apropiată de zero decît valoarea critică înseamnă nerespingere; o $p$-valoare mare înseamnă, de asemenea, nerespingere. Valoarea 1,96 a distribuției Normale nu se aplică: sub ipoteza de rădăcină unitară statistica urmează distribuția nestandard Dickey-Fuller, cu valori critice mai negative."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Interpreting an ADF test",
                "text": "An ADF test with a constant gives the statistic $-2.1$; the critical values are $-3.43$ (1%), $-2.86$ (5%) and $-2.57$ (10%). What is the conclusion?",
                "options": [
                    "Reject $H_0$ at all levels: the series is stationary",
                    "Reject $H_0$ only at the 10% level",
                    "Reject $H_0$ at the 5% and 10% levels",
                    "Do not reject $H_0$ at any level: the series probably has a unit root"
                ],
                "correctExplanation": "$-2.1 > -2.57$ (the 10% critical value), so $H_0$ is not rejected at any conventional level.",
                "incorrectExplanation": "Rejection requires a statistic BELOW the critical value. $-2.1$ is less negative than all three critical values, including $-2.57$ at 10%, so there is no level at which the unit root is rejected."
            },
            "ro": {
                "title": "Interpretarea unui test ADF",
                "text": "Un test ADF cu constantă dă statistica $-2{,}1$; valorile critice sînt $-3{,}43$ (1%), $-2{,}86$ (5%) și $-2{,}57$ (10%). Care este concluzia?",
                "options": [
                    "Respingem $H_0$ la toate pragurile: seria este staționară",
                    "Respingem $H_0$ doar la pragul de 10%",
                    "Respingem $H_0$ la pragurile de 5% și 10%",
                    "Nu respingem $H_0$ la niciun prag: seria are probabil rădăcină unitară"
                ],
                "correctExplanation": "$-2{,}1 > -2{,}57$ (valoarea critică la 10%), deci $H_0$ nu este respinsă la niciun prag uzual.",
                "incorrectExplanation": "Respingerea cere o statistică SUB valoarea critică. $-2{,}1$ este mai puțin negativă decît toate cele trei valori critice, inclusiv $-2{,}57$ la 10%, deci nu există niciun prag la care rădăcina unitară să fie respinsă."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The KPSS test",
                "text": "What distinguishes the KPSS test from the ADF and Phillips-Perron tests?",
                "options": [
                    "KPSS reverses the null hypothesis: $H_0$ is stationarity, not a unit root",
                    "KPSS can only detect seasonal unit roots",
                    "KPSS does not need a bandwidth choice for the long-run variance",
                    "KPSS only works on samples larger than 500 observations"
                ],
                "correctExplanation": "KPSS: $H_0$ = (level or trend) stationarity. ADF and PP: $H_0$ = unit root. Using both gives a more robust diagnosis.",
                "incorrectExplanation": "KPSS tests ordinary (non-seasonal) stationarity, it does use a long-run variance estimator with a bandwidth choice, and it has no 500-observation requirement. Its defining feature is the reversed null hypothesis."
            },
            "ro": {
                "title": "Testul KPSS",
                "text": "Ce deosebește testul KPSS de testele ADF și Phillips-Perron?",
                "options": [
                    "KPSS are ipoteza nulă inversată: $H_0$ este staționaritatea, nu rădăcina unitară",
                    "KPSS poate detecta doar rădăcini unitare sezoniere",
                    "KPSS nu necesită alegerea unei lățimi de bandă pentru varianța de lungă durată",
                    "KPSS funcționează doar pe eșantioane mai mari de 500 de observații"
                ],
                "correctExplanation": "KPSS: $H_0$ = staționaritate (în nivel sau în jurul unui trend). ADF și PP: $H_0$ = rădăcină unitară. Folosirea ambelor oferă un diagnostic mai robust.",
                "incorrectExplanation": "KPSS testează staționaritatea obișnuită (nesezonieră), folosește un estimator al varianței de lungă durată cu alegerea unei lățimi de bandă și nu are o cerință de 500 de observații. Trăsătura sa definitorie este ipoteza nulă inversată."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Using ADF and KPSS together",
                "text": "The ADF test does not reject $H_0$ (unit root) and the KPSS test rejects $H_0$ (stationarity). What is the conclusion?",
                "options": [
                    "The series is stationary",
                    "The results are inconclusive",
                    "The series has a unit root (it is nonstationary)",
                    "Both tests were applied incorrectly"
                ],
                "correctExplanation": "ADF fails to reject the unit root and KPSS rejects stationarity: both tests point to nonstationarity, a consistent result.",
                "incorrectExplanation": "This is the agreeing combination, not a conflict. The inconclusive cases are the other two: both tests reject, or neither rejects. Nothing in the result suggests a wrong application."
            },
            "ro": {
                "title": "Folosirea împreună a testelor ADF și KPSS",
                "text": "Testul ADF nu respinge $H_0$ (rădăcină unitară), iar testul KPSS respinge $H_0$ (staționaritate). Care este concluzia?",
                "options": [
                    "Seria este staționară",
                    "Rezultatele sînt neconcludente",
                    "Seria are rădăcină unitară (este nestaționară)",
                    "Ambele teste au fost aplicate incorect"
                ],
                "correctExplanation": "ADF nu respinge rădăcina unitară, iar KPSS respinge staționaritatea: ambele teste indică nestaționaritate, un rezultat coerent.",
                "incorrectExplanation": "Aceasta este combinația concordantă, nu un conflict. Cazurile neconcludente sînt celelalte două: ambele teste resping sau niciunul nu respinge. Nimic din rezultat nu sugerează o aplicare greșită."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The Phillips-Perron test",
                "text": "What is the main difference between the Phillips-Perron (PP) test and the ADF test?",
                "options": [
                    "PP has stationarity as the null hypothesis",
                    "PP corrects for autocorrelation nonparametrically, without adding lagged differences",
                    "PP only works on stationary series",
                    "PP requires the order of integration to be known exactly"
                ],
                "correctExplanation": "Phillips-Perron uses a nonparametric (Newey-West type) correction of the test statistic for autocorrelation and heteroskedasticity, whereas ADF adds lagged differences $\\Delta Y_{t-k}$ to the regression.",
                "incorrectExplanation": "PP and ADF share the same null (unit root); stationarity as the null is KPSS. Like ADF, PP is applied to possibly nonstationary series precisely to find out the order of integration."
            },
            "ro": {
                "title": "Testul Phillips-Perron",
                "text": "Care este principala diferență dintre testul Phillips-Perron (PP) și testul ADF?",
                "options": [
                    "PP are staționaritatea ca ipoteză nulă",
                    "PP corectează autocorelația neparametric, fără a adăuga diferențe întîrziate",
                    "PP funcționează doar pe serii staționare",
                    "PP necesită cunoașterea exactă a ordinului de integrare"
                ],
                "correctExplanation": "Phillips-Perron corectează neparametric (de tip Newey-West) statistica de test pentru autocorelație și heteroscedasticitate, în timp ce ADF adaugă în regresie diferențe întîrziate $\\Delta Y_{t-k}$.",
                "incorrectExplanation": "PP și ADF au aceeași ipoteză nulă (rădăcină unitară); staționaritatea ca ipoteză nulă este specifică testului KPSS. Ca și ADF, PP se aplică unor serii posibil nestaționare tocmai pentru a afla ordinul de integrare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Spurious regression",
                "text": "What happens if OLS is used to regress one I(1) series on another, non-cointegrated I(1) series?",
                "options": [
                    "We obtain consistent and efficient estimators",
                    "We obtain a high $R^2$ and significant $t$-statistics, but the relationship is spurious",
                    "The coefficients converge to zero as $T \\to \\infty$",
                    "The residuals are automatically stationary"
                ],
                "correctExplanation": "Spurious regression (Granger and Newbold, 1974): two independent, non-cointegrated I(1) series typically produce a high $R^2$, apparently significant $t$-tests and a very low Durbin-Watson statistic, although no real relationship exists.",
                "incorrectExplanation": "Without cointegration the residuals are themselves I(1), the slope does not converge to zero (it converges to a random variable) and the $t$-statistics diverge. The estimates are therefore neither consistent nor meaningful."
            },
            "ro": {
                "title": "Regresia falsă",
                "text": "Ce se întîmplă dacă aplicăm regresia OLS între două serii I(1) necointegrate?",
                "options": [
                    "Obținem estimatori consistenți și eficienți",
                    "Obținem un $R^2$ mare și statistici $t$ semnificative, dar relația este falsă",
                    "Coeficienții converg la zero cînd $T \\to \\infty$",
                    "Reziduurile sînt automat staționare"
                ],
                "correctExplanation": "Regresia falsă (spurious regression; Granger și Newbold, 1974): două serii I(1) independente și necointegrate produc de regulă un $R^2$ ridicat, teste $t$ aparent semnificative și o statistică Durbin-Watson foarte mică, deși nu există nicio relație reală.",
                "incorrectExplanation": "Fără cointegrare, reziduurile sînt ele însele I(1), panta nu converge la zero (converge către o variabilă aleatoare), iar statisticile $t$ diverg. Estimările nu sînt, așadar, nici consistente, nici interpretabile."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "ARIMA notation",
                "text": "What does ARIMA(2,1,1) represent?",
                "options": [
                    "An ARMA(2,1) model fitted to the first difference of the series",
                    "An AR(1) model with two differences and an MA(1) term",
                    "An MA(2) model with one difference and an AR(1) term",
                    "Two lags, one trend and one seasonal component"
                ],
                "correctExplanation": "ARIMA($p,d,q$): $p$ = AR order, $d$ = number of differences, $q$ = MA order. ARIMA(2,1,1) is an ARMA(2,1) for $\\Delta Y_t$.",
                "incorrectExplanation": "The order of the triple is fixed: ($p,d,q$). Swapping the AR and MA orders or reading $d$ as the AR order misreads it, and the notation contains no seasonal component (that requires SARIMA)."
            },
            "ro": {
                "title": "Notația ARIMA",
                "text": "Ce reprezintă modelul ARIMA(2,1,1)?",
                "options": [
                    "Un model ARMA(2,1) estimat pe prima diferență a seriei",
                    "Un model AR(1) cu două diferențieri și un termen MA(1)",
                    "Un model MA(2) cu o diferențiere și un termen AR(1)",
                    "Două lag-uri, un trend și o componentă sezonieră"
                ],
                "correctExplanation": "ARIMA($p,d,q$): $p$ = ordinul AR, $d$ = numărul de diferențieri, $q$ = ordinul MA. ARIMA(2,1,1) este un ARMA(2,1) pentru $\\Delta Y_t$.",
                "incorrectExplanation": "Ordinea din triplet este fixă: ($p,d,q$). Inversarea ordinelor AR și MA sau citirea lui $d$ drept ordin AR este o lectură greșită, iar notația nu conține nicio componentă sezonieră (pentru aceasta există SARIMA)."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "ARIMA(0,1,0)",
                "text": "What does the ARIMA(0,1,0) model represent?",
                "options": [
                    "A white noise process",
                    "A stationary AR(1) process",
                    "A random walk",
                    "An MA(1) process"
                ],
                "correctExplanation": "ARIMA(0,1,0): no AR terms, one difference, no MA terms, so $\\Delta Y_t = \\varepsilon_t$, i.e. $Y_t = Y_{t-1} + \\varepsilon_t$: a random walk.",
                "incorrectExplanation": "It is the DIFFERENCE $\\Delta Y_t$ that is white noise, not $Y_t$ itself. An AR(1) with $\\phi = 1$ is not stationary, and there is no MA term in the model."
            },
            "ro": {
                "title": "ARIMA(0,1,0)",
                "text": "Ce reprezintă modelul ARIMA(0,1,0)?",
                "options": [
                    "Un proces zgomot alb",
                    "Un proces AR(1) staționar",
                    "Un mers aleator",
                    "Un proces MA(1)"
                ],
                "correctExplanation": "ARIMA(0,1,0): fără termeni AR, o diferențiere, fără termeni MA, deci $\\Delta Y_t = \\varepsilon_t$, adică $Y_t = Y_{t-1} + \\varepsilon_t$: un mers aleator.",
                "incorrectExplanation": "DIFERENȚA $\\Delta Y_t$ este zgomot alb, nu seria $Y_t$ însăși. Un AR(1) cu $\\phi = 1$ nu este staționar, iar modelul nu conține niciun termen MA."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Special cases of ARIMA",
                "text": "The ARIMA(0,1,1) model, $\\Delta Y_t = \\varepsilon_t + \\theta_1 \\varepsilon_{t-1}$ (IMA(1,1)), is equivalent to:",
                "options": [
                    "A differenced AR(1) model",
                    "A random walk with drift",
                    "A stationary ARMA(1,1) model",
                    "Simple exponential smoothing"
                ],
                "correctExplanation": "IMA(1,1) produces the same forecasts as simple exponential smoothing (SES) with smoothing constant $\\alpha = 1 + \\theta_1$ (with $-1 < \\theta_1 < 0$).",
                "incorrectExplanation": "A random walk with drift is ARIMA(0,1,0) with a constant, a differenced AR(1) is ARIMA(1,1,0), and an ARIMA(0,1,1) is not stationary in levels. The MA(1) term on the differences is what turns the forecast into an exponentially weighted average of past values."
            },
            "ro": {
                "title": "Cazuri particulare ARIMA",
                "text": "Modelul ARIMA(0,1,1), $\\Delta Y_t = \\varepsilon_t + \\theta_1 \\varepsilon_{t-1}$ (IMA(1,1)), este echivalent cu:",
                "options": [
                    "Un model AR(1) diferențiat",
                    "Un mers aleator cu drift",
                    "Un model ARMA(1,1) staționar",
                    "Netezirea exponențială simplă"
                ],
                "correctExplanation": "IMA(1,1) produce aceleași prognoze ca netezirea exponențială simplă (SES), cu constanta de netezire $\\alpha = 1 + \\theta_1$ (pentru $-1 < \\theta_1 < 0$).",
                "incorrectExplanation": "Mersul aleator cu drift este ARIMA(0,1,0) cu constantă, un AR(1) diferențiat este ARIMA(1,1,0), iar ARIMA(0,1,1) nu este staționar în niveluri. Termenul MA(1) pe diferențe face ca prognoza să fie o medie ponderată exponențial a valorilor trecute."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The constant in ARIMA",
                "text": "In an ARIMA($p$,1,$q$) model $\\Delta Y_t = c + \\phi_1 \\Delta Y_{t-1} + \\dots + \\phi_p \\Delta Y_{t-p} + \\varepsilon_t + \\dots$, what does the constant $c$ generate?",
                "options": [
                    "The unconditional mean of the level $Y_t$",
                    "The variance of the errors $\\varepsilon_t$",
                    "A drift: a nonzero mean of $\\Delta Y_t$, equal to $c/(1-\\phi_1-\\dots-\\phi_p)$, i.e. a linear trend in levels",
                    "The lag-1 autocorrelation coefficient"
                ],
                "correctExplanation": "With $d=1$ the constant produces a drift: $E[\\Delta Y_t] = c/(1-\\phi_1-\\dots-\\phi_p)$, so the level $Y_t$ acquires a deterministic linear trend with this slope. For $p=0$ the drift equals $c$.",
                "incorrectExplanation": "An I(1) level has no unconditional mean, so $c$ cannot be one; the error variance is a separate parameter $\\sigma^2$, and autocorrelation is governed by the $\\phi$ and $\\theta$ coefficients. Including $c$ when $d=1$ means assuming a trend in the levels."
            },
            "ro": {
                "title": "Constanta în ARIMA",
                "text": "Într-un model ARIMA($p$,1,$q$) $\\Delta Y_t = c + \\phi_1 \\Delta Y_{t-1} + \\dots + \\phi_p \\Delta Y_{t-p} + \\varepsilon_t + \\dots$, ce generează constanta $c$?",
                "options": [
                    "Media necondiționată a nivelului $Y_t$",
                    "Varianța erorilor $\\varepsilon_t$",
                    "Un drift: o medie nenulă a lui $\\Delta Y_t$, egală cu $c/(1-\\phi_1-\\dots-\\phi_p)$, adică un trend liniar în niveluri",
                    "Coeficientul de autocorelație de ordinul 1"
                ],
                "correctExplanation": "Cu $d=1$, constanta produce un drift: $E[\\Delta Y_t] = c/(1-\\phi_1-\\dots-\\phi_p)$, deci nivelul $Y_t$ capătă un trend liniar determinist cu această pantă. Pentru $p=0$, drift-ul este chiar $c$.",
                "incorrectExplanation": "Un nivel I(1) nu are medie necondiționată, deci $c$ nu poate fi media; varianța erorilor este un parametru separat, $\\sigma^2$, iar autocorelația este dată de coeficienții $\\phi$ și $\\theta$. Includerea lui $c$ cînd $d=1$ înseamnă a presupune un trend în niveluri."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The log transformation",
                "text": "What is the main effect of a log transformation on a time series whose fluctuations grow with its level?",
                "options": [
                    "It removes the trend",
                    "It stabilises the variance and makes proportional changes comparable",
                    "It turns the series into a stationary process",
                    "It removes seasonality"
                ],
                "correctExplanation": "Taking logs stabilises a variance proportional to the level, and $\\Delta \\log Y_t \\approx (Y_t - Y_{t-1})/Y_{t-1}$ is the relative growth rate.",
                "incorrectExplanation": "A log-transformed series keeps its trend and its seasonality, and is usually still I(1); differencing is still needed. The log addresses the scale of the fluctuations, not the trend."
            },
            "ro": {
                "title": "Transformarea logaritmică",
                "text": "Care este efectul principal al transformării logaritmice asupra unei serii de timp ale cărei fluctuații cresc odată cu nivelul?",
                "options": [
                    "Elimină trendul din serie",
                    "Stabilizează varianța și face comparabile variațiile proporționale",
                    "Transformă seria într-un proces staționar",
                    "Elimină sezonalitatea"
                ],
                "correctExplanation": "Logaritmarea stabilizează o varianță proporțională cu nivelul, iar $\\Delta \\log Y_t \\approx (Y_t - Y_{t-1})/Y_{t-1}$ este rata de creștere relativă.",
                "incorrectExplanation": "O serie logaritmată își păstrează trendul și sezonalitatea și este de regulă tot I(1); diferențierea rămîne necesară. Logaritmul acționează asupra amplitudinii fluctuațiilor, nu asupra trendului."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "First step of the Box-Jenkins method",
                "text": "What is the first step of the Box-Jenkins methodology?",
                "options": [
                    "Estimating the parameters",
                    "Identifying the model orders ($p,d,q$)",
                    "Diagnostic checking of the residuals",
                    "Producing forecasts"
                ],
                "correctExplanation": "The cycle is identification (choose $d$ with unit root tests, then $p$ and $q$ from ACF/PACF) → estimation → diagnostic checking, followed by forecasting.",
                "incorrectExplanation": "Parameters cannot be estimated before the model orders are chosen, residuals exist only after estimation, and forecasts come last, once the model has passed the diagnostic checks."
            },
            "ro": {
                "title": "Prima etapă a metodei Box-Jenkins",
                "text": "Care este prima etapă a metodologiei Box-Jenkins?",
                "options": [
                    "Estimarea parametrilor",
                    "Identificarea ordinelor modelului ($p,d,q$)",
                    "Verificarea diagnostică a reziduurilor",
                    "Calculul prognozelor"
                ],
                "correctExplanation": "Ciclul este identificare (alegerea lui $d$ cu teste de rădăcină unitară, apoi a lui $p$ și $q$ pe baza ACF/PACF) → estimare → verificare diagnostică, urmate de prognoză.",
                "incorrectExplanation": "Parametrii nu pot fi estimați înainte de alegerea ordinelor modelului, reziduurile există doar după estimare, iar prognozele vin la final, după ce modelul a trecut verificările diagnostice."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ACF and PACF identification",
                "text": "The ACF of the differenced series cuts off after lag $q$, while its PACF decays exponentially. Which model does this pattern suggest for the differenced series?",
                "options": [
                    "AR($q$)",
                    "MA($q$)",
                    "ARMA($q$, $q$)",
                    "White noise"
                ],
                "correctExplanation": "An ACF cut-off at lag $q$ together with a decaying PACF is the classic signature of an MA($q$) process (so ARIMA($0,1,q$) for the levels).",
                "incorrectExplanation": "The rule is: ACF cuts off → MA; PACF cuts off → AR; both decay → ARMA. White noise would show no significant lags at all."
            },
            "ro": {
                "title": "Identificarea pe baza ACF și PACF",
                "text": "ACF-ul seriei diferențiate se anulează după lag-ul $q$, iar PACF-ul ei scade exponențial. Ce model sugerează această configurație pentru seria diferențiată?",
                "options": [
                    "AR($q$)",
                    "MA($q$)",
                    "ARMA($q$, $q$)",
                    "Zgomot alb"
                ],
                "correctExplanation": "Anularea ACF după lag-ul $q$, împreună cu un PACF descrescător, este semnătura clasică a unui proces MA($q$) (deci ARIMA($0,1,q$) pentru niveluri).",
                "incorrectExplanation": "Regula este: ACF se anulează → MA; PACF se anulează → AR; ambele scad treptat → ARMA. Zgomotul alb nu ar avea niciun lag semnificativ."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Invertibility condition",
                "text": "For the MA(1) model $Y_t = \\varepsilon_t + \\theta_1 \\varepsilon_{t-1}$, what is the invertibility condition?",
                "options": [
                    "$\\theta_1 > 0$",
                    "$\\theta_1 = 0$",
                    "$|\\theta_1| < 1$",
                    "$\\theta_1 > 1$"
                ],
                "correctExplanation": "MA(1) invertibility requires $|\\theta_1| < 1$, which guarantees a convergent AR($\\infty$) representation $\\varepsilon_t = \\sum_{j\\ge 0} (-\\theta_1)^j Y_{t-j}$.",
                "incorrectExplanation": "The sign of $\\theta_1$ is irrelevant; what matters is its modulus. With $|\\theta_1| \\ge 1$ the weights $(-\\theta_1)^j$ do not die out, and $\\theta_1$ and $1/\\theta_1$ give the same ACF, so only the invertible version identifies the model uniquely."
            },
            "ro": {
                "title": "Condiția de invertibilitate",
                "text": "Pentru modelul MA(1) $Y_t = \\varepsilon_t + \\theta_1 \\varepsilon_{t-1}$, care este condiția de invertibilitate?",
                "options": [
                    "$\\theta_1 > 0$",
                    "$\\theta_1 = 0$",
                    "$|\\theta_1| < 1$",
                    "$\\theta_1 > 1$"
                ],
                "correctExplanation": "Invertibilitatea MA(1) cere $|\\theta_1| < 1$, ceea ce asigură o reprezentare AR($\\infty$) convergentă, $\\varepsilon_t = \\sum_{j\\ge 0} (-\\theta_1)^j Y_{t-j}$.",
                "incorrectExplanation": "Semnul lui $\\theta_1$ nu contează, ci modulul său. Pentru $|\\theta_1| \\ge 1$ ponderile $(-\\theta_1)^j$ nu se sting, iar $\\theta_1$ și $1/\\theta_1$ dau același ACF, deci doar varianta invertibilă identifică unic modelul."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Estimating ARIMA models",
                "text": "What is the standard estimation method for the parameters of an ARIMA model?",
                "options": [
                    "Ordinary least squares on the original levels",
                    "Maximum likelihood (exact or conditional)",
                    "Yule-Walker equations for both the AR and the MA part",
                    "Minimising the in-sample MAPE"
                ],
                "correctExplanation": "ARIMA models are estimated by maximum likelihood on the differenced series (exact likelihood via the Kalman filter, or conditional likelihood), which handles the MA part efficiently.",
                "incorrectExplanation": "OLS on the levels ignores the differencing and cannot handle the unobserved MA errors; Yule-Walker equations are moment estimators suited to pure AR models and are inefficient for MA terms; MAPE is a forecast accuracy measure, not an estimation criterion."
            },
            "ro": {
                "title": "Estimarea modelelor ARIMA",
                "text": "Care este metoda standard de estimare a parametrilor unui model ARIMA?",
                "options": [
                    "Metoda celor mai mici pătrate (OLS) pe nivelurile inițiale",
                    "Metoda verosimilității maxime (exactă sau condiționată)",
                    "Ecuațiile Yule-Walker, atît pentru partea AR, cît și pentru partea MA",
                    "Minimizarea MAPE în eșantion"
                ],
                "correctExplanation": "Modelele ARIMA se estimează prin verosimilitate maximă pe seria diferențiată (verosimilitatea exactă, prin filtrul Kalman, sau verosimilitatea condiționată), care tratează eficient partea MA.",
                "incorrectExplanation": "OLS pe niveluri ignoră diferențierea și nu poate trata erorile MA neobservate; ecuațiile Yule-Walker sînt estimatori de momente potriviți modelelor AR pure și ineficienți pentru termenii MA; MAPE este o măsură a acurateței prognozei, nu un criteriu de estimare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Automatic ARIMA selection",
                "text": "Once $d$ has been fixed by unit root tests, how does auto_arima (Hyndman-Khandakar algorithm) choose the orders $p$ and $q$?",
                "options": [
                    "By visual inspection of the ACF and PACF",
                    "By minimising an information criterion (AIC, AICc or BIC) in a stepwise search",
                    "By maximising the in-sample $R^2$",
                    "By choosing the model with the largest Ljung-Box $p$-value"
                ],
                "correctExplanation": "auto_arima uses KPSS-type tests for $d$ and then a stepwise search over $(p,q)$ that minimises AICc (or AIC/BIC), trading fit against complexity.",
                "incorrectExplanation": "Visual inspection is the manual Box-Jenkins route; in-sample $R^2$ always rewards larger models; the Ljung-Box test is a residual diagnostic, not a selection criterion."
            },
            "ro": {
                "title": "Selecția automată ARIMA",
                "text": "După ce $d$ a fost fixat cu teste de rădăcină unitară, cum alege auto_arima (algoritmul Hyndman-Khandakar) ordinele $p$ și $q$?",
                "options": [
                    "Prin inspecția vizuală a ACF și PACF",
                    "Prin minimizarea unui criteriu informațional (AIC, AICc sau BIC) într-o căutare pas cu pas",
                    "Prin maximizarea $R^2$ în eșantion",
                    "Prin alegerea modelului cu cea mai mare $p$-valoare Ljung-Box"
                ],
                "correctExplanation": "auto_arima folosește teste de tip KPSS pentru $d$, apoi o căutare pas cu pas în spațiul $(p,q)$ care minimizează AICc (sau AIC/BIC), echilibrînd ajustarea și complexitatea.",
                "incorrectExplanation": "Inspecția vizuală este calea manuală Box-Jenkins; $R^2$ în eșantion favorizează mereu modelele mai mari; testul Ljung-Box este un diagnostic al reziduurilor, nu un criteriu de selecție."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Ljung-Box test on ARIMA residuals",
                "text": "The Ljung-Box test on the residuals of an ARIMA model gives a $p$-value of 0.02. What should you conclude?",
                "options": [
                    "The model is adequate",
                    "The model is inadequate: the residuals are autocorrelated, so it should be re-specified",
                    "The residuals are not Normally distributed",
                    "The model is overfitted and lags should be removed"
                ],
                "correctExplanation": "$p < 0.05$ rejects $H_0$ (no autocorrelation up to the chosen lag): the residuals still contain linear dependence the model has not captured.",
                "incorrectExplanation": "An adequate model would leave white-noise residuals and a large p-value. The Ljung-Box test concerns autocorrelation, not normality, and remaining autocorrelation points to too FEW terms, not to overfitting."
            },
            "ro": {
                "title": "Testul Ljung-Box pe reziduurile ARIMA",
                "text": "Testul Ljung-Box aplicat reziduurilor unui model ARIMA dă o $p$-valoare de 0,02. Ce concluzie trageți?",
                "options": [
                    "Modelul este adecvat",
                    "Modelul este inadecvat: reziduurile sînt autocorelate, deci trebuie respecificat",
                    "Reziduurile nu urmează distribuția Normală",
                    "Modelul este supraparametrizat și trebuie eliminate lag-uri"
                ],
                "correctExplanation": "$p < 0{,}05$ respinge $H_0$ (absența autocorelației pînă la lag-ul ales): reziduurile conțin încă dependență liniară pe care modelul nu a captat-o.",
                "incorrectExplanation": "Un model adecvat ar lăsa reziduuri de tip zgomot alb și o $p$-valoare mare. Testul Ljung-Box privește autocorelația, nu normalitatea, iar autocorelația rămasă indică PREA PUȚINI termeni, nu supraparametrizare."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Forecast: a numerical example",
                "text": "An ARIMA(1,1,0) model without a constant has $\\phi_1 = 0.6$. The data show $Y_T = 108$ and $\\Delta Y_T = 5$. What is the point forecast $\\hat{Y}_{T+1|T}$?",
                "options": [
                    "108",
                    "113",
                    "111",
                    "110"
                ],
                "correctExplanation": "$\\Delta \\hat{Y}_{T+1} = \\phi_1 \\Delta Y_T = 0.6 \\times 5 = 3$, so $\\hat{Y}_{T+1} = 108 + 3 = 111$.",
                "incorrectExplanation": "108 ignores the AR dynamics of the differences (that is the random-walk forecast), and 113 adds the full last change instead of $\\phi_1$ times it. The model works on differences: $\\hat{Y}_{T+1} = Y_T + \\phi_1 \\Delta Y_T$."
            },
            "ro": {
                "title": "Prognoză: un exemplu numeric",
                "text": "Un model ARIMA(1,1,0) fără constantă are $\\phi_1 = 0{,}6$. Datele arată $Y_T = 108$ și $\\Delta Y_T = 5$. Care este prognoza punctuală $\\hat{Y}_{T+1|T}$?",
                "options": [
                    "108",
                    "113",
                    "111",
                    "110"
                ],
                "correctExplanation": "$\\Delta \\hat{Y}_{T+1} = \\phi_1 \\Delta Y_T = 0{,}6 \\times 5 = 3$, deci $\\hat{Y}_{T+1} = 108 + 3 = 111$.",
                "incorrectExplanation": "108 ignoră dinamica AR a diferențelor (este prognoza mersului aleator), iar 113 adaugă întreaga ultimă variație în loc de $\\phi_1$ ori aceasta. Modelul lucrează pe diferențe: $\\hat{Y}_{T+1} = Y_T + \\phi_1 \\Delta Y_T$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Forecast error variance of a random walk",
                "text": "For ARIMA(0,1,0) (a random walk), how does the forecast error variance behave as the horizon $h$ increases?",
                "options": [
                    "It stays constant",
                    "It decreases to zero",
                    "It grows linearly with $h$",
                    "It converges to a finite limit"
                ],
                "correctExplanation": "$Y_{T+h} - \\hat{Y}_{T+h|T} = \\varepsilon_{T+1} + \\dots + \\varepsilon_{T+h}$, so the forecast error variance is $h\\sigma^2$: the prediction intervals widen like $\\sqrt{h}$.",
                "incorrectExplanation": "Convergence to a finite limit (the unconditional variance) holds for STATIONARY ARMA models. For an I(1) process the future shocks accumulate, so uncertainty grows without bound."
            },
            "ro": {
                "title": "Varianța erorii de prognoză pentru mersul aleator",
                "text": "Pentru ARIMA(0,1,0) (mers aleator), cum evoluează varianța erorii de prognoză pe măsură ce orizontul $h$ crește?",
                "options": [
                    "Rămîne constantă",
                    "Scade la zero",
                    "Crește liniar cu $h$",
                    "Converge la o limită finită"
                ],
                "correctExplanation": "$Y_{T+h} - \\hat{Y}_{T+h|T} = \\varepsilon_{T+1} + \\dots + \\varepsilon_{T+h}$, deci varianța erorii de prognoză este $h\\sigma^2$: intervalele de prognoză se lărgesc proporțional cu $\\sqrt{h}$.",
                "incorrectExplanation": "Convergența la o limită finită (varianța necondiționată) are loc pentru modelele ARMA STAȚIONARE. La un proces I(1) șocurile viitoare se acumulează, deci incertitudinea crește nelimitat."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Rolling-window forecasting",
                "text": "What is the main advantage of rolling-window forecast evaluation over a single estimation on the full sample?",
                "options": [
                    "It reduces the number of estimated parameters",
                    "It removes the need for unit root tests",
                    "It always yields smaller forecast errors than an expanding window",
                    "It mimics real-time forecasting and avoids using future information (look-ahead bias)"
                ],
                "correctExplanation": "With a rolling window the model is re-estimated at each step using only data available at that date, so the out-of-sample errors are realistic and free of look-ahead bias.",
                "incorrectExplanation": "The number of parameters and the need for unit root tests are unchanged. A rolling window is not guaranteed to beat an expanding window: it adapts better to structural change but uses fewer observations. Its value lies in honest out-of-sample evaluation."
            },
            "ro": {
                "title": "Prognoza cu fereastră mobilă",
                "text": "Care este principalul avantaj al evaluării prognozei cu fereastră mobilă (rolling window) față de o singură estimare pe întregul eșantion?",
                "options": [
                    "Reduce numărul de parametri estimați",
                    "Elimină necesitatea testelor de rădăcină unitară",
                    "Produce întotdeauna erori de prognoză mai mici decît o fereastră extinsă",
                    "Reproduce prognoza în timp real și evită folosirea informației din viitor (look-ahead bias)"
                ],
                "correctExplanation": "Cu o fereastră mobilă, modelul este reestimat la fiecare pas folosind doar datele disponibile la acea dată, deci erorile în afara eșantionului sînt realiste și lipsite de look-ahead bias.",
                "incorrectExplanation": "Numărul de parametri și necesitatea testelor de rădăcină unitară rămîn aceleași. Fereastra mobilă nu bate garantat fereastra extinsă: se adaptează mai bine la schimbările structurale, dar folosește mai puține observații. Valoarea ei constă în evaluarea onestă în afara eșantionului."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Limitations of ARIMA models",
                "text": "Which of the following is NOT a limitation of ARIMA models?",
                "options": [
                    "They assume a linear structure",
                    "They cannot capture volatility clustering",
                    "They can handle nonstationary data through differencing",
                    "They assume a constant error variance"
                ],
                "correctExplanation": "Handling unit-root nonstationarity by differencing is a strength of ARIMA, not a limitation.",
                "incorrectExplanation": "Linearity, the inability to model volatility clustering (which requires ARCH/GARCH) and the constant-variance assumption are genuine limitations of ARIMA models."
            },
            "ro": {
                "title": "Limitele modelelor ARIMA",
                "text": "Care dintre următoarele NU este o limită a modelelor ARIMA?",
                "options": [
                    "Presupun o structură liniară",
                    "Nu pot surprinde volatility clustering",
                    "Pot trata date nestaționare prin diferențiere",
                    "Presupun o varianță constantă a erorilor"
                ],
                "correctExplanation": "Tratarea nestaționarității de tip rădăcină unitară prin diferențiere este un punct forte al modelelor ARIMA, nu o limită.",
                "incorrectExplanation": "Liniaritatea, incapacitatea de a modela volatility clustering (pentru care sînt necesare modelele ARCH/GARCH) și ipoteza de varianță constantă sînt limite reale ale modelelor ARIMA."
            }
        }
    ]
};
