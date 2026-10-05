// ============================================================
// Chapter 8 quiz bank: Long memory and ARFIMA (EN + RO)
// 6 questions ported from the 2025/2026 site; 6 drawn per attempt.
// The bank grows to 24 questions when the chapter is rebuilt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['long-memory'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Interpreting the Hurst exponent",
                "text": "The increments of a time series have Hurst exponent $H = 0.8$. What does this indicate?",
                "options": [
                    "The increments are uncorrelated, as for a pure random walk",
                    "The increments have long memory and are persistent (trend-following)",
                    "The increments are anti-persistent (mean-reverting)",
                    "The increments are a short-memory I(0) process"
                ],
                "correctExplanation": "$H > 0.5$ means persistence and long memory: positive increments tend to be followed by positive ones and the autocorrelations decay hyperbolically. With $H = 0.8$ this persistence is strong.",
                "incorrectExplanation": "$H = 0.5$ corresponds to uncorrelated increments (random walk), $H < 0.5$ to anti-persistence, and a short-memory I(0) process has exponentially decaying autocorrelations. $H = 0.8$ signals persistent long memory."
            },
            "ro": {
                "title": "Interpretarea exponentului Hurst",
                "text": "Incrementele unei serii de timp au exponentul Hurst $H = 0{,}8$. Ce indică această valoare?",
                "options": [
                    "Incrementele sînt necorelate, ca în cazul unui mers aleator pur",
                    "Incrementele au memorie lungă și sînt persistente (trend-following)",
                    "Incrementele sînt anti-persistente (mean-reverting)",
                    "Incrementele formează un proces I(0) cu memorie scurtă"
                ],
                "correctExplanation": "$H > 0{,}5$ înseamnă persistență și memorie lungă: incrementele pozitive tind să fie urmate de incremente pozitive, iar autocorelațiile scad hiperbolic. Pentru $H = 0{,}8$ persistența este puternică.",
                "incorrectExplanation": "$H = 0{,}5$ corespunde incrementelor necorelate (mers aleator), $H < 0{,}5$ anti-persistenței, iar un proces I(0) cu memorie scurtă are autocorelații care scad exponențial. $H = 0{,}8$ indică memorie lungă persistentă."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The ARFIMA parameter d",
                "text": "In the ARFIMA$(p, d, q)$ model, the parameter $d$ can take:",
                "options": [
                    "Only integer values (0, 1, 2, ...)",
                    "Only the values $d = 0$ or $d = 1$",
                    "Any real value, including fractional values",
                    "Only negative values"
                ],
                "correctExplanation": "ARFIMA allows fractional differencing $(1-L)^d$ with real $d$: $0 < d < 0.5$ gives stationary long memory, $0.5 \\leq d < 1$ gives non-stationary but mean-reverting long memory, and $-0.5 < d < 0$ gives anti-persistence. For a stationary series, $d = H - 0.5$.",
                "incorrectExplanation": "Integer values of $d$, including 0 and 1, are the ARIMA special cases, and negative values are only one part of the admissible range. In ARFIMA, $d$ can be any real number."
            },
            "ro": {
                "title": "Parametrul d din ARFIMA",
                "text": "În modelul ARFIMA$(p, d, q)$, parametrul $d$ poate lua:",
                "options": [
                    "Doar valori întregi (0, 1, 2, ...)",
                    "Doar valorile $d = 0$ sau $d = 1$",
                    "Orice valoare reală, inclusiv valori fracționare",
                    "Doar valori negative"
                ],
                "correctExplanation": "ARFIMA permite diferențierea fracționară $(1-L)^d$ cu $d$ real: $0 < d < 0{,}5$ dă memorie lungă staționară, $0{,}5 \\leq d < 1$ dă memorie lungă nestaționară, dar cu revenire la medie, iar $-0{,}5 < d < 0$ dă anti-persistență. Pentru o serie staționară, $d = H - 0{,}5$.",
                "incorrectExplanation": "Valorile întregi ale lui $d$, inclusiv 0 și 1, sînt cazurile particulare ARIMA, iar valorile negative sînt doar o parte din domeniul admis. În ARFIMA, $d$ poate fi orice număr real."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Long memory in finance",
                "text": "In which financial series is long memory most commonly documented?",
                "options": [
                    "Stock prices",
                    "Daily returns",
                    "Volatility (squared or absolute returns)",
                    "Trading volume"
                ],
                "correctExplanation": "Returns are close to uncorrelated ($H \\approx 0.5$), but squared and absolute returns show autocorrelations that decay very slowly (estimates of $H$ around 0.7 to 0.9). This is a key stylised fact and the motivation for FIGARCH-type models.",
                "incorrectExplanation": "Prices are non-stationary (unit root) rather than long-memory stationary, and daily returns show little autocorrelation. Volume can also be persistent, but the best-documented case is volatility, measured by squared or absolute returns."
            },
            "ro": {
                "title": "Memoria lungă în finanțe",
                "text": "În ce serie financiară este memoria lungă documentată cel mai frecvent?",
                "options": [
                    "Prețurile acțiunilor",
                    "Randamentele zilnice",
                    "Volatilitatea (pătratele sau valorile absolute ale randamentelor)",
                    "Volumul tranzacțiilor"
                ],
                "correctExplanation": "Randamentele sînt aproape necorelate ($H \\approx 0{,}5$), însă pătratele și valorile absolute ale randamentelor au autocorelații care scad foarte lent (estimări ale lui $H$ între 0,7 și 0,9). Acesta este un fapt stilizat esențial și motivația modelelor de tip FIGARCH.",
                "incorrectExplanation": "Prețurile sînt nestaționare (rădăcină unitară), nu staționare cu memorie lungă, iar randamentele zilnice au autocorelații foarte mici. Și volumul poate fi persistent, dar cazul cel mai bine documentat este volatilitatea, măsurată prin pătratele sau valorile absolute ale randamentelor."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ACF decay pattern",
                "text": "The ACF of a long-memory process decays:",
                "options": [
                    "Exponentially fast",
                    "Hyperbolically (slowly)",
                    "Linearly",
                    "Immediately to zero"
                ],
                "correctExplanation": "Long memory: $\\rho(k) \\sim c\\,k^{2d-1}$, a hyperbolic (power-law) decay whose sum diverges for $0 < d < 0.5$. Short-memory ARMA processes have exponentially decaying ACF.",
                "incorrectExplanation": "Exponential decay is the ARMA (short-memory) pattern, a near-linear decline is typical of a unit root in finite samples, and an immediate drop to zero corresponds to white noise or an MA process. Long memory means hyperbolic decay."
            },
            "ro": {
                "title": "Tiparul de scădere al ACF",
                "text": "ACF a unui proces cu memorie lungă scade:",
                "options": [
                    "Exponențial, rapid",
                    "Hiperbolic (lent)",
                    "Liniar",
                    "Imediat la zero"
                ],
                "correctExplanation": "Memorie lungă: $\\rho(k) \\sim c\\,k^{2d-1}$, o scădere hiperbolică (de tip putere) a cărei sumă diverge pentru $0 < d < 0{,}5$. Procesele ARMA, cu memorie scurtă, au o ACF care scade exponențial.",
                "incorrectExplanation": "Scăderea exponențială este tiparul ARMA (memorie scurtă), scăderea aproape liniară este tipică unei rădăcini unitare în eșantioane finite, iar anularea imediată corespunde zgomotului alb sau unui proces MA. Memoria lungă înseamnă scădere hiperbolică."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Stationarity of ARFIMA",
                "text": "An ARFIMA process with $d = 0.3$ is:",
                "options": [
                    "Non-stationary",
                    "Stationary with long memory",
                    "Stationary with short memory",
                    "A unit-root process"
                ],
                "correctExplanation": "For $0 < d < 0.5$ the process is stationary, but its autocorrelations decay hyperbolically, so it has long memory. For $d \\geq 0.5$ it becomes non-stationary.",
                "incorrectExplanation": "Non-stationarity starts only at $d \\geq 0.5$, and a unit root corresponds to $d = 1$; short memory corresponds to $d = 0$. With $d = 0.3$ the process is stationary with long memory."
            },
            "ro": {
                "title": "Staționaritatea ARFIMA",
                "text": "Un proces ARFIMA cu $d = 0{,}3$ este:",
                "options": [
                    "Nestaționar",
                    "Staționar, cu memorie lungă",
                    "Staționar, cu memorie scurtă",
                    "Un proces cu rădăcină unitară"
                ],
                "correctExplanation": "Pentru $0 < d < 0{,}5$ procesul este staționar, dar autocorelațiile sale scad hiperbolic, deci are memorie lungă. Pentru $d \\geq 0{,}5$ devine nestaționar.",
                "incorrectExplanation": "Nestaționaritatea începe abia de la $d \\geq 0{,}5$, iar rădăcina unitară corespunde lui $d = 1$; memoria scurtă corespunde lui $d = 0$. Pentru $d = 0{,}3$ procesul este staționar, cu memorie lungă."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "R/S analysis",
                "text": "The R/S (rescaled range) statistic is used to estimate:",
                "options": [
                    "The ARIMA order",
                    "The Hurst exponent",
                    "The GARCH parameters",
                    "The seasonal period"
                ],
                "correctExplanation": "R/S analysis uses $\\mathbb{E}[R/S] \\sim c\\,n^H$: regressing $\\log(R/S)$ on $\\log n$ gives a slope that estimates $H$. Hurst developed it while studying the Nile floods.",
                "incorrectExplanation": "ARIMA orders are chosen with ACF, PACF and information criteria, GARCH parameters by maximum likelihood, and the seasonal period from the sampling frequency or the periodogram. R/S is a classical estimator of the Hurst exponent."
            },
            "ro": {
                "title": "Analiza R/S",
                "text": "Statistica R/S (rescaled range) se folosește pentru estimarea:",
                "options": [
                    "Ordinului ARIMA",
                    "Exponentului Hurst",
                    "Parametrilor GARCH",
                    "Perioadei sezoniere"
                ],
                "correctExplanation": "Analiza R/S folosește relația $\\mathbb{E}[R/S] \\sim c\\,n^H$: panta regresiei lui $\\log(R/S)$ pe $\\log n$ estimează $H$. Hurst a dezvoltat-o studiind inundațiile Nilului.",
                "incorrectExplanation": "Ordinele ARIMA se aleg cu ACF, PACF și criterii informaționale, parametrii GARCH prin verosimilitate maximă, iar perioada sezonieră din frecvența datelor sau din periodogramă. R/S este un estimator clasic al exponentului Hurst."
            }
        }
    ]
};
