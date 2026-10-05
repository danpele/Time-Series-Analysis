// ============================================================
// Chapter 8 quiz bank: Long memory and ARFIMA (EN + RO)
// 24 questions, 20 drawn per attempt.
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
        },
        {
            "correct": 0,
            "en": {
                "title": "Weights of the fractional difference",
                "text": "What is the second weight $\\pi_2$ of $(1-L)^{0.4} = 1 + \\pi_1 L + \\pi_2 L^2 + \\dots$?",
                "options": [
                    "$-0.12$",
                    "$-0.4$",
                    "$0.12$",
                    "$-0.24$"
                ],
                "correctExplanation": "With $\\pi_k = \\pi_{k-1}(k-1-d)/k$: $\\pi_1 = -0.4$ and $\\pi_2 = -0.4 \\cdot 0.6/2 = -0.12$.",
                "incorrectExplanation": "$-0.4$ is $\\pi_1$, the sign of $\\pi_2$ is negative for $0 < d < 1$, and $-0.24$ forgets the division by $k = 2$. The recursion gives $\\pi_2 = -0.12$."
            },
            "ro": {
                "title": "Ponderile diferenței fracționare",
                "text": "Cît este a doua pondere $\\pi_2$ din $(1-L)^{0{,}4} = 1 + \\pi_1 L + \\pi_2 L^2 + \\dots$?",
                "options": [
                    "$-0{,}12$",
                    "$-0{,}4$",
                    "$0{,}12$",
                    "$-0{,}24$"
                ],
                "correctExplanation": "Cu $\\pi_k = \\pi_{k-1}(k-1-d)/k$: $\\pi_1 = -0{,}4$ și $\\pi_2 = -0{,}4 \\cdot 0{,}6/2 = -0{,}12$.",
                "incorrectExplanation": "$-0{,}4$ este $\\pi_1$, semnul lui $\\pi_2$ este negativ pentru $0 < d < 1$, iar $-0{,}24$ omite împărțirea la $k = 2$. Recurența dă $\\pi_2 = -0{,}12$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "First autocorrelation of ARFIMA(0,d,0)",
                "text": "What is $\\rho(1)$ of an ARFIMA$(0, 0.25, 0)$ process?",
                "options": [
                    "$1/3$",
                    "$0.25$",
                    "$0.75$",
                    "$0.5$"
                ],
                "correctExplanation": "For ARFIMA$(0,d,0)$, $\\rho(1) = d/(1-d) = 0.25/0.75 = 1/3$.",
                "incorrectExplanation": "$0.25$ is $d$ itself, $0.75$ is $H$, and $0.5$ has no link with this $d$. The formula $\\rho(1) = d/(1-d)$ gives $1/3$."
            },
            "ro": {
                "title": "Prima autocorelație a unui ARFIMA(0,d,0)",
                "text": "Cît este $\\rho(1)$ pentru un proces ARFIMA$(0;\\ 0{,}25;\\ 0)$?",
                "options": [
                    "$1/3$",
                    "$0{,}25$",
                    "$0{,}75$",
                    "$0{,}5$"
                ],
                "correctExplanation": "Pentru ARFIMA$(0,d,0)$, $\\rho(1) = d/(1-d) = 0{,}25/0{,}75 = 1/3$.",
                "incorrectExplanation": "$0{,}25$ este chiar $d$, $0{,}75$ este $H$, iar $0{,}5$ nu are legătură cu acest $d$. Formula $\\rho(1) = d/(1-d)$ dă $1/3$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Non-stationary but mean-reverting",
                "text": "A series is fractionally integrated with $d = 0.7$. Which statement is true?",
                "options": [
                    "It is non-stationary, but the effect of a shock dies out",
                    "It is stationary with long memory",
                    "It has a unit root: shocks are permanent",
                    "It is not invertible"
                ],
                "correctExplanation": "For $0.5 \\le d < 1$ the variance is not finite, but the impulse responses $\\psi_k \\approx k^{d-1}/\\Gamma(d)$ tend to zero: the series reverts to its mean.",
                "incorrectExplanation": "Stationarity needs $d < 0.5$, a unit root means $d = 1$, and invertibility only fails for $d \\le -0.5$. With $d = 0.7$ the series is non-stationary and mean-reverting."
            },
            "ro": {
                "title": "Nestaționar, dar cu revenire la medie",
                "text": "O serie este integrată fracționar cu $d = 0{,}7$. Care afirmație este adevărată?",
                "options": [
                    "Este nestaționară, dar efectul unui șoc se stinge",
                    "Este staționară, cu memorie lungă",
                    "Are rădăcină unitară: șocurile sînt permanente",
                    "Nu este inversabilă"
                ],
                "correctExplanation": "Pentru $0{,}5 \\le d < 1$ varianța nu este finită, dar răspunsurile la impuls $\\psi_k \\approx k^{d-1}/\\Gamma(d)$ tind spre zero: seria revine la medie.",
                "incorrectExplanation": "Staționaritatea cere $d < 0{,}5$, rădăcina unitară înseamnă $d = 1$, iar inversabilitatea se pierde doar pentru $d \\le -0{,}5$. Cu $d = 0{,}7$ seria este nestaționară și revine la medie."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The GPH regression",
                "text": "In the GPH regression of $\\log I(\\lambda_j)$ on $-\\log(4\\sin^2(\\lambda_j/2))$, $j = 1, \\dots, m$, what does the OLS slope estimate?",
                "options": [
                    "The memory parameter $d$",
                    "The Hurst exponent $H$",
                    "$-2d$",
                    "The variance of the shocks"
                ],
                "correctExplanation": "Near zero $f(\\lambda) \\approx G(4\\sin^2(\\lambda/2))^{-d}$; taking logs, the slope on $-\\log(4\\sin^2(\\lambda_j/2))$ is $d$ (Geweke and Porter-Hudak, 1983).",
                "incorrectExplanation": "$H = d + 1/2$ comes from R/S or DFA, the slope would be $-2d$ only on $\\log\\lambda_j$ with the opposite sign convention, and the variance enters the intercept. The slope is $d$."
            },
            "ro": {
                "title": "Regresia GPH",
                "text": "În regresia GPH a lui $\\log I(\\lambda_j)$ pe $-\\log(4\\sin^2(\\lambda_j/2))$, $j = 1, \\dots, m$, ce estimează panta OLS?",
                "options": [
                    "Parametrul de memorie $d$",
                    "Exponentul Hurst $H$",
                    "$-2d$",
                    "Varianța șocurilor"
                ],
                "correctExplanation": "În apropierea lui zero $f(\\lambda) \\approx G(4\\sin^2(\\lambda/2))^{-d}$; logaritmînd, panta față de $-\\log(4\\sin^2(\\lambda_j/2))$ este $d$ (Geweke și Porter-Hudak, 1983).",
                "incorrectExplanation": "$H = d + 1/2$ se obține din R/S sau DFA, panta ar fi $-2d$ doar față de $\\log\\lambda_j$, iar varianța intră în termenul liber. Panta este $d$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The shuffle test",
                "text": "After a random permutation of the days, the local Whittle estimate of $d$ for S&P 500 absolute returns falls from about 0.53 to about 0. What does this show?",
                "options": [
                    "The memory lies in the order of the observations, not in their distribution",
                    "Absolute returns have fat tails",
                    "The estimator is biased",
                    "The returns themselves have long memory"
                ],
                "correctExplanation": "A permutation keeps every value (so the distribution and the fat tails) and destroys only the time order; the memory disappears, so it comes from the clustering of calm and turbulent days.",
                "incorrectExplanation": "The tails are unchanged by the permutation, the estimator works correctly on the shuffled series, and the returns themselves have $d \\approx 0$. The memory sits in the order of the days."
            },
            "ro": {
                "title": "Testul permutării",
                "text": "După o permutare aleatoare a zilelor, estimarea Whittle locală a lui $d$ pentru randamentele absolute S&P 500 scade de la aproximativ 0,53 la aproximativ 0. Ce arată acest rezultat?",
                "options": [
                    "Memoria se află în ordinea observațiilor, nu în distribuția lor",
                    "Randamentele absolute au cozi groase",
                    "Estimatorul este deplasat",
                    "Randamentele însele au memorie lungă"
                ],
                "correctExplanation": "O permutare păstrează fiecare valoare (deci distribuția și cozile groase) și distruge doar ordinea în timp; memoria dispare, deci provine din gruparea zilelor calme și agitate.",
                "incorrectExplanation": "Permutarea nu schimbă cozile, estimatorul funcționează corect pe seria permutată, iar randamentele au $d \\approx 0$. Memoria stă în ordinea zilelor."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Overlapping data",
                "text": "The 12-month Romanian inflation rate gives $\\hat d \\approx 1$, while the monthly rate gives $\\hat d \\approx 0.3$. What is the main reason?",
                "options": [
                    "Consecutive 12-month rates overlap in 11 months, which steepens the low-frequency periodogram",
                    "Monthly data are less precise",
                    "The 12-month rate is seasonally adjusted",
                    "Romanian inflation has a unit root"
                ],
                "correctExplanation": "The 12-month rate is a moving sum of 12 monthly changes: this filter removes power near frequency $2\\pi/12$ and makes the log-periodogram steeper over the frequencies used by the estimator, pushing $\\hat d$ towards 1.",
                "incorrectExplanation": "Both series come from the same index, the 12-month rate removes seasonality by construction but that is not why $\\hat d$ rises, and the monthly rate shows no unit root. The overlap is the cause."
            },
            "ro": {
                "title": "Date suprapuse",
                "text": "Rata anuală a inflației din România dă $\\hat d \\approx 1$, iar rata lunară dă $\\hat d \\approx 0{,}3$. Care este motivul principal?",
                "options": [
                    "Două rate anuale consecutive au 11 luni comune, ceea ce înclină periodograma la frecvențele joase",
                    "Datele lunare sînt mai puțin precise",
                    "Rata anuală este ajustată sezonier",
                    "Inflația din România are rădăcină unitară"
                ],
                "correctExplanation": "Rata anuală este o sumă mobilă a 12 variații lunare: acest filtru elimină puterea din jurul frecvenței $2\\pi/12$ și înclină log-periodograma pe frecvențele folosite de estimator, împingînd $\\hat d$ spre 1.",
                "incorrectExplanation": "Ambele serii provin din același indice, rata anuală elimină sezonalitatea prin construcție, dar nu de aceea crește $\\hat d$, iar rata lunară nu arată o rădăcină unitară. Cauza este suprapunerea."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Fractional Gaussian noise",
                "text": "Fractional Gaussian noise with $H = 0.3$ has a lag-1 autocorrelation $2^{2H-1} - 1$ that is:",
                "options": [
                    "Positive",
                    "Negative",
                    "Zero",
                    "Equal to 0.3"
                ],
                "correctExplanation": "$2^{2 \\cdot 0.3 - 1} - 1 = 2^{-0.4} - 1 \\approx -0.24$: for $H < 1/2$ the increments are anti-persistent.",
                "incorrectExplanation": "The autocorrelation is positive only for $H > 1/2$, zero for $H = 1/2$, and it does not equal $H$. For $H = 0.3$ it is about $-0.24$."
            },
            "ro": {
                "title": "Zgomotul gaussian fracționar",
                "text": "Zgomotul gaussian fracționar cu $H = 0{,}3$ are autocorelația de ordinul 1, $2^{2H-1} - 1$:",
                "options": [
                    "Pozitivă",
                    "Negativă",
                    "Zero",
                    "Egală cu 0,3"
                ],
                "correctExplanation": "$2^{2 \\cdot 0{,}3 - 1} - 1 = 2^{-0{,}4} - 1 \\approx -0{,}24$: pentru $H < 1/2$ incrementele sînt antipersistente.",
                "incorrectExplanation": "Autocorelația este pozitivă doar pentru $H > 1/2$, zero pentru $H = 1/2$ și nu este egală cu $H$. Pentru $H = 0{,}3$ este aproximativ $-0{,}24$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Bandwidth of semiparametric estimators",
                "text": "What happens when the bandwidth $m$ of the GPH or local Whittle estimator is increased?",
                "options": [
                    "Both the variance and the bias fall",
                    "The variance falls, but short-run dynamics can bias $\\hat d$",
                    "The variance rises and the bias falls",
                    "Nothing: the estimate does not depend on $m$"
                ],
                "correctExplanation": "More frequencies reduce the variance (SE $\\propto 1/\\sqrt m$), but frequencies further from zero carry the short-run (ARMA) part of the spectrum, which biases $\\hat d$.",
                "incorrectExplanation": "A larger $m$ does not reduce the bias, it lowers the variance rather than raising it, and the estimate clearly depends on $m$. Report $\\hat d$ for several bandwidths."
            },
            "ro": {
                "title": "Lățimea de bandă a estimatorilor semiparametrici",
                "text": "Ce se întîmplă cînd crește lățimea de bandă $m$ a estimatorului GPH sau Whittle local?",
                "options": [
                    "Scad atît varianța, cît și deplasarea",
                    "Varianța scade, dar dinamica pe termen scurt poate deplasa $\\hat d$",
                    "Varianța crește și deplasarea scade",
                    "Nimic: estimarea nu depinde de $m$"
                ],
                "correctExplanation": "Mai multe frecvențe reduc varianța (SE $\\propto 1/\\sqrt m$), dar frecvențele mai depărtate de zero poartă partea pe termen scurt (ARMA) a spectrului, care deplasează $\\hat d$.",
                "incorrectExplanation": "Un $m$ mai mare nu reduce deplasarea, scade varianța în loc să o crească, iar estimarea depinde clar de $m$. Raportați $\\hat d$ pentru mai multe lățimi de bandă."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Spurious long memory",
                "text": "A short-memory series has one large shift in its mean. What do long-memory estimators typically report?",
                "options": [
                    "$\\hat d$ close to zero",
                    "$\\hat d$ negative",
                    "$\\hat d$ clearly positive",
                    "An error"
                ],
                "correctExplanation": "A level shift adds power at the lowest frequencies and makes the sample ACF decay slowly, so GPH and local Whittle report $\\hat d > 0$ (Diebold and Inoue, 2001).",
                "incorrectExplanation": "The estimators do not detect the break, they do not turn negative, and they run without error. A break produces a spurious positive $\\hat d$."
            },
            "ro": {
                "title": "Memoria lungă aparentă",
                "text": "O serie cu memorie scurtă are o singură schimbare mare a mediei. Ce raportează de obicei estimatorii memoriei lungi?",
                "options": [
                    "$\\hat d$ apropiat de zero",
                    "$\\hat d$ negativ",
                    "$\\hat d$ clar pozitiv",
                    "O eroare"
                ],
                "correctExplanation": "O schimbare de nivel adaugă putere la cele mai joase frecvențe și face ca ACF de selecție să scadă lent, deci GPH și Whittle local raportează $\\hat d > 0$ (Diebold și Inoue, 2001).",
                "incorrectExplanation": "Estimatorii nu detectează ruptura, nu devin negativi și rulează fără eroare. O ruptură produce un $\\hat d$ pozitiv aparent."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The Nile and its break",
                "text": "For the Nile flow 1871–1970, $\\hat d$ is about 0.36 on the raw series and about 0.05 after removing the means before and after 1898. What follows?",
                "options": [
                    "The Nile has no memory at all",
                    "Exact ML is unreliable",
                    "For this sample the apparent long memory comes mainly from one break",
                    "The 1898 break is caused by long memory"
                ],
                "correctExplanation": "Once the two regime means are removed, the memory disappears: in these 100 years a single change point (Cobb, 1978) explains most of the persistence.",
                "incorrectExplanation": "Short-run dependence may remain, the ML estimates are consistent with the other estimators, and the break (the new regime of flows) is not a product of long memory. The break explains the memory here."
            },
            "ro": {
                "title": "Nilul și ruptura lui",
                "text": "Pentru debitul Nilului din 1871–1970, $\\hat d$ este aproximativ 0,36 pe seria brută și aproximativ 0,05 după eliminarea mediilor de dinainte și de după 1898. Ce rezultă?",
                "options": [
                    "Nilul nu are deloc memorie",
                    "ML exactă nu este de încredere",
                    "Pentru acest eșantion memoria lungă aparentă provine în principal dintr-o singură ruptură",
                    "Ruptura din 1898 este cauzată de memoria lungă"
                ],
                "correctExplanation": "După eliminarea celor două medii de regim memoria dispare: în acești 100 de ani un singur punct de schimbare (Cobb, 1978) explică cea mai mare parte a persistenței.",
                "incorrectExplanation": "Poate rămîne o dependență pe termen scurt, estimările ML sînt în acord cu ceilalți estimatori, iar ruptura nu este un produs al memoriei lungi. Aici ruptura explică memoria."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "FIGARCH against GARCH",
                "text": "How do the ARCH($\\infty$) weights of $\\varepsilon_{t-k}^2$ in $\\sigma_t^2$ decay in GARCH(1,1) and in FIGARCH(1,d,1)?",
                "options": [
                    "Both decay hyperbolically",
                    "Both decay exponentially",
                    "GARCH: exponentially ($\\alpha\\beta^{k-1}$); FIGARCH: hyperbolically (about $k^{-1-d}$)",
                    "GARCH: hyperbolically; FIGARCH: exponentially"
                ],
                "correctExplanation": "GARCH(1,1) gives weights $\\alpha\\beta^{k-1}$; FIGARCH applies $(1-L)^d$ and gives weights that fall like $k^{-1-d}$, so old shocks keep a much larger weight (Baillie, Bollerslev and Mikkelsen, 1996).",
                "incorrectExplanation": "Only FIGARCH has hyperbolic weights; GARCH has exponential ones, not the reverse. FIGARCH was built to replace the exponential decay of GARCH by a hyperbolic one."
            },
            "ro": {
                "title": "FIGARCH comparat cu GARCH",
                "text": "Cum scad ponderile ARCH($\\infty$) ale lui $\\varepsilon_{t-k}^2$ în $\\sigma_t^2$ pentru GARCH(1,1) și FIGARCH(1,d,1)?",
                "options": [
                    "Ambele scad hiperbolic",
                    "Ambele scad exponențial",
                    "GARCH: exponențial ($\\alpha\\beta^{k-1}$); FIGARCH: hiperbolic (aproximativ $k^{-1-d}$)",
                    "GARCH: hiperbolic; FIGARCH: exponențial"
                ],
                "correctExplanation": "GARCH(1,1) dă ponderile $\\alpha\\beta^{k-1}$; FIGARCH aplică $(1-L)^d$ și dă ponderi care scad ca $k^{-1-d}$, deci șocurile vechi păstrează o pondere mult mai mare (Baillie, Bollerslev și Mikkelsen, 1996).",
                "incorrectExplanation": "Doar FIGARCH are ponderi hiperbolice; GARCH are ponderi exponențiale, nu invers. FIGARCH a fost construit pentru a înlocui descreșterea exponențială a GARCH cu una hiperbolică."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Forecasting with ARFIMA",
                "text": "How does an ARFIMA$(0,d,0)$ forecast with $0 < d < 0.5$ behave as the horizon $h$ grows?",
                "options": [
                    "It stays at the last observation",
                    "It reaches the mean after $p$ steps",
                    "It approaches the mean hyperbolically, more slowly than an AR(1)",
                    "It diverges"
                ],
                "correctExplanation": "The forecast uses all past values through the AR($\\infty$) weights and returns to the mean at a hyperbolic rate, slower than the exponential rate of a stationary ARMA.",
                "incorrectExplanation": "Staying at the last value is the random-walk forecast, reaching the mean after a finite number of steps is the MA($q$) case, and a stationary ARFIMA forecast does not diverge."
            },
            "ro": {
                "title": "Prognoza cu ARFIMA",
                "text": "Cum se comportă prognoza unui ARFIMA$(0,d,0)$ cu $0 < d < 0{,}5$ cînd orizontul $h$ crește?",
                "options": [
                    "Rămîne la ultima observație",
                    "Atinge media după $p$ pași",
                    "Se apropie de medie hiperbolic, mai lent decît la un AR(1)",
                    "Diverge"
                ],
                "correctExplanation": "Prognoza folosește toate valorile trecute prin ponderile AR($\\infty$) și revine la medie cu o viteză hiperbolică, mai lentă decît viteza exponențială a unui ARMA staționar.",
                "incorrectExplanation": "Rămînerea la ultima valoare este prognoza mersului aleator, atingerea mediei după un număr finit de pași este cazul MA($q$), iar prognoza unui ARFIMA staționar nu diverge."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Spectral definition of long memory",
                "text": "Near frequency zero, the spectral density of a long-memory process with $0 < d < 1/2$ behaves like:",
                "options": [
                    "A finite positive constant",
                    "Zero",
                    "A periodic function",
                    "$G\\lambda^{-2d}$, which goes to infinity"
                ],
                "correctExplanation": "Long memory means a pole at frequency zero: $f(\\lambda) \\sim G\\lambda^{-2d}$ as $\\lambda \\to 0$, the frequency-domain counterpart of $\\rho(k) \\sim Ck^{2d-1}$.",
                "incorrectExplanation": "A finite positive constant at zero is short memory (ARMA), zero at the origin corresponds to over-differenced or anti-persistent series, and periodicity is seasonality. Long memory gives a pole."
            },
            "ro": {
                "title": "Definiția spectrală a memoriei lungi",
                "text": "În apropierea frecvenței zero, densitatea spectrală a unui proces cu memorie lungă, cu $0 < d < 1/2$, se comportă ca:",
                "options": [
                    "O constantă pozitivă finită",
                    "Zero",
                    "O funcție periodică",
                    "$G\\lambda^{-2d}$, care tinde la infinit"
                ],
                "correctExplanation": "Memoria lungă înseamnă un pol la frecvența zero: $f(\\lambda) \\sim G\\lambda^{-2d}$ cînd $\\lambda \\to 0$, echivalentul în domeniul frecvenței al relației $\\rho(k) \\sim Ck^{2d-1}$.",
                "incorrectExplanation": "O constantă pozitivă finită la zero înseamnă memorie scurtă (ARMA), valoarea zero corespunde seriilor diferențiate excesiv sau antipersistente, iar periodicitatea înseamnă sezonalitate. Memoria lungă dă un pol."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Invertibility range",
                "text": "For which values of $d$ is ARFIMA$(0,d,0)$ both stationary and invertible?",
                "options": [
                    "$0 < d < 1$",
                    "$d < 1/2$",
                    "$d > -1/2$",
                    "$-1/2 < d < 1/2$"
                ],
                "correctExplanation": "Stationarity requires $d < 1/2$ and invertibility requires $d > -1/2$ (Hosking, 1981); both hold on $(-1/2, 1/2)$.",
                "incorrectExplanation": "$0 < d < 1$ includes non-stationary values, $d < 1/2$ includes non-invertible ones, and $d > -1/2$ includes non-stationary ones. Both conditions together give $-1/2 < d < 1/2$."
            },
            "ro": {
                "title": "Intervalul de inversabilitate",
                "text": "Pentru ce valori ale lui $d$ este ARFIMA$(0,d,0)$ atît staționar, cît și inversabil?",
                "options": [
                    "$0 < d < 1$",
                    "$d < 1/2$",
                    "$d > -1/2$",
                    "$-1/2 < d < 1/2$"
                ],
                "correctExplanation": "Staționaritatea cere $d < 1/2$, iar inversabilitatea cere $d > -1/2$ (Hosking, 1981); ambele sînt îndeplinite pe $(-1/2; 1/2)$.",
                "incorrectExplanation": "$0 < d < 1$ include valori nestaționare, $d < 1/2$ include valori neinversabile, iar $d > -1/2$ include valori nestaționare. Ambele condiții împreună dau $-1/2 < d < 1/2$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Python and ARFIMA",
                "text": "A student writes `ARIMA(x, order=(0, 0.3, 0))` in statsmodels to fit an ARFIMA model. What is the problem?",
                "options": [
                    "Nothing: statsmodels estimates fractional $d$ this way",
                    "The order must be given as a list",
                    "The value $d = 0.3$ is non-stationary",
                    "The ARIMA order must be an integer; statsmodels has no ARFIMA class"
                ],
                "correctExplanation": "The ARIMA class accepts only integer differencing orders; fractional $d$ needs an ARFIMA likelihood (exact ML or Whittle) written out or taken from a dedicated package.",
                "incorrectExplanation": "statsmodels does not estimate fractional $d$ through ARIMA, the tuple form of the order is valid, and $d = 0.3$ is stationary. The integer requirement is the problem."
            },
            "ro": {
                "title": "Python și ARFIMA",
                "text": "Un student scrie `ARIMA(x, order=(0, 0.3, 0))` în statsmodels pentru a estima un model ARFIMA. Care este problema?",
                "options": [
                    "Niciuna: statsmodels estimează astfel un $d$ fracționar",
                    "Ordinul trebuie dat ca listă",
                    "Valoarea $d = 0{,}3$ este nestaționară",
                    "Ordinul ARIMA trebuie să fie întreg; statsmodels nu are o clasă ARFIMA"
                ],
                "correctExplanation": "Clasa ARIMA acceptă doar ordine de diferențiere întregi; un $d$ fracționar cere o verosimilitate ARFIMA (ML exactă sau Whittle) scrisă explicit sau preluată dintr-un pachet dedicat.",
                "incorrectExplanation": "statsmodels nu estimează un $d$ fracționar prin ARIMA, forma de tuplu a ordinului este validă, iar $d = 0{,}3$ este staționar. Problema este cerința de număr întreg."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Misspecified parametric estimation",
                "text": "An AR(1) with $\\phi = 0.6$ (no long memory) is fitted as ARFIMA$(0,d,0)$ by Whittle maximum likelihood. What is the likely result?",
                "options": [
                    "$\\hat d \\approx 0$",
                    "$\\hat d \\approx -0.5$",
                    "The likelihood cannot be computed",
                    "$\\hat d$ well above zero, because $d$ absorbs the short-run dependence"
                ],
                "correctExplanation": "With no AR term in the model, the only way to fit the positive autocorrelations is a positive $d$: in simulations the Whittle estimate is about 0.5. Always include ARMA terms as alternatives.",
                "incorrectExplanation": "A correctly specified model would give $\\hat d \\approx 0$, a negative value would contradict the positive autocorrelations, and the likelihood is computable. Misspecification inflates $\\hat d$."
            },
            "ro": {
                "title": "Estimare parametrică greșit specificată",
                "text": "Un AR(1) cu $\\phi = 0{,}6$ (fără memorie lungă) este estimat ca ARFIMA$(0,d,0)$ prin verosimilitate maximă Whittle. Care este rezultatul probabil?",
                "options": [
                    "$\\hat d \\approx 0$",
                    "$\\hat d \\approx -0{,}5$",
                    "Verosimilitatea nu poate fi calculată",
                    "$\\hat d$ mult peste zero, deoarece $d$ absoarbe dependența pe termen scurt"
                ],
                "correctExplanation": "Fără un termen AR în model, singura cale de a reproduce autocorelațiile pozitive este un $d$ pozitiv: în simulări estimarea Whittle este aproximativ 0,5. Includeți întotdeauna termeni ARMA ca alternative.",
                "incorrectExplanation": "Un model corect specificat ar da $\\hat d \\approx 0$, o valoare negativă ar contrazice autocorelațiile pozitive, iar verosimilitatea se poate calcula. Specificarea greșită umflă $\\hat d$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The HAR model",
                "text": "Why is the HAR model of Corsi (2009) often used instead of an ARFIMA model for realised volatility?",
                "options": [
                    "It has a true hyperbolic memory",
                    "It needs no data on volatility",
                    "It is a GARCH model with fewer parameters",
                    "Its daily, weekly and monthly averages mimic long memory and it is estimated by OLS"
                ],
                "correctExplanation": "HAR regresses tomorrow's realised volatility on its daily, weekly and monthly averages: three steps that approximate a hyperbolic decay over the relevant horizons, with simple OLS estimation.",
                "incorrectExplanation": "HAR is not a true long-memory model, it is fitted to realised-volatility data, and it is a regression for observed volatility, not a GARCH model. Its appeal is simplicity with approximate long memory."
            },
            "ro": {
                "title": "Modelul HAR",
                "text": "De ce este folosit adesea modelul HAR al lui Corsi (2009) în locul unui model ARFIMA pentru volatilitatea realizată?",
                "options": [
                    "Are o memorie cu adevărat hiperbolică",
                    "Nu are nevoie de date despre volatilitate",
                    "Este un model GARCH cu mai puțini parametri",
                    "Mediile zilnică, săptămînală și lunară imită memoria lungă, iar estimarea se face prin OLS"
                ],
                "correctExplanation": "HAR regresează volatilitatea realizată de mîine pe mediile ei zilnică, săptămînală și lunară: trei trepte care aproximează o descreștere hiperbolică pe orizonturile relevante, cu o estimare simplă prin OLS.",
                "incorrectExplanation": "HAR nu este un model cu memorie lungă propriu-zisă, se estimează pe date de volatilitate realizată și este o regresie pentru volatilitatea observată, nu un model GARCH. Avantajul lui este simplitatea, cu o memorie lungă aproximativă."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Hurst exponent and d",
                "text": "A DFA analysis of a stationary series gives $H = 0.8$. What is the corresponding memory parameter $d$?",
                "options": [
                    "$d = 0.8$",
                    "$d = 1.3$",
                    "$d = -0.3$",
                    "$d = 0.3$"
                ],
                "correctExplanation": "For a stationary series $H = d + 1/2$, so $d = H - 1/2 = 0.3$: stationary long memory.",
                "incorrectExplanation": "$0.8$ confuses $d$ with $H$, $1.3$ adds instead of subtracting, and $-0.3$ has the wrong sign. $d = H - 0.5 = 0.3$."
            },
            "ro": {
                "title": "Exponentul Hurst și d",
                "text": "O analiză DFA a unei serii staționare dă $H = 0{,}8$. Cît este parametrul de memorie $d$ corespunzător?",
                "options": [
                    "$d = 0{,}8$",
                    "$d = 1{,}3$",
                    "$d = -0{,}3$",
                    "$d = 0{,}3$"
                ],
                "correctExplanation": "Pentru o serie staționară $H = d + 1/2$, deci $d = H - 1/2 = 0{,}3$: memorie lungă staționară.",
                "incorrectExplanation": "$0{,}8$ confundă $d$ cu $H$, $1{,}3$ adună în loc să scadă, iar $-0{,}3$ are semnul greșit. $d = H - 0{,}5 = 0{,}3$."
            }
        }
    ]
};
