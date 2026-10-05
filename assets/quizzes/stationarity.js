// ============================================================
// Chapter 1 quiz bank: Stochastic processes and stationarity (EN + RO)
// 20 questions ported from the 2025/2026 site; 20 drawn per attempt.
// The bank grows to 24 questions when the chapter is rebuilt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['stationarity'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 2,
            "en": {
                "title": "Stochastic process",
                "text": "What does a stochastic process $\\{X_t\\}$ describe?",
                "options": [
                    "A deterministic function of time",
                    "A fixed numerical value",
                    "A family of random variables indexed by time",
                    "A sequence of numbers in increasing order"
                ],
                "correctExplanation": "A stochastic process is a collection $\\{X_t : t \\in T\\}$ of random variables, one for each point in time. An observed time series is one realisation of it.",
                "incorrectExplanation": "A deterministic function or a fixed value contains no randomness, and an ordered sequence of numbers is just data. A stochastic process is a family of random variables indexed by time."
            },
            "ro": {
                "title": "Proces stochastic",
                "text": "Ce descrie un proces stochastic $\\{X_t\\}$?",
                "options": [
                    "O funcție deterministă de timp",
                    "O valoare numerică fixă",
                    "O familie de variabile aleatoare indexate după timp",
                    "Un șir de numere ordonate crescător"
                ],
                "correctExplanation": "Un proces stochastic este o colecție $\\{X_t : t \\in T\\}$ de variabile aleatoare, cîte una pentru fiecare moment de timp. O serie de timp observată este o realizare a acestuia.",
                "incorrectExplanation": "O funcție deterministă sau o valoare fixă nu conțin nimic aleator, iar un șir ordonat de numere este doar un set de date. Un proces stochastic este o familie de variabile aleatoare indexate după timp."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Autocovariance function",
                "text": "Which expression defines the autocovariance function $\\gamma(t,s)$?",
                "options": [
                    "$\\mathbb{E}[X_t] \\cdot \\mathbb{E}[X_s]$",
                    "$\\text{Var}(X_t + X_s)$",
                    "$\\mathbb{E}[(X_t - \\mu_t)(X_s - \\mu_s)]$",
                    "$\\mathbb{E}[X_t^2]$"
                ],
                "correctExplanation": "The autocovariance is the covariance between the process at two dates: $\\gamma(t,s) = \\text{Cov}(X_t, X_s) = \\mathbb{E}[(X_t - \\mu_t)(X_s - \\mu_s)]$. It measures linear dependence between $X_t$ and $X_s$.",
                "incorrectExplanation": "The product of the means ignores the joint behaviour, the variance of the sum mixes variances and covariance, and $\\mathbb{E}[X_t^2]$ is a raw second moment at one date. The autocovariance is $\\mathbb{E}[(X_t - \\mu_t)(X_s - \\mu_s)]$."
            },
            "ro": {
                "title": "Funcția de autocovarianță",
                "text": "Ce expresie definește funcția de autocovarianță $\\gamma(t,s)$?",
                "options": [
                    "$\\mathbb{E}[X_t] \\cdot \\mathbb{E}[X_s]$",
                    "$\\text{Var}(X_t + X_s)$",
                    "$\\mathbb{E}[(X_t - \\mu_t)(X_s - \\mu_s)]$",
                    "$\\mathbb{E}[X_t^2]$"
                ],
                "correctExplanation": "Autocovarianța este covarianța procesului la două momente: $\\gamma(t,s) = \\text{Cov}(X_t, X_s) = \\mathbb{E}[(X_t - \\mu_t)(X_s - \\mu_s)]$. Ea măsoară dependența liniară dintre $X_t$ și $X_s$.",
                "incorrectExplanation": "Produsul mediilor ignoră comportamentul comun, varianța sumei amestecă varianțele cu covarianța, iar $\\mathbb{E}[X_t^2]$ este un moment de ordinul doi necentrat la un singur moment. Autocovarianța este $\\mathbb{E}[(X_t - \\mu_t)(X_s - \\mu_s)]$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Strict stationarity",
                "text": "What does strict stationarity require?",
                "options": [
                    "Only a constant mean",
                    "Only a constant variance",
                    "All finite-dimensional distributions are invariant to time shifts",
                    "No visible trend in the plot"
                ],
                "correctExplanation": "Strict stationarity requires $(X_{t_1}, \\ldots, X_{t_k}) \\overset{d}{=} (X_{t_1+h}, \\ldots, X_{t_k+h})$ for every $k$, every set of dates and every shift $h$.",
                "incorrectExplanation": "A constant mean or a constant variance concerns a single moment, and the absence of a visible trend is not a definition. Strict stationarity requires the whole joint distribution to be invariant to time shifts."
            },
            "ro": {
                "title": "Staționaritate strictă",
                "text": "Ce presupune staționaritatea strictă?",
                "options": [
                    "Doar o medie constantă",
                    "Doar o varianță constantă",
                    "Toate distribuțiile finit-dimensionale sînt invariante la translații în timp",
                    "Absența unui trend vizibil pe grafic"
                ],
                "correctExplanation": "Staționaritatea strictă cere $(X_{t_1}, \\ldots, X_{t_k}) \\overset{d}{=} (X_{t_1+h}, \\ldots, X_{t_k+h})$ pentru orice $k$, orice set de momente și orice translație $h$.",
                "incorrectExplanation": "O medie sau o varianță constantă privesc un singur moment al distribuției, iar absența unui trend vizibil nu este o definiție. Staționaritatea strictă cere ca întreaga distribuție comună să fie invariantă la translații în timp."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Strict and weak stationarity",
                "text": "Which statement about the relationship between strict and weak stationarity is correct?",
                "options": [
                    "Weak stationarity implies strict stationarity",
                    "The two concepts are identical",
                    "Strict stationarity is irrelevant in practice",
                    "Strict stationarity with finite second moments implies weak stationarity"
                ],
                "correctExplanation": "If the process is strictly stationary and $\\mathbb{E}[X_t^2] < \\infty$, its mean, variance and autocovariances do not depend on time, so it is weakly stationary. The converse fails in general; it holds for Gaussian processes, whose distribution is fully determined by the first two moments.",
                "incorrectExplanation": "Weak stationarity constrains only the first two moments, so it does not imply strict stationarity (except for Gaussian processes), and the two concepts are not identical. The valid implication runs from strict (with finite second moments) to weak."
            },
            "ro": {
                "title": "Staționaritate strictă și staționaritate slabă",
                "text": "Care afirmație despre relația dintre staționaritatea strictă și cea slabă este corectă?",
                "options": [
                    "Staționaritatea slabă implică staționaritatea strictă",
                    "Cele două concepte sînt identice",
                    "Staționaritatea strictă nu are relevanță practică",
                    "Staționaritatea strictă, cu momente de ordinul doi finite, implică staționaritatea slabă"
                ],
                "correctExplanation": "Dacă procesul este strict staționar și $\\mathbb{E}[X_t^2] < \\infty$, media, varianța și autocovarianțele nu depind de timp, deci procesul este slab staționar. Reciproca nu este adevărată în general; ea are loc pentru procesele gaussiene, a căror distribuție este complet determinată de primele două momente.",
                "incorrectExplanation": "Staționaritatea slabă impune condiții doar asupra primelor două momente, deci nu implică staționaritatea strictă (cu excepția proceselor gaussiene), iar cele două concepte nu sînt identice. Implicația corectă merge de la staționaritatea strictă (cu momente de ordinul doi finite) spre cea slabă."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Ergodicity",
                "text": "What does ergodicity allow in practice?",
                "options": [
                    "Estimating parameters from many realisations of the process",
                    "Estimating population moments from a single realisation",
                    "Predicting future values exactly",
                    "Eliminating uncertainty completely"
                ],
                "correctExplanation": "Ergodicity guarantees that time averages converge to ensemble averages, e.g. $\\frac{1}{T}\\sum_{t=1}^{T} X_t \\to \\mathbb{E}[X_t]$. Since we observe only one trajectory of an economic series, this is what makes inference possible.",
                "incorrectExplanation": "In practice we never observe many realisations of the same process, and ergodicity says nothing about exact prediction or eliminating uncertainty. It allows us to learn population moments from a single observed trajectory."
            },
            "ro": {
                "title": "Ergodicitate",
                "text": "Ce permite ergodicitatea în practică?",
                "options": [
                    "Estimarea parametrilor din mai multe realizări ale procesului",
                    "Estimarea momentelor populației dintr-o singură realizare",
                    "Prognoza exactă a valorilor viitoare",
                    "Eliminarea completă a incertitudinii"
                ],
                "correctExplanation": "Ergodicitatea garantează convergența mediilor temporale către mediile de ansamblu, de exemplu $\\frac{1}{T}\\sum_{t=1}^{T} X_t \\to \\mathbb{E}[X_t]$. Deoarece observăm o singură traiectorie a unei serii economice, tocmai această proprietate face posibilă inferența.",
                "incorrectExplanation": "În practică nu observăm niciodată mai multe realizări ale aceluiași proces, iar ergodicitatea nu spune nimic despre prognoza exactă sau despre eliminarea incertitudinii. Ea permite estimarea momentelor populației dintr-o singură traiectorie observată."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Wold's theorem",
                "text": "What does the Wold decomposition theorem state?",
                "options": [
                    "Every time series is stationary",
                    "Every weakly stationary process decomposes into an MA($\\infty$) part plus a deterministic component",
                    "Every time series has a trend",
                    "Only AR processes are stationary"
                ],
                "correctExplanation": "Wold: $X_t = \\sum_{j=0}^{\\infty}\\psi_j\\varepsilon_{t-j} + D_t$, with $\\psi_0 = 1$, $\\sum\\psi_j^2 < \\infty$, $\\varepsilon_t$ white noise and $D_t$ deterministic. This is why linear ARMA models are a natural approximation for stationary series.",
                "incorrectExplanation": "The theorem assumes weak stationarity rather than proving it, says nothing about trends, and covers MA and ARMA processes as well as AR. Its content is the MA($\\infty$) plus deterministic decomposition."
            },
            "ro": {
                "title": "Teorema lui Wold",
                "text": "Ce afirmă teorema de descompunere a lui Wold?",
                "options": [
                    "Orice serie de timp este staționară",
                    "Orice proces slab staționar se descompune într-o parte MA($\\infty$) și o componentă deterministă",
                    "Orice serie de timp are trend",
                    "Doar procesele AR sînt staționare"
                ],
                "correctExplanation": "Wold: $X_t = \\sum_{j=0}^{\\infty}\\psi_j\\varepsilon_{t-j} + D_t$, cu $\\psi_0 = 1$, $\\sum\\psi_j^2 < \\infty$, $\\varepsilon_t$ zgomot alb și $D_t$ deterministă. De aceea modelele liniare ARMA sînt o aproximare firească pentru seriile staționare.",
                "incorrectExplanation": "Teorema presupune staționaritatea slabă, nu o demonstrează, nu spune nimic despre trend și acoperă și procesele MA și ARMA, nu doar pe cele AR. Conținutul ei este descompunerea în parte MA($\\infty$) și componentă deterministă."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Lag operator",
                "text": "What is $(1-L)X_t$, where $L$ is the lag operator?",
                "options": [
                    "$X_t + X_{t-1}$",
                    "$X_t - X_{t-1}$ (the first difference)",
                    "$X_t \\cdot X_{t-1}$",
                    "$X_t / X_{t-1}$"
                ],
                "correctExplanation": "$L X_t = X_{t-1}$, so $(1-L)X_t = X_t - X_{t-1} = \\Delta X_t$.",
                "incorrectExplanation": "The lag operator is linear and shifts the series back one period; it never produces sums, products or ratios of $X_t$ and $X_{t-1}$. The operator $(1-L)$ gives the first difference $\\Delta X_t = X_t - X_{t-1}$."
            },
            "ro": {
                "title": "Operatorul lag",
                "text": "Cît este $(1-L)X_t$, unde $L$ este operatorul lag?",
                "options": [
                    "$X_t + X_{t-1}$",
                    "$X_t - X_{t-1}$ (diferența de ordinul întîi)",
                    "$X_t \\cdot X_{t-1}$",
                    "$X_t / X_{t-1}$"
                ],
                "correctExplanation": "$L X_t = X_{t-1}$, deci $(1-L)X_t = X_t - X_{t-1} = \\Delta X_t$.",
                "incorrectExplanation": "Operatorul lag este liniar și deplasează seria cu o perioadă înapoi; nu produce sume, produse sau rapoarte între $X_t$ și $X_{t-1}$. Operatorul $(1-L)$ dă diferența de ordinul întîi $\\Delta X_t = X_t - X_{t-1}$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Random walk with drift",
                "text": "For $X_t = \\mu + X_{t-1} + \\varepsilon_t$ with $\\mu > 0$ and $X_0 = 0$, which statement is correct?",
                "options": [
                    "The series is stationary",
                    "$\\mathbb{E}[X_t] = \\mu$ (constant)",
                    "$\\mathbb{E}[X_t] = \\mu t$ (grows linearly)",
                    "The variance is constant"
                ],
                "correctExplanation": "Iterating gives $X_t = \\mu t + \\sum_{i=1}^{t}\\varepsilon_i$, so $\\mathbb{E}[X_t] = \\mu t$ and $\\text{Var}(X_t) = t\\sigma^2$. Both depend on $t$, so the process is non-stationary.",
                "incorrectExplanation": "The drift accumulates: the mean is $\\mu t$, not $\\mu$, and the variance $t\\sigma^2$ grows as well, so the series is neither stationary nor of constant variance."
            },
            "ro": {
                "title": "Mers aleator cu drift",
                "text": "Pentru $X_t = \\mu + X_{t-1} + \\varepsilon_t$, cu $\\mu > 0$ și $X_0 = 0$, care afirmație este corectă?",
                "options": [
                    "Seria este staționară",
                    "$\\mathbb{E}[X_t] = \\mu$ (constantă)",
                    "$\\mathbb{E}[X_t] = \\mu t$ (crește liniar)",
                    "Varianța este constantă"
                ],
                "correctExplanation": "Prin iterare, $X_t = \\mu t + \\sum_{i=1}^{t}\\varepsilon_i$, deci $\\mathbb{E}[X_t] = \\mu t$ și $\\text{Var}(X_t) = t\\sigma^2$. Ambele depind de $t$, deci procesul este nestaționar.",
                "incorrectExplanation": "Drift-ul se cumulează: media este $\\mu t$, nu $\\mu$, iar varianța $t\\sigma^2$ crește și ea, deci seria nu este nici staționară, nici de varianță constantă."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "ACF and nonstationarity",
                "text": "An autocorrelation function (ACF) that decays very slowly towards zero suggests that:",
                "options": [
                    "The series is white noise",
                    "The series is stationary with short memory",
                    "The series is non-stationary (possibly a unit root)",
                    "The series is seasonal"
                ],
                "correctExplanation": "A sample ACF that stays high and declines almost linearly over many lags is the classic sign of a unit root. A formal test such as ADF or KPSS should confirm it.",
                "incorrectExplanation": "White noise has ACF close to zero at all non-zero lags, a short-memory stationary series has an ACF that dies out quickly, and seasonality shows up as spikes at multiples of the seasonal lag. Very slow decay points to non-stationarity."
            },
            "ro": {
                "title": "ACF și nestaționaritate",
                "text": "O funcție de autocorelație (ACF) care scade foarte lent spre zero sugerează că:",
                "options": [
                    "Seria este zgomot alb",
                    "Seria este staționară, cu memorie scurtă",
                    "Seria este nestaționară (posibil cu rădăcină unitară)",
                    "Seria are sezonalitate"
                ],
                "correctExplanation": "O ACF de selecție care rămîne ridicată și scade aproape liniar pe multe lag-uri este semnul clasic al unei rădăcini unitare. Un test formal, de exemplu ADF sau KPSS, trebuie să confirme acest lucru.",
                "incorrectExplanation": "Zgomotul alb are ACF apropiată de zero la toate lag-urile nenule, o serie staționară cu memorie scurtă are o ACF care se stinge repede, iar sezonalitatea apare ca vîrfuri la multiplii lag-ului sezonier. Scăderea foarte lentă indică nestaționaritatea."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Properties of the autocorrelation function",
                "text": "For a weakly stationary process, $\\rho(h) = \\gamma(h)/\\gamma(0)$ satisfies:",
                "options": [
                    "$\\rho(0) = 0$",
                    "$\\rho(h) = \\rho(-h)$ and $|\\rho(h)| \\leq 1$",
                    "$\\rho(h)$ increases with $h$",
                    "$\\rho(h) = 1$ for all $h$"
                ],
                "correctExplanation": "The ACF is symmetric ($\\rho(h) = \\rho(-h)$), equals 1 at lag 0 and is bounded: $|\\rho(h)| \\leq 1$ for all $h$ (Cauchy-Schwarz).",
                "incorrectExplanation": "By construction $\\rho(0) = \\gamma(0)/\\gamma(0) = 1$, not 0; $\\rho(h) = 1$ at every lag would mean a perfectly persistent process, and there is no reason for $\\rho(h)$ to increase. The ACF is symmetric and bounded by 1 in absolute value."
            },
            "ro": {
                "title": "Proprietățile funcției de autocorelație",
                "text": "Pentru un proces slab staționar, $\\rho(h) = \\gamma(h)/\\gamma(0)$ satisface:",
                "options": [
                    "$\\rho(0) = 0$",
                    "$\\rho(h) = \\rho(-h)$ și $|\\rho(h)| \\leq 1$",
                    "$\\rho(h)$ crește odată cu $h$",
                    "$\\rho(h) = 1$ pentru orice $h$"
                ],
                "correctExplanation": "ACF este simetrică ($\\rho(h) = \\rho(-h)$), egală cu 1 la lag-ul 0 și mărginită: $|\\rho(h)| \\leq 1$ pentru orice $h$ (inegalitatea Cauchy-Schwarz).",
                "incorrectExplanation": "Prin construcție $\\rho(0) = \\gamma(0)/\\gamma(0) = 1$, nu 0; $\\rho(h) = 1$ la toate lag-urile ar însemna un proces perfect persistent, și nu există niciun motiv ca $\\rho(h)$ să crească. ACF este simetrică și mărginită în valoare absolută de 1."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Log returns",
                "text": "How do we obtain a stationary series from financial prices $P_t$?",
                "options": [
                    "Apply the transformation $\\sqrt{P_t}$",
                    "Multiply by a constant",
                    "Compute log returns $r_t = \\ln(P_t / P_{t-1})$",
                    "Apply a moving average"
                ],
                "correctExplanation": "Log returns $r_t = \\ln P_t - \\ln P_{t-1} = \\Delta \\ln P_t$ are the first difference of log prices; they are typically stationary, unlike prices.",
                "incorrectExplanation": "A square root or a constant factor changes the scale but not the unit root, and a moving average smooths the series but keeps its stochastic trend. Differencing the log price removes the unit root."
            },
            "ro": {
                "title": "Randamente logaritmice",
                "text": "Cum obținem o serie staționară din prețurile financiare $P_t$?",
                "options": [
                    "Aplicăm transformarea $\\sqrt{P_t}$",
                    "Înmulțim cu o constantă",
                    "Calculăm randamentele logaritmice $r_t = \\ln(P_t / P_{t-1})$",
                    "Aplicăm o medie mobilă"
                ],
                "correctExplanation": "Randamentele logaritmice $r_t = \\ln P_t - \\ln P_{t-1} = \\Delta \\ln P_t$ sînt prima diferență a logaritmului prețului; ele sînt de regulă staționare, spre deosebire de prețuri.",
                "incorrectExplanation": "Radicalul sau un factor constant schimbă scala, dar nu elimină rădăcina unitară, iar media mobilă netezește seria, dar îi păstrează trendul stochastic. Diferențierea logaritmului prețului elimină rădăcina unitară."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Weak stationarity",
                "text": "Which condition is NOT required for weak stationarity?",
                "options": [
                    "Constant mean: $\\mathbb{E}[X_t] = \\mu$",
                    "Constant variance: $\\text{Var}(X_t) = \\sigma^2$",
                    "A Normal distribution",
                    "Autocovariance that depends only on the lag"
                ],
                "correctExplanation": "Weak stationarity requires only a constant mean, a constant (finite) variance and autocovariances that depend on the lag alone. It makes no assumption about the distribution.",
                "incorrectExplanation": "Constant mean, constant variance and lag-dependent autocovariance are exactly the three conditions of weak stationarity. Normality is not among them."
            },
            "ro": {
                "title": "Staționaritate slabă",
                "text": "Care condiție NU este necesară pentru staționaritatea slabă?",
                "options": [
                    "Medie constantă: $\\mathbb{E}[X_t] = \\mu$",
                    "Varianță constantă: $\\text{Var}(X_t) = \\sigma^2$",
                    "Distribuția Normală",
                    "Autocovarianță care depinde doar de lag"
                ],
                "correctExplanation": "Staționaritatea slabă cere doar medie constantă, varianță constantă (finită) și autocovarianțe care depind numai de lag. Nu face nicio ipoteză asupra distribuției.",
                "incorrectExplanation": "Media constantă, varianța constantă și autocovarianța care depinde doar de lag sînt chiar cele trei condiții ale staționarității slabe. Distribuția Normală nu se numără printre ele."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Variance of a random walk",
                "text": "For a random walk $X_t = X_{t-1} + \\varepsilon_t$ with $X_0 = 100$ and $\\sigma^2 = 4$, what is $\\text{Var}(X_{25})$?",
                "options": [
                    "4",
                    "25",
                    "100",
                    "625"
                ],
                "correctExplanation": "$X_{25} = X_0 + \\sum_{i=1}^{25}\\varepsilon_i$ with $X_0$ fixed, so $\\text{Var}(X_{25}) = 25 \\cdot \\sigma^2 = 25 \\times 4 = 100$.",
                "incorrectExplanation": "4 is the variance of a single shock, 25 is the number of shocks, and 625 is $25^2$. The starting value does not affect the variance; the 25 independent shocks of variance 4 give $25 \\times 4 = 100$."
            },
            "ro": {
                "title": "Varianța unui mers aleator",
                "text": "Pentru un mers aleator $X_t = X_{t-1} + \\varepsilon_t$, cu $X_0 = 100$ și $\\sigma^2 = 4$, cît este $\\text{Var}(X_{25})$?",
                "options": [
                    "4",
                    "25",
                    "100",
                    "625"
                ],
                "correctExplanation": "$X_{25} = X_0 + \\sum_{i=1}^{25}\\varepsilon_i$, cu $X_0$ fixat, deci $\\text{Var}(X_{25}) = 25 \\cdot \\sigma^2 = 25 \\times 4 = 100$.",
                "incorrectExplanation": "4 este varianța unui singur șoc, 25 este numărul de șocuri, iar 625 este $25^2$. Valoarea inițială nu influențează varianța; cele 25 de șocuri independente, de varianță 4, dau $25 \\times 4 = 100$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "White noise",
                "text": "Which statement about white noise is FALSE?",
                "options": [
                    "White noise has zero mean",
                    "White noise has constant variance",
                    "White noise must follow the Normal distribution",
                    "White noise has no autocorrelation"
                ],
                "correctExplanation": "White noise only requires zero mean, constant variance and zero autocorrelation at all non-zero lags. Gaussian white noise is a special case, not the definition.",
                "incorrectExplanation": "Zero mean, constant variance and absence of autocorrelation are all part of the definition, so those statements are true. The false one is the claim that white noise must be Normally distributed."
            },
            "ro": {
                "title": "Zgomot alb",
                "text": "Care afirmație despre zgomotul alb este FALSĂ?",
                "options": [
                    "Zgomotul alb are medie zero",
                    "Zgomotul alb are varianță constantă",
                    "Zgomotul alb trebuie să urmeze distribuția Normală",
                    "Zgomotul alb nu are autocorelație"
                ],
                "correctExplanation": "Zgomotul alb cere doar medie zero, varianță constantă și autocorelație nulă la toate lag-urile nenule. Zgomotul alb gaussian este un caz particular, nu definiția.",
                "incorrectExplanation": "Media zero, varianța constantă și absența autocorelației fac toate parte din definiție, deci acele afirmații sînt adevărate. Falsă este afirmația că zgomotul alb trebuie să urmeze distribuția Normală."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ACF of an AR(1) process",
                "text": "For an AR(1) process with $\\phi = 0.8$, the ACF:",
                "options": [
                    "Cuts off after lag 1",
                    "Decays exponentially",
                    "Oscillates in sign around zero",
                    "Is zero at all lags"
                ],
                "correctExplanation": "For AR(1), $\\rho(h) = \\phi^h = 0.8^h$, which decays exponentially (geometrically) towards zero and stays positive because $\\phi > 0$.",
                "incorrectExplanation": "A cut-off after lag 1 is the signature of MA(1), sign oscillation would require $\\phi < 0$, and a zero ACF corresponds to white noise. With $\\phi = 0.8$ the ACF decays exponentially."
            },
            "ro": {
                "title": "ACF a unui proces AR(1)",
                "text": "Pentru un proces AR(1) cu $\\phi = 0{,}8$, ACF:",
                "options": [
                    "Se anulează după lag-ul 1",
                    "Scade exponențial",
                    "Își alternează semnul în jurul lui zero",
                    "Este zero la toate lag-urile"
                ],
                "correctExplanation": "Pentru AR(1), $\\rho(h) = \\phi^h = 0{,}8^h$, care scade exponențial (geometric) spre zero și rămîne pozitivă, deoarece $\\phi > 0$.",
                "incorrectExplanation": "Anularea după lag-ul 1 este specifică procesului MA(1), alternarea semnului ar cere $\\phi < 0$, iar o ACF nulă corespunde zgomotului alb. Pentru $\\phi = 0{,}8$, ACF scade exponențial."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "PACF of an MA(1) process",
                "text": "The partial autocorrelation function (PACF) of an MA(1) process:",
                "options": [
                    "Cuts off after lag 1",
                    "Decays exponentially",
                    "Is zero at all lags",
                    "Shows a seasonal pattern"
                ],
                "correctExplanation": "An invertible MA(1) can be written as an AR($\\infty$), so its PACF does not cut off: it decays exponentially (with alternating signs when $\\theta > 0$). Its ACF is the one that cuts off after lag 1.",
                "incorrectExplanation": "The cut-off after lag 1 belongs to the ACF of MA(1), not to its PACF. For MA processes the ACF cuts off and the PACF decays; for AR processes it is the other way round."
            },
            "ro": {
                "title": "PACF a unui proces MA(1)",
                "text": "Funcția de autocorelație parțială (PACF) a unui proces MA(1):",
                "options": [
                    "Se anulează după lag-ul 1",
                    "Scade exponențial",
                    "Este zero la toate lag-urile",
                    "Prezintă un tipar sezonier"
                ],
                "correctExplanation": "Un proces MA(1) inversabil se poate scrie ca AR($\\infty$), deci PACF nu se anulează, ci scade exponențial (cu semne alternante cînd $\\theta > 0$). ACF este cea care se anulează după lag-ul 1.",
                "incorrectExplanation": "Anularea după lag-ul 1 caracterizează ACF a procesului MA(1), nu PACF. Pentru procesele MA, ACF se anulează și PACF scade; pentru procesele AR este invers."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Differencing a random walk",
                "text": "If $X_t$ is a random walk, then $\\Delta X_t = X_t - X_{t-1}$ is:",
                "options": [
                    "Still a random walk",
                    "White noise (stationary)",
                    "A stationary AR(1) process with $0 < \\phi < 1$",
                    "A trending series"
                ],
                "correctExplanation": "If $X_t = X_{t-1} + \\varepsilon_t$, then $\\Delta X_t = \\varepsilon_t$, which is white noise.",
                "incorrectExplanation": "Differencing removes the unit root, so the result is no longer a random walk; it has no autoregressive dependence left, so it is not an AR(1) with $0 < \\phi < 1$; and a driftless random walk has no deterministic trend to leave behind. The first difference is the white noise $\\varepsilon_t$."
            },
            "ro": {
                "title": "Diferențierea unui mers aleator",
                "text": "Dacă $X_t$ este un mers aleator, atunci $\\Delta X_t = X_t - X_{t-1}$ este:",
                "options": [
                    "Tot un mers aleator",
                    "Zgomot alb (staționar)",
                    "Un proces AR(1) staționar, cu $0 < \\phi < 1$",
                    "O serie cu trend"
                ],
                "correctExplanation": "Dacă $X_t = X_{t-1} + \\varepsilon_t$, atunci $\\Delta X_t = \\varepsilon_t$, adică zgomot alb.",
                "incorrectExplanation": "Diferențierea elimină rădăcina unitară, deci rezultatul nu mai este un mers aleator; nu mai rămîne nicio dependență autoregresivă, deci nu este un AR(1) cu $0 < \\phi < 1$; iar un mers aleator fără drift nu lasă în urmă niciun trend determinist. Prima diferență este zgomotul alb $\\varepsilon_t$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ADF test",
                "text": "In the augmented Dickey-Fuller (ADF) test, the null hypothesis is:",
                "options": [
                    "The series is stationary",
                    "The series has a unit root",
                    "The series is white noise",
                    "The series has no trend"
                ],
                "correctExplanation": "ADF: $H_0$: unit root (non-stationary). Rejecting $H_0$ is evidence of stationarity (around a constant or a trend, depending on the specification).",
                "incorrectExplanation": "Stationarity as the null hypothesis is the KPSS setup, not ADF; white noise is the null of the Ljung-Box test, and the presence of a trend is handled by the deterministic terms in the regression. The ADF null is a unit root."
            },
            "ro": {
                "title": "Testul ADF",
                "text": "În testul Dickey-Fuller augmentat (ADF), ipoteza nulă este:",
                "options": [
                    "Seria este staționară",
                    "Seria are rădăcină unitară",
                    "Seria este zgomot alb",
                    "Seria nu are trend"
                ],
                "correctExplanation": "ADF: $H_0$: rădăcină unitară (serie nestaționară). Respingerea lui $H_0$ indică staționaritatea (în jurul unei constante sau al unui trend, după specificație).",
                "incorrectExplanation": "Staționaritatea ca ipoteză nulă corespunde testului KPSS, nu ADF; zgomotul alb este ipoteza nulă a testului Ljung-Box, iar prezența trendului se tratează prin termenii determiniști din regresie. Ipoteza nulă ADF este rădăcina unitară."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "KPSS test",
                "text": "If the KPSS test rejects $H_0$, we conclude that:",
                "options": [
                    "The series is stationary",
                    "The series is non-stationary",
                    "The series is white noise",
                    "More data are needed"
                ],
                "correctExplanation": "KPSS: $H_0$: the series is stationary (around a level or a trend). Rejecting $H_0$ is evidence of non-stationarity.",
                "incorrectExplanation": "KPSS has the opposite hypotheses to ADF: stationarity is the null, so rejection cannot mean stationarity; the test does not address white noise, and a rejection is a conclusion, not a request for more data. Rejecting KPSS points to non-stationarity."
            },
            "ro": {
                "title": "Testul KPSS",
                "text": "Dacă testul KPSS respinge $H_0$, concluzionăm că:",
                "options": [
                    "Seria este staționară",
                    "Seria este nestaționară",
                    "Seria este zgomot alb",
                    "Sînt necesare mai multe date"
                ],
                "correctExplanation": "KPSS: $H_0$: seria este staționară (în jurul unui nivel sau al unui trend). Respingerea lui $H_0$ indică nestaționaritatea.",
                "incorrectExplanation": "KPSS are ipoteze opuse față de ADF: staționaritatea este ipoteza nulă, deci respingerea nu poate însemna staționaritate; testul nu privește zgomotul alb, iar o respingere este o concluzie, nu o cerere de date suplimentare. Respingerea în testul KPSS indică nestaționaritatea."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Removing a stochastic trend",
                "text": "How should a stochastic trend (unit root) be removed?",
                "options": [
                    "Fit a linear regression on time",
                    "Apply differencing",
                    "Use a moving average",
                    "Apply seasonal adjustment"
                ],
                "correctExplanation": "A stochastic trend is removed by differencing: $\\Delta X_t$ is stationary when $X_t$ is I(1). A deterministic trend is removed by regression on time.",
                "incorrectExplanation": "Regression on time removes a deterministic trend but leaves the unit root in the residuals, a moving average only smooths the series, and seasonal adjustment targets seasonality. A unit root calls for differencing."
            },
            "ro": {
                "title": "Eliminarea unui trend stochastic",
                "text": "Cum trebuie eliminat un trend stochastic (rădăcină unitară)?",
                "options": [
                    "Prin estimarea unei regresii liniare pe timp",
                    "Prin diferențiere",
                    "Prin aplicarea unei medii mobile",
                    "Prin ajustare sezonieră"
                ],
                "correctExplanation": "Un trend stochastic se elimină prin diferențiere: $\\Delta X_t$ este staționară cînd $X_t$ este I(1). Un trend determinist se elimină prin regresie pe timp.",
                "incorrectExplanation": "Regresia pe timp elimină un trend determinist, dar lasă rădăcina unitară în reziduuri; media mobilă doar netezește seria, iar ajustarea sezonieră privește sezonalitatea. O rădăcină unitară cere diferențiere."
            }
        }
    ]
};
