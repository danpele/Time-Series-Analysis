// ============================================================
// Chapter 1 quiz bank: Stochastic processes and stationarity (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['stationarity'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "Stochastic process",
                "text": "What does a stochastic process $\\{X_t\\}$ describe?",
                "options": [
                    "A family of random variables indexed by time",
                    "A deterministic function of time",
                    "A fixed numerical value",
                    "A sequence of numbers in increasing order"
                ],
                "correctExplanation": "A stochastic process is a collection $\\{X_t : t \\in T\\}$ of random variables, one for each point in time. An observed time series is one realisation of it.",
                "incorrectExplanation": "A deterministic function or a fixed value contains no randomness, and an ordered sequence of numbers is just data. A stochastic process is a family of random variables indexed by time."
            },
            "ro": {
                "title": "Proces stochastic",
                "text": "Ce descrie un proces stochastic $\\{X_t\\}$?",
                "options": [
                    "O familie de variabile aleatoare indexate după timp",
                    "O funcție deterministă de timp",
                    "O valoare numerică fixă",
                    "Un șir de numere ordonate crescător"
                ],
                "correctExplanation": "Un proces stochastic este o colecție $\\{X_t : t \\in T\\}$ de variabile aleatoare, cîte una pentru fiecare moment de timp. O serie de timp observată este o realizare a acestuia.",
                "incorrectExplanation": "O funcție deterministă sau o valoare fixă nu conțin nimic aleator, iar un șir ordonat de numere este doar un set de date. Un proces stochastic este o familie de variabile aleatoare indexate după timp."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Autocovariance function",
                "text": "Which expression defines the autocovariance function $\\gamma(t,s)$?",
                "options": [
                    "$\\mathbb{E}[X_t] \\cdot \\mathbb{E}[X_s]$",
                    "$\\mathbb{E}[(X_t - \\mu_t)(X_s - \\mu_s)]$",
                    "$\\text{Var}(X_t + X_s)$",
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
                    "$\\mathbb{E}[(X_t - \\mu_t)(X_s - \\mu_s)]$",
                    "$\\text{Var}(X_t + X_s)$",
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
            "correct": 0,
            "en": {
                "title": "Ergodicity",
                "text": "What does ergodicity allow in practice?",
                "options": [
                    "Estimating population moments from a single realisation",
                    "Estimating parameters from many realisations of the process",
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
                    "Estimarea momentelor populației dintr-o singură realizare",
                    "Estimarea parametrilor din mai multe realizări ale procesului",
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
            "correct": 2,
            "en": {
                "title": "Lag operator",
                "text": "What is $(1-L)X_t$, where $L$ is the lag operator?",
                "options": [
                    "$X_t + X_{t-1}$",
                    "$X_t \\cdot X_{t-1}$",
                    "$X_t - X_{t-1}$ (the first difference)",
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
                    "$X_t \\cdot X_{t-1}$",
                    "$X_t - X_{t-1}$ (diferența de ordinul întîi)",
                    "$X_t / X_{t-1}$"
                ],
                "correctExplanation": "$L X_t = X_{t-1}$, deci $(1-L)X_t = X_t - X_{t-1} = \\Delta X_t$.",
                "incorrectExplanation": "Operatorul lag este liniar și deplasează seria cu o perioadă înapoi; nu produce sume, produse sau rapoarte între $X_t$ și $X_{t-1}$. Operatorul $(1-L)$ dă diferența de ordinul întîi $\\Delta X_t = X_t - X_{t-1}$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Random walk with drift",
                "text": "For $X_t = \\mu + X_{t-1} + \\varepsilon_t$ with $\\mu > 0$ and $X_0 = 0$, which statement is correct?",
                "options": [
                    "The series is stationary",
                    "$\\mathbb{E}[X_t] = \\mu$ (constant)",
                    "The variance is constant",
                    "$\\mathbb{E}[X_t] = \\mu t$ (grows linearly)"
                ],
                "correctExplanation": "Iterating gives $X_t = \\mu t + \\sum_{i=1}^{t}\\varepsilon_i$, so $\\mathbb{E}[X_t] = \\mu t$ and $\\text{Var}(X_t) = t\\sigma^2$. Both depend on $t$, so the process is non-stationary.",
                "incorrectExplanation": "The drift accumulates: the mean is $\\mu t$, not $\\mu$, and the variance $t\\sigma^2$ grows as well, so the series is neither stationary nor of constant variance."
            },
            "ro": {
                "title": "Mers aleator cu derivă",
                "text": "Pentru $X_t = \\mu + X_{t-1} + \\varepsilon_t$, cu $\\mu > 0$ și $X_0 = 0$, care afirmație este corectă?",
                "options": [
                    "Seria este staționară",
                    "$\\mathbb{E}[X_t] = \\mu$ (constantă)",
                    "Varianța este constantă",
                    "$\\mathbb{E}[X_t] = \\mu t$ (crește liniar)"
                ],
                "correctExplanation": "Prin iterare, $X_t = \\mu t + \\sum_{i=1}^{t}\\varepsilon_i$, deci $\\mathbb{E}[X_t] = \\mu t$ și $\\text{Var}(X_t) = t\\sigma^2$. Ambele depind de $t$, deci procesul este nestaționar.",
                "incorrectExplanation": "Deriva se cumulează: media este $\\mu t$, nu $\\mu$, iar varianța $t\\sigma^2$ crește și ea, deci seria nu este nici staționară, nici de varianță constantă."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "ACF and nonstationarity",
                "text": "An autocorrelation function (ACF) that decays very slowly towards zero suggests that:",
                "options": [
                    "The series is non-stationary (possibly a unit root)",
                    "The series is white noise",
                    "The series is stationary with short memory",
                    "The series is seasonal"
                ],
                "correctExplanation": "A sample ACF that stays high and declines almost linearly over many lags is the classic sign of a unit root. A formal test such as ADF or KPSS should confirm it.",
                "incorrectExplanation": "White noise has ACF close to zero at all non-zero lags, a short-memory stationary series has an ACF that dies out quickly, and seasonality shows up as spikes at multiples of the seasonal lag. Very slow decay points to non-stationarity."
            },
            "ro": {
                "title": "ACF și nestaționaritate",
                "text": "O funcție de autocorelație (ACF) care scade foarte lent spre zero sugerează că:",
                "options": [
                    "Seria este nestaționară (posibil cu rădăcină unitară)",
                    "Seria este zgomot alb",
                    "Seria este staționară, cu memorie scurtă",
                    "Seria are sezonalitate"
                ],
                "correctExplanation": "O ACF de selecție care rămîne ridicată și scade aproape liniar pe multe laguri este semnul clasic al unei rădăcini unitare. Un test formal, de exemplu ADF sau KPSS, trebuie să confirme acest lucru.",
                "incorrectExplanation": "Zgomotul alb are ACF apropiată de zero la toate lagurile nenule, o serie staționară cu memorie scurtă are o ACF care se stinge repede, iar sezonalitatea apare ca vîrfuri la multiplii lagului sezonier. Scăderea foarte lentă indică nestaționaritatea."
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
                "correctExplanation": "ACF este simetrică ($\\rho(h) = \\rho(-h)$), egală cu 1 la lagul 0 și mărginită: $|\\rho(h)| \\leq 1$ pentru orice $h$ (inegalitatea Cauchy-Schwarz).",
                "incorrectExplanation": "Prin construcție $\\rho(0) = \\gamma(0)/\\gamma(0) = 1$, nu 0; $\\rho(h) = 1$ la toate lagurile ar însemna un proces perfect persistent, și nu există niciun motiv ca $\\rho(h)$ să crească. ACF este simetrică și mărginită în valoare absolută de 1."
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
            "correct": 3,
            "en": {
                "title": "Weak stationarity",
                "text": "Which condition is NOT required for weak stationarity?",
                "options": [
                    "Constant mean: $\\mathbb{E}[X_t] = \\mu$",
                    "Constant variance: $\\text{Var}(X_t) = \\sigma^2$",
                    "Autocovariance that depends only on the lag",
                    "A Normal distribution"
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
                    "Autocovarianță care depinde doar de lag",
                    "Distribuția Normală"
                ],
                "correctExplanation": "Staționaritatea slabă cere doar medie constantă, varianță constantă (finită) și autocovarianțe care depind numai de lag. Nu face nicio ipoteză asupra distribuției.",
                "incorrectExplanation": "Media constantă, varianța constantă și autocovarianța care depinde doar de lag sînt chiar cele trei condiții ale staționarității slabe. Distribuția Normală nu se numără printre ele."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Variance of a random walk",
                "text": "For a random walk $X_t = X_{t-1} + \\varepsilon_t$ with $X_0 = 100$ and $\\sigma^2 = 4$, what is $\\text{Var}(X_{25})$?",
                "options": [
                    "100",
                    "4",
                    "25",
                    "625"
                ],
                "correctExplanation": "$X_{25} = X_0 + \\sum_{i=1}^{25}\\varepsilon_i$ with $X_0$ fixed, so $\\text{Var}(X_{25}) = 25 \\cdot \\sigma^2 = 25 \\times 4 = 100$.",
                "incorrectExplanation": "4 is the variance of a single shock, 25 is the number of shocks, and 625 is $25^2$. The starting value does not affect the variance; the 25 independent shocks of variance 4 give $25 \\times 4 = 100$."
            },
            "ro": {
                "title": "Varianța unui mers aleator",
                "text": "Pentru un mers aleator $X_t = X_{t-1} + \\varepsilon_t$, cu $X_0 = 100$ și $\\sigma^2 = 4$, cît este $\\text{Var}(X_{25})$?",
                "options": [
                    "100",
                    "4",
                    "25",
                    "625"
                ],
                "correctExplanation": "$X_{25} = X_0 + \\sum_{i=1}^{25}\\varepsilon_i$, cu $X_0$ fixat, deci $\\text{Var}(X_{25}) = 25 \\cdot \\sigma^2 = 25 \\times 4 = 100$.",
                "incorrectExplanation": "4 este varianța unui singur șoc, 25 este numărul de șocuri, iar 625 este $25^2$. Valoarea inițială nu influențează varianța; cele 25 de șocuri independente, de varianță 4, dau $25 \\times 4 = 100$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "White noise",
                "text": "Which statement about white noise is FALSE?",
                "options": [
                    "White noise has zero mean",
                    "White noise must follow the Normal distribution",
                    "White noise has constant variance",
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
                    "Zgomotul alb trebuie să urmeze distribuția Normală",
                    "Zgomotul alb are varianță constantă",
                    "Zgomotul alb nu are autocorelație"
                ],
                "correctExplanation": "Zgomotul alb cere doar medie zero, varianță constantă și autocorelație nulă la toate lagurile nenule. Zgomotul alb gaussian este un caz particular, nu definiția.",
                "incorrectExplanation": "Media zero, varianța constantă și absența autocorelației fac toate parte din definiție, deci acele afirmații sînt adevărate. Falsă este afirmația că zgomotul alb trebuie să urmeze distribuția Normală."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "ACF of an AR(1) process",
                "text": "For an AR(1) process with $\\phi = 0.8$, the ACF:",
                "options": [
                    "Cuts off after lag 1",
                    "Oscillates in sign around zero",
                    "Decays exponentially",
                    "Is zero at all lags"
                ],
                "correctExplanation": "For AR(1), $\\rho(h) = \\phi^h = 0.8^h$, which decays exponentially (geometrically) towards zero and stays positive because $\\phi > 0$.",
                "incorrectExplanation": "A cut-off after lag 1 is the signature of MA(1), sign oscillation would require $\\phi < 0$, and a zero ACF corresponds to white noise. With $\\phi = 0.8$ the ACF decays exponentially."
            },
            "ro": {
                "title": "ACF a unui proces AR(1)",
                "text": "Pentru un proces AR(1) cu $\\phi = 0{,}8$, ACF:",
                "options": [
                    "Se anulează după lagul 1",
                    "Își alternează semnul în jurul lui zero",
                    "Scade exponențial",
                    "Este zero la toate lagurile"
                ],
                "correctExplanation": "Pentru AR(1), $\\rho(h) = \\phi^h = 0{,}8^h$, care scade exponențial (geometric) spre zero și rămîne pozitivă, deoarece $\\phi > 0$.",
                "incorrectExplanation": "Anularea după lagul 1 este specifică procesului MA(1), alternarea semnului ar cere $\\phi < 0$, iar o ACF nulă corespunde zgomotului alb. Pentru $\\phi = 0{,}8$, ACF scade exponențial."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "PACF of an MA(1) process",
                "text": "The partial autocorrelation function (PACF) of an MA(1) process:",
                "options": [
                    "Cuts off after lag 1",
                    "Is zero at all lags",
                    "Shows a seasonal pattern",
                    "Decays exponentially"
                ],
                "correctExplanation": "An invertible MA(1) can be written as an AR($\\infty$), so its PACF does not cut off: it decays exponentially (with alternating signs when $\\theta > 0$). Its ACF is the one that cuts off after lag 1.",
                "incorrectExplanation": "The cut-off after lag 1 belongs to the ACF of MA(1), not to its PACF. For MA processes the ACF cuts off and the PACF decays; for AR processes it is the other way round."
            },
            "ro": {
                "title": "PACF a unui proces MA(1)",
                "text": "Funcția de autocorelație parțială (PACF) a unui proces MA(1):",
                "options": [
                    "Se anulează după lagul 1",
                    "Este zero la toate lagurile",
                    "Prezintă un tipar sezonier",
                    "Scade exponențial"
                ],
                "correctExplanation": "Un proces MA(1) inversabil se poate scrie ca AR($\\infty$), deci PACF nu se anulează, ci scade exponențial (cu semne alternante cînd $\\theta > 0$). ACF este cea care se anulează după lagul 1.",
                "incorrectExplanation": "Anularea după lagul 1 caracterizează ACF a procesului MA(1), nu PACF. Pentru procesele MA, ACF se anulează și PACF scade; pentru procesele AR este invers."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Differencing a random walk",
                "text": "If $X_t$ is a random walk, then $\\Delta X_t = X_t - X_{t-1}$ is:",
                "options": [
                    "White noise (stationary)",
                    "Still a random walk",
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
                    "Zgomot alb (staționar)",
                    "Tot un mers aleator",
                    "Un proces AR(1) staționar, cu $0 < \\phi < 1$",
                    "O serie cu trend"
                ],
                "correctExplanation": "Dacă $X_t = X_{t-1} + \\varepsilon_t$, atunci $\\Delta X_t = \\varepsilon_t$, adică zgomot alb.",
                "incorrectExplanation": "Diferențierea elimină rădăcina unitară, deci rezultatul nu mai este un mers aleator; nu mai rămîne nicio dependență autoregresivă, deci nu este un AR(1) cu $0 < \\phi < 1$; iar un mers aleator fără derivă nu lasă în urmă niciun trend determinist. Prima diferență este zgomotul alb $\\varepsilon_t$."
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
        },
        {
            "correct": 2,
            "en": {
                "title": "ACF of an MA(1)",
                "text": "For $X_t = \\varepsilon_t + 0.6\\,\\varepsilon_{t-1}$, with $\\varepsilon_t$ white noise, what is $\\rho(1)$?",
                "options": [
                    "$0.6$",
                    "$0.36$",
                    "$0.6/1.36 \\approx 0.44$",
                    "$0$"
                ],
                "correctExplanation": "$\\gamma(0) = \\sigma^2(1 + \\theta^2) = 1.36\\,\\sigma^2$ and $\\gamma(1) = \\theta\\sigma^2 = 0.6\\,\\sigma^2$, so $\\rho(1) = 0.6/1.36 \\approx 0.44$.",
                "incorrectExplanation": "The coefficient 0.6 is not the autocorrelation: it must be divided by $1 + \\theta^2$. The value 0.36 is $\\theta^2$, and the ACF of an MA(1) is zero only from lag 2 on. Here $\\rho(1) = \\theta/(1 + \\theta^2) \\approx 0.44$."
            },
            "ro": {
                "title": "ACF a unui MA(1)",
                "text": "Pentru $X_t = \\varepsilon_t + 0{,}6\\,\\varepsilon_{t-1}$, cu $\\varepsilon_t$ zgomot alb, cît este $\\rho(1)$?",
                "options": [
                    "$0{,}6$",
                    "$0{,}36$",
                    "$0{,}6/1{,}36 \\approx 0{,}44$",
                    "$0$"
                ],
                "correctExplanation": "$\\gamma(0) = \\sigma^2(1 + \\theta^2) = 1{,}36\\,\\sigma^2$ și $\\gamma(1) = \\theta\\sigma^2 = 0{,}6\\,\\sigma^2$, deci $\\rho(1) = 0{,}6/1{,}36 \\approx 0{,}44$.",
                "incorrectExplanation": "Coeficientul 0,6 nu este autocorelația: trebuie împărțit la $1 + \\theta^2$. Valoarea 0,36 este $\\theta^2$, iar ACF a unui MA(1) este nulă abia de la lagul 2. Aici $\\rho(1) = \\theta/(1 + \\theta^2) \\approx 0{,}44$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Confidence band of the sample ACF",
                "text": "A series has $T = 400$ observations. Under the hypothesis of i.i.d. data, what is the 95% band of the sample ACF?",
                "options": [
                    "$\\pm 0.05$",
                    "$\\pm 0.196$",
                    "$\\pm 1.96$",
                    "$\\pm 0.098$"
                ],
                "correctExplanation": "Bartlett: $\\hat\\rho(h) \\approx N(0, 1/T)$, so the band is $\\pm 1.96/\\sqrt{400} = \\pm 1.96/20 = \\pm 0.098$.",
                "incorrectExplanation": "The band shrinks with $\\sqrt{T}$, not with $T$ and not with a fixed number: $\\pm 1.96$ forgets the division, $\\pm 0.196$ uses $\\sqrt{100}$, and $\\pm 0.05$ is a significance level. The band is $\\pm 1.96/\\sqrt{T} = \\pm 0.098$."
            },
            "ro": {
                "title": "Banda de încredere a ACF de selecție",
                "text": "O serie are $T = 400$ de observații. În ipoteza unor date i.i.d., care este banda de 95% a ACF de selecție?",
                "options": [
                    "$\\pm 0{,}05$",
                    "$\\pm 0{,}196$",
                    "$\\pm 1{,}96$",
                    "$\\pm 0{,}098$"
                ],
                "correctExplanation": "Bartlett: $\\hat\\rho(h) \\approx N(0, 1/T)$, deci banda este $\\pm 1{,}96/\\sqrt{400} = \\pm 1{,}96/20 = \\pm 0{,}098$.",
                "incorrectExplanation": "Banda scade cu $\\sqrt{T}$, nu cu $T$ și nu este un număr fix: $\\pm 1{,}96$ uită împărțirea, $\\pm 0{,}196$ folosește $\\sqrt{100}$, iar $\\pm 0{,}05$ este un nivel de semnificație. Banda este $\\pm 1{,}96/\\sqrt{T} = \\pm 0{,}098$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Ljung–Box test",
                "text": "For a series of daily returns, the Ljung–Box statistic is $Q^*(10) = 25.4$; the 5% critical value of $\\chi^2(10)$ is 18.31. What do we conclude?",
                "options": [
                    "We reject the hypothesis that the first 10 autocorrelations are all zero",
                    "The returns are stationary",
                    "The returns are independent",
                    "The returns follow a Normal distribution"
                ],
                "correctExplanation": "$Q^* > \\chi^2_{0.95}(10)$, so at 5% we reject $H_0$: $\\rho(1) = \\dots = \\rho(10) = 0$. At least one autocorrelation differs from zero.",
                "incorrectExplanation": "The Ljung–Box test is about autocorrelation only: it says nothing about stationarity or normality, and not rejecting would still not prove independence. Here $25.4 > 18.31$, so white noise up to lag 10 is rejected."
            },
            "ro": {
                "title": "Testul Ljung–Box",
                "text": "Pentru o serie de randamente zilnice, statistica Ljung–Box este $Q^*(10) = 25{,}4$; valoarea critică de 5% a lui $\\chi^2(10)$ este 18,31. Ce concluzie tragem?",
                "options": [
                    "Respingem ipoteza că primele 10 autocorelații sînt toate nule",
                    "Randamentele sînt staționare",
                    "Randamentele sînt independente",
                    "Randamentele urmează distribuția Normală"
                ],
                "correctExplanation": "$Q^* > \\chi^2_{0{,}95}(10)$, deci la 5% respingem $H_0$: $\\rho(1) = \\dots = \\rho(10) = 0$. Cel puțin o autocorelație diferă de zero.",
                "incorrectExplanation": "Testul Ljung–Box privește doar autocorelația: nu spune nimic despre staționaritate sau normalitate, iar nerespingerea nu ar dovedi independența. Aici $25{,}4 > 18{,}31$, deci ipoteza de zgomot alb pînă la lagul 10 este respinsă."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Over-differencing",
                "text": "A white noise series is differenced once by mistake. What is $\\rho(1)$ of the differenced series?",
                "options": [
                    "$0$",
                    "$-0.5$",
                    "$+0.5$",
                    "$1$"
                ],
                "correctExplanation": "$\\Delta\\varepsilon_t = \\varepsilon_t - \\varepsilon_{t-1}$ is an MA(1) with $\\theta = -1$: $\\gamma(0) = 2\\sigma^2$, $\\gamma(1) = -\\sigma^2$, so $\\rho(1) = -0.5$.",
                "incorrectExplanation": "Differencing is not harmless: consecutive differences share the shock $\\varepsilon_{t-1}$ with opposite signs, so the correlation is negative, not zero or positive. It equals $-\\sigma^2/(2\\sigma^2) = -0.5$."
            },
            "ro": {
                "title": "Supradiferențierea",
                "text": "O serie de zgomot alb este diferențiată o dată, din greșeală. Cît este $\\rho(1)$ al seriei diferențiate?",
                "options": [
                    "$0$",
                    "$-0{,}5$",
                    "$+0{,}5$",
                    "$1$"
                ],
                "correctExplanation": "$\\Delta\\varepsilon_t = \\varepsilon_t - \\varepsilon_{t-1}$ este un MA(1) cu $\\theta = -1$: $\\gamma(0) = 2\\sigma^2$, $\\gamma(1) = -\\sigma^2$, deci $\\rho(1) = -0{,}5$.",
                "incorrectExplanation": "Diferențierea nu este inofensivă: diferențele consecutive au în comun șocul $\\varepsilon_{t-1}$, cu semne opuse, deci corelația este negativă, nu nulă sau pozitivă. Ea este $-\\sigma^2/(2\\sigma^2) = -0{,}5$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Box–Cox transformation",
                "text": "In the Box–Cox family $w = (y^\\lambda - 1)/\\lambda$, which value of $\\lambda$ corresponds to the logarithm?",
                "options": [
                    "$\\lambda = 1$",
                    "$\\lambda = 0.5$",
                    "$\\lambda = 0$ (as the limit)",
                    "$\\lambda = -1$"
                ],
                "correctExplanation": "As $\\lambda \\to 0$, $(y^\\lambda - 1)/\\lambda \\to \\ln y$; the Box–Cox family defines $w = \\ln y$ for $\\lambda = 0$.",
                "incorrectExplanation": "$\\lambda = 1$ only shifts the series, $\\lambda = 0.5$ is a square root and $\\lambda = -1$ an inverse. The log is the limit case $\\lambda = 0$."
            },
            "ro": {
                "title": "Transformarea Box–Cox",
                "text": "În familia Box–Cox $w = (y^\\lambda - 1)/\\lambda$, ce valoare a lui $\\lambda$ corespunde logaritmului?",
                "options": [
                    "$\\lambda = 1$",
                    "$\\lambda = 0{,}5$",
                    "$\\lambda = 0$ (ca limită)",
                    "$\\lambda = -1$"
                ],
                "correctExplanation": "Cînd $\\lambda \\to 0$, $(y^\\lambda - 1)/\\lambda \\to \\ln y$; familia Box–Cox definește $w = \\ln y$ pentru $\\lambda = 0$.",
                "incorrectExplanation": "$\\lambda = 1$ doar deplasează seria, $\\lambda = 0{,}5$ este un radical, iar $\\lambda = -1$ o inversă. Logaritmul este cazul limită $\\lambda = 0$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Returns and their squares",
                "text": "Daily returns have almost no autocorrelation, but their squares are strongly autocorrelated. Which description fits?",
                "options": [
                    "Gaussian white noise",
                    "A random walk",
                    "A non-stationary series with a trend",
                    "Weak white noise: uncorrelated but not independent"
                ],
                "correctExplanation": "Uncorrelated returns with correlated squares are a weak white noise: there is no linear dependence, but the size of the moves is predictable (volatility clustering).",
                "incorrectExplanation": "Correlated squares rule out independence, so the returns are neither i.i.d. nor Gaussian white noise. A random walk or a trend would show up as a slowly decaying ACF of the returns themselves."
            },
            "ro": {
                "title": "Randamentele și pătratele lor",
                "text": "Randamentele zilnice nu au aproape deloc autocorelație, dar pătratele lor sînt puternic autocorelate. Ce descriere li se potrivește?",
                "options": [
                    "Zgomot alb gaussian",
                    "Un mers aleator",
                    "O serie nestaționară cu trend",
                    "Zgomot alb slab: necorelat, dar nu independent"
                ],
                "correctExplanation": "Randamentele necorelate cu pătrate corelate formează un zgomot alb slab: nu există dependență liniară, dar mărimea mișcărilor este previzibilă (volatility clustering).",
                "incorrectExplanation": "Pătratele corelate exclud independența, deci randamentele nu sînt nici zgomot alb i.i.d., nici gaussian. Un mers aleator sau un trend s-ar vedea într-o ACF a randamentelor care scade lent."
            }
        }
    ]
};
