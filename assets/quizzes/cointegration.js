// ============================================================
// Chapter 7 quiz bank: Cointegration and VECM (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['cointegration'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "Spurious regression",
                "text": "What does the rule of thumb $R^2 > DW$ indicate in a regression between time series?",
                "options": [
                    "A suspicion of spurious regression",
                    "The model is well specified",
                    "The variables are stationary",
                    "There is no autocorrelation in the residuals"
                ],
                "correctExplanation": "Granger and Newbold's rule of thumb: a high $R^2$ with a Durbin-Watson statistic below it signals strongly autocorrelated, possibly $I(1)$, residuals, the typical symptom of a regression between unrelated $I(1)$ series.",
                "incorrectExplanation": "A low DW means strongly autocorrelated residuals, the opposite of a well-specified model. A high $R^2$ is exactly what trending $I(1)$ series produce even when they are unrelated."
            },
            "ro": {
                "title": "Regresia falsă",
                "text": "Ce indică regula empirică $R^2 > DW$ într-o regresie între serii de timp?",
                "options": [
                    "O suspiciune de regresie falsă",
                    "Modelul este bine specificat",
                    "Variabilele sînt staționare",
                    "Reziduurile nu sînt autocorelate"
                ],
                "correctExplanation": "Regula empirică a lui Granger și Newbold: un $R^2$ mare, însoțit de o statistică Durbin-Watson mai mică decît el, semnalează reziduuri puternic autocorelate, posibil $I(1)$, simptomul tipic al unei regresii între serii $I(1)$ fără legătură.",
                "incorrectExplanation": "Un DW mic înseamnă reziduuri puternic autocorelate, opusul unui model bine specificat. Un $R^2$ mare este exact ceea ce produc seriile $I(1)$ cu trend, chiar și atunci cînd nu au nicio legătură."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Definition of cointegration",
                "text": "The series $y_{1t}$ and $y_{2t}$ are cointegrated if:",
                "options": [
                    "Both are stationary, $I(0)$",
                    "Both are $I(1)$ and some non-zero linear combination of them is $I(0)$",
                    "Their correlation in levels is close to 1",
                    "One is $I(1)$ and the other is $I(0)$"
                ],
                "correctExplanation": "Cointegration (Engle and Granger 1987): each series has a unit root, but a combination $\\beta^\\top \\mathbf y_t$ is stationary, a long-run equilibrium.",
                "incorrectExplanation": "Stationary series need no cointegration; a high correlation of levels also occurs between unrelated random walks; an $I(0)$ and an $I(1)$ series cannot have a stationary combination with a non-zero weight on the $I(1)$ series."
            },
            "ro": {
                "title": "Definiția cointegrării",
                "text": "Seriile $y_{1t}$ și $y_{2t}$ sînt cointegrate dacă:",
                "options": [
                    "Ambele sînt staționare, $I(0)$",
                    "Ambele sînt $I(1)$ și o combinație liniară nenulă a lor este $I(0)$",
                    "Corelația lor în niveluri este apropiată de 1",
                    "Una este $I(1)$, iar cealaltă $I(0)$"
                ],
                "correctExplanation": "Cointegrarea (Engle și Granger 1987): fiecare serie are o rădăcină unitară, dar o combinație $\\beta^\\top \\mathbf y_t$ este staționară, un echilibru pe termen lung.",
                "incorrectExplanation": "Seriile staționare nu au nevoie de cointegrare; o corelație mare a nivelurilor apare și între mersuri aleatoare fără legătură; o serie $I(0)$ și una $I(1)$ nu pot avea o combinație staționară cu pondere nenulă pe seria $I(1)$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Common stochastic trends",
                "text": "Four $I(1)$ variables have cointegration rank $r = 1$. How many common stochastic trends drive them?",
                "options": [
                    "1",
                    "4",
                    "3",
                    "0"
                ],
                "correctExplanation": "With $n$ variables and $r$ cointegrating vectors there are $n - r$ common stochastic trends (Stock and Watson 1988): $4 - 1 = 3$.",
                "incorrectExplanation": "The rank counts the stationary combinations, not the trends; the number of common trends is $n - r = 3$."
            },
            "ro": {
                "title": "Trenduri stochastice comune",
                "text": "Patru variabile $I(1)$ au rangul de cointegrare $r = 1$. Cîte trenduri stochastice comune le antrenează?",
                "options": [
                    "1",
                    "4",
                    "3",
                    "0"
                ],
                "correctExplanation": "Cu $n$ variabile și $r$ vectori de cointegrare există $n - r$ trenduri stochastice comune (Stock și Watson 1988): $4 - 1 = 3$.",
                "incorrectExplanation": "Rangul numără combinațiile staționare, nu trendurile; numărul trendurilor comune este $n - r = 3$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The Engle-Granger procedure",
                "text": "In the Engle-Granger procedure, what is tested in step 2?",
                "options": [
                    "The stationarity of the original variables",
                    "The normality of the residuals",
                    "The absence of autocorrelation in the original variables",
                    "The stationarity of the residuals of the cointegrating regression"
                ],
                "correctExplanation": "Step 1 estimates $y_t = a + b x_t + u_t$ by OLS; step 2 runs an ADF test on $\\hat u_t$, with $H_0$: unit root in the residuals, i.e. no cointegration.",
                "incorrectExplanation": "The unit-root tests on each variable are a pretest, before step 1. Normality and autocorrelation of the levels are not the question: step 2 asks whether the equilibrium error is stationary."
            },
            "ro": {
                "title": "Procedura Engle-Granger",
                "text": "În procedura Engle-Granger, ce se testează în pasul 2?",
                "options": [
                    "Staționaritatea variabilelor inițiale",
                    "Normalitatea reziduurilor",
                    "Absența autocorelației în variabilele inițiale",
                    "Staționaritatea reziduurilor regresiei de cointegrare"
                ],
                "correctExplanation": "Pasul 1 estimează $y_t = a + b x_t + u_t$ prin OLS; pasul 2 aplică un test ADF pe $\\hat u_t$, cu $H_0$: rădăcină unitară în reziduuri, adică fără cointegrare.",
                "incorrectExplanation": "Testele de rădăcină unitară pe fiecare variabilă sînt o testare prealabilă, înaintea pasului 1. Normalitatea și autocorelația nivelurilor nu sînt întrebarea: pasul 2 verifică dacă eroarea de echilibru este staționară."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Engle-Granger critical values",
                "text": "Why are the Engle-Granger critical values more negative than the Dickey-Fuller ones?",
                "options": [
                    "OLS chooses the slope to minimise the residual variance, so the residuals look more stationary than a true error",
                    "Because the residuals are always normally distributed",
                    "Because the sample is shorter after differencing",
                    "Because the test uses a two-sided alternative"
                ],
                "correctExplanation": "The step-1 regression fits the series as closely as possible; under the null of no cointegration the residual-based statistic is therefore shifted to the left, and its critical values depend on the number of variables (MacKinnon 2010).",
                "incorrectExplanation": "Normality, the sample size after differencing and the alternative do not explain the shift; the test is one-sided. The shift comes from estimating the cointegrating vector before testing."
            },
            "ro": {
                "title": "Valorile critice Engle-Granger",
                "text": "De ce sînt valorile critice Engle-Granger mai negative decît cele Dickey-Fuller?",
                "options": [
                    "OLS alege panta astfel încît să minimizeze varianța reziduurilor, deci reziduurile par mai staționare decît o eroare adevărată",
                    "Pentru că reziduurile au întotdeauna distribuția Normală",
                    "Pentru că eșantionul este mai scurt după diferențiere",
                    "Pentru că testul folosește o alternativă bilaterală"
                ],
                "correctExplanation": "Regresia din pasul 1 ajustează seriile cît mai bine; în ipoteza nulă fără cointegrare, statistica pe reziduuri este deci deplasată la stînga, iar valorile ei critice depind de numărul de variabile (MacKinnon 2010).",
                "incorrectExplanation": "Normalitatea, mărimea eșantionului după diferențiere și alternativa nu explică deplasarea; testul este unilateral. Deplasarea provine din estimarea vectorului de cointegrare înainte de testare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "An Engle-Granger decision",
                "text": "With two variables and $T = 300$, the Engle-Granger statistic is $\\tau = -3.1$; the 5% critical value is about $-3.36$ and the Dickey-Fuller value $-2.87$. What do you conclude at 5%?",
                "options": [
                    "Reject the null: the series are cointegrated",
                    "Do not reject the null of no cointegration",
                    "Reject, because $-3.1 < -2.87$",
                    "The test cannot be used with two variables"
                ],
                "correctExplanation": "The relevant critical value is the Engle-Granger one: $-3.1 > -3.36$, so the null of no cointegration is not rejected at 5%.",
                "incorrectExplanation": "Comparing with the Dickey-Fuller value $-2.87$ would wrongly report cointegration; the Engle-Granger test is designed exactly for two or more variables."
            },
            "ro": {
                "title": "O decizie Engle-Granger",
                "text": "Cu două variabile și $T = 300$, statistica Engle-Granger este $\\tau = -3{,}1$; valoarea critică de 5% este circa $-3{,}36$, iar cea Dickey-Fuller $-2{,}87$. Ce concluzionați la 5%?",
                "options": [
                    "Respingem ipoteza nulă: seriile sînt cointegrate",
                    "Nu respingem ipoteza nulă de absență a cointegrării",
                    "Respingem, deoarece $-3{,}1 < -2{,}87$",
                    "Testul nu poate fi folosit cu două variabile"
                ],
                "correctExplanation": "Valoarea critică relevantă este cea Engle-Granger: $-3{,}1 > -3{,}36$, deci ipoteza nulă de absență a cointegrării nu este respinsă la 5%.",
                "incorrectExplanation": "Comparația cu valoarea Dickey-Fuller $-2{,}87$ ar raporta greșit cointegrare; testul Engle-Granger este construit exact pentru două sau mai multe variabile."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Super-consistency",
                "text": "If $y_t$ and $x_t$ are cointegrated, the OLS slope of $y_t$ on $x_t$ in levels is super-consistent. What does this mean?",
                "options": [
                    "Its usual $t$-statistic has a standard Normal distribution",
                    "It is unbiased in small samples",
                    "Its estimation error shrinks like $1/T$, faster than the usual $1/\\sqrt{T}$",
                    "It equals 1 for every pair of cointegrated series"
                ],
                "correctExplanation": "Because the regressor is $I(1)$ and the error is $I(0)$, the OLS slope converges at rate $T$; yet its OLS standard errors are not valid, which is why dynamic OLS (Stock and Watson 1993) is used for inference.",
                "incorrectExplanation": "Super-consistency does not make the OLS $t$-statistics valid, does not remove the small-sample bias and says nothing about the value of the slope."
            },
            "ro": {
                "title": "Superconsistența",
                "text": "Dacă $y_t$ și $x_t$ sînt cointegrate, panta OLS a lui $y_t$ pe $x_t$ în niveluri este superconsistentă. Ce înseamnă aceasta?",
                "options": [
                    "Statistica ei $t$ obișnuită are distribuția Normală standard",
                    "Este nedeplasată în eșantioane mici",
                    "Eroarea ei de estimare scade ca $1/T$, mai repede decît de obicei ($1/\\sqrt{T}$)",
                    "Este egală cu 1 pentru orice pereche de serii cointegrate"
                ],
                "correctExplanation": "Deoarece regresorul este $I(1)$, iar eroarea $I(0)$, panta OLS converge cu viteza $T$; totuși erorile ei standard din OLS nu sînt valide, motiv pentru care inferența folosește OLS dinamic (Stock și Watson 1993).",
                "incorrectExplanation": "Superconsistența nu face valide statisticile $t$ din OLS, nu elimină deplasarea în eșantioane mici și nu spune nimic despre valoarea pantei."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The Phillips-Ouliaris test",
                "text": "How does the Phillips-Ouliaris $Z_t$ test differ from the Engle-Granger test?",
                "options": [
                    "It has stationarity as the null hypothesis",
                    "It does not need a first-step regression",
                    "It estimates several cointegrating vectors",
                    "It uses no lagged differences and corrects the $t$-statistic with the long-run variance of the residuals"
                ],
                "correctExplanation": "Phillips and Ouliaris (1990) keep the same step 1 and the same null (no cointegration) but replace the lagged differences by a non-parametric long-run variance correction, as the Phillips-Perron test does for a single series.",
                "incorrectExplanation": "The null is still no cointegration; the residuals still come from a first-step regression; only the Johansen method estimates several cointegrating vectors."
            },
            "ro": {
                "title": "Testul Phillips-Ouliaris",
                "text": "Prin ce diferă testul Phillips-Ouliaris $Z_t$ de testul Engle-Granger?",
                "options": [
                    "Are staționaritatea ca ipoteză nulă",
                    "Nu are nevoie de o regresie în primul pas",
                    "Estimează mai mulți vectori de cointegrare",
                    "Nu folosește diferențe decalate și corectează statistica $t$ cu varianța pe termen lung a reziduurilor"
                ],
                "correctExplanation": "Phillips și Ouliaris (1990) păstrează același pas 1 și aceeași ipoteză nulă (fără cointegrare), dar înlocuiesc diferențele decalate printr-o corecție neparametrică cu varianța pe termen lung, așa cum face testul Phillips-Perron pentru o singură serie.",
                "incorrectExplanation": "Ipoteza nulă rămîne absența cointegrării; reziduurile provin tot dintr-o regresie în primul pas; doar metoda Johansen estimează mai mulți vectori de cointegrare."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The error correction coefficient",
                "text": "In the ECM $\\Delta y_t = c + \\gamma (y_{t-1} - a - b x_{t-1}) + \\delta_0 \\Delta x_t + \\varepsilon_t$, which value of $\\gamma$ means error correction?",
                "options": [
                    "$-0.2$",
                    "$+0.2$",
                    "$0$",
                    "$-2.5$"
                ],
                "correctExplanation": "Error correction requires $-2 < \\gamma < 0$: when $y_{t-1}$ is above equilibrium, $\\Delta y_t$ tends to be negative; with $\\gamma = -0.2$, a fifth of the gap closes each period.",
                "incorrectExplanation": "A positive $\\gamma$ pushes $y$ further from equilibrium; $\\gamma = 0$ means no adjustment; $\\gamma = -2.5$ overshoots so much that the gap explodes, since $|1 + \\gamma| > 1$."
            },
            "ro": {
                "title": "Coeficientul de corecție a erorii",
                "text": "În ECM $\\Delta y_t = c + \\gamma (y_{t-1} - a - b x_{t-1}) + \\delta_0 \\Delta x_t + \\varepsilon_t$, ce valoare a lui $\\gamma$ înseamnă corecția erorii?",
                "options": [
                    "$-0{,}2$",
                    "$+0{,}2$",
                    "$0$",
                    "$-2{,}5$"
                ],
                "correctExplanation": "Corecția erorii cere $-2 < \\gamma < 0$: cînd $y_{t-1}$ este peste echilibru, $\\Delta y_t$ tinde să fie negativ; cu $\\gamma = -0{,}2$, o cincime din distanță se închide în fiecare perioadă.",
                "incorrectExplanation": "Un $\\gamma$ pozitiv îndepărtează și mai mult $y$ de echilibru; $\\gamma = 0$ înseamnă lipsa ajustării; $\\gamma = -2{,}5$ corectează atît de mult încît distanța explodează, deoarece $|1 + \\gamma| > 1$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Half-life",
                "text": "An ECM estimated on monthly data gives $\\hat\\gamma = -0.25$. What is the half-life of a deviation from equilibrium?",
                "options": [
                    "About 4 months",
                    "About 2.4 months",
                    "About 0.25 months",
                    "About 25 months"
                ],
                "correctExplanation": "The gap decays like $(1 + \\gamma)^h = 0.75^h$; half of it is gone after $\\ln 0.5/\\ln 0.75 \\approx 2.41$ months.",
                "incorrectExplanation": "The half-life is not $1/|\\gamma|$ (4) nor $|\\gamma|$ itself; it solves $0.75^h = 0.5$."
            },
            "ro": {
                "title": "Timpul de înjumătățire",
                "text": "Un ECM estimat pe date lunare dă $\\hat\\gamma = -0{,}25$. Cît este timpul de înjumătățire al unei abateri de la echilibru?",
                "options": [
                    "Circa 4 luni",
                    "Circa 2,4 luni",
                    "Circa 0,25 luni",
                    "Circa 25 de luni"
                ],
                "correctExplanation": "Distanța scade ca $(1 + \\gamma)^h = 0{,}75^h$; jumătate din ea dispare după $\\ln 0{,}5/\\ln 0{,}75 \\approx 2{,}41$ luni.",
                "incorrectExplanation": "Timpul de înjumătățire nu este $1/|\\gamma|$ (4) și nici $|\\gamma|$; el rezolvă ecuația $0{,}75^h = 0{,}5$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "From a VAR to a VECM",
                "text": "A VAR(2) $\\mathbf y_t = A_1 \\mathbf y_{t-1} + A_2 \\mathbf y_{t-2} + \\mathbf u_t$ is written as a VECM. What is the matrix $\\Pi$ of $\\mathbf y_{t-1}$?",
                "options": [
                    "$A_1 - A_2$",
                    "$-A_2$",
                    "$A_1 + A_2 - I$",
                    "$I - A_1$"
                ],
                "correctExplanation": "Subtract $\\mathbf y_{t-1}$ and add and subtract $A_2 \\mathbf y_{t-1}$: $\\Delta \\mathbf y_t = (A_1 + A_2 - I)\\mathbf y_{t-1} - A_2 \\Delta \\mathbf y_{t-1} + \\mathbf u_t$, so $\\Pi = A_1 + A_2 - I$ and $\\Gamma_1 = -A_2$.",
                "incorrectExplanation": "$-A_2$ is the coefficient $\\Gamma_1$ of the lagged difference; the other expressions do not come out of the rewriting."
            },
            "ro": {
                "title": "De la VAR la VECM",
                "text": "Un VAR(2) $\\mathbf y_t = A_1 \\mathbf y_{t-1} + A_2 \\mathbf y_{t-2} + \\mathbf u_t$ este scris ca VECM. Care este matricea $\\Pi$ a lui $\\mathbf y_{t-1}$?",
                "options": [
                    "$A_1 - A_2$",
                    "$-A_2$",
                    "$A_1 + A_2 - I$",
                    "$I - A_1$"
                ],
                "correctExplanation": "Scădem $\\mathbf y_{t-1}$, apoi adunăm și scădem $A_2 \\mathbf y_{t-1}$: $\\Delta \\mathbf y_t = (A_1 + A_2 - I)\\mathbf y_{t-1} - A_2 \\Delta \\mathbf y_{t-1} + \\mathbf u_t$, deci $\\Pi = A_1 + A_2 - I$ și $\\Gamma_1 = -A_2$.",
                "incorrectExplanation": "$-A_2$ este coeficientul $\\Gamma_1$ al diferenței decalate; celelalte expresii nu rezultă din rescriere."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The rank of $\\Pi$",
                "text": "The Johansen tests give rank$(\\Pi) = 0$ for three $I(1)$ series. Which model should you use?",
                "options": [
                    "A VECM with three cointegrating vectors",
                    "A VAR in levels, since the series are stationary",
                    "A single-equation ECM",
                    "A VAR in differences"
                ],
                "correctExplanation": "Rank 0 means $\\Pi = 0$: no stationary combination, no error correction term; the right model is a VAR for $\\Delta \\mathbf y_t$ (Chapter 6).",
                "incorrectExplanation": "Three cointegrating vectors (full rank) would mean stationary series; a VAR in levels assumes stationarity or keeps the long-run information that does not exist here; an ECM needs a cointegrating relation."
            },
            "ro": {
                "title": "Rangul lui $\\Pi$",
                "text": "Testele Johansen dau rang$(\\Pi) = 0$ pentru trei serii $I(1)$. Ce model folosiți?",
                "options": [
                    "Un VECM cu trei vectori de cointegrare",
                    "Un VAR în niveluri, deoarece seriile sînt staționare",
                    "Un ECM cu o singură ecuație",
                    "Un VAR în diferențe"
                ],
                "correctExplanation": "Rangul 0 înseamnă $\\Pi = 0$: nicio combinație staționară, niciun termen de corecție a erorii; modelul potrivit este un VAR pentru $\\Delta \\mathbf y_t$ (Capitolul 6).",
                "incorrectExplanation": "Trei vectori de cointegrare (rang maxim) ar însemna serii staționare; un VAR în niveluri presupune staționaritate sau păstrează o informație pe termen lung care aici nu există; un ECM cere o relație de cointegrare."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Granger representation theorem",
                "text": "What does the Granger representation theorem state?",
                "options": [
                    "Cointegrated $I(1)$ series have an error correction representation, and conversely",
                    "Every pair of $I(1)$ series is cointegrated",
                    "Cointegrated series cannot Granger-cause each other",
                    "A VAR in differences is always the best model for $I(1)$ series"
                ],
                "correctExplanation": "Engle and Granger (1987): cointegration with rank $r$ is equivalent to a VECM with $\\Pi = \\alpha\\beta^\\top$; a VAR in differences omits $\\alpha\\beta^\\top \\mathbf y_{t-1}$ and is misspecified.",
                "incorrectExplanation": "Most pairs of $I(1)$ series are not cointegrated; cointegration implies Granger causality in at least one direction; a VAR in differences is wrong when the series are cointegrated."
            },
            "ro": {
                "title": "Teorema de reprezentare a lui Granger",
                "text": "Ce afirmă teorema de reprezentare a lui Granger?",
                "options": [
                    "Seriile $I(1)$ cointegrate au o reprezentare cu corecția erorii, și reciproc",
                    "Orice pereche de serii $I(1)$ este cointegrată",
                    "Seriile cointegrate nu se pot cauza reciproc în sens Granger",
                    "Un VAR în diferențe este întotdeauna cel mai bun model pentru serii $I(1)$"
                ],
                "correctExplanation": "Engle și Granger (1987): cointegrarea cu rangul $r$ este echivalentă cu un VECM cu $\\Pi = \\alpha\\beta^\\top$; un VAR în diferențe omite $\\alpha\\beta^\\top \\mathbf y_{t-1}$ și este greșit specificat.",
                "incorrectExplanation": "Majoritatea perechilor de serii $I(1)$ nu sînt cointegrate; cointegrarea implică o cauzalitate Granger în cel puțin o direcție; un VAR în diferențe este greșit cînd seriile sînt cointegrate."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Adjustment coefficients in a VECM",
                "text": "In a VECM $\\Delta \\mathbf y_t = \\alpha \\beta^\\top \\mathbf y_{t-1} + \\sum_i \\Gamma_i \\Delta \\mathbf y_{t-i} + \\mathbf u_t$, what do the coefficients $\\alpha$ represent?",
                "options": [
                    "The cointegrating vectors",
                    "The adjustment coefficients: how strongly each variable reacts to the equilibrium errors",
                    "The short-run lag coefficients",
                    "The error variance"
                ],
                "correctExplanation": "$\\beta^\\top \\mathbf y_{t-1}$ are the equilibrium errors and $\\alpha$ (loadings) measures how each variable responds to them, i.e. the speed of adjustment.",
                "incorrectExplanation": "$\\beta$ holds the cointegrating vectors and $\\Gamma_i$ the short-run dynamics; the error variance is the covariance matrix of $\\mathbf u_t$."
            },
            "ro": {
                "title": "Coeficienții de ajustare într-un VECM",
                "text": "Într-un VECM $\\Delta \\mathbf y_t = \\alpha \\beta^\\top \\mathbf y_{t-1} + \\sum_i \\Gamma_i \\Delta \\mathbf y_{t-i} + \\mathbf u_t$, ce reprezintă coeficienții $\\alpha$?",
                "options": [
                    "Vectorii de cointegrare",
                    "Coeficienții de ajustare: cît de puternic reacționează fiecare variabilă la erorile de echilibru",
                    "Coeficienții decalajelor pe termen scurt",
                    "Varianța erorilor"
                ],
                "correctExplanation": "$\\beta^\\top \\mathbf y_{t-1}$ sînt erorile de echilibru, iar $\\alpha$ (loadings) măsoară cum răspunde fiecare variabilă la ele, adică viteza de ajustare.",
                "incorrectExplanation": "$\\beta$ conține vectorii de cointegrare, iar $\\Gamma_i$ dinamica pe termen scurt; varianța erorilor este matricea de covarianță a lui $\\mathbf u_t$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Weak exogeneity",
                "text": "A variable is weakly exogenous for the cointegrating vectors in a VECM if:",
                "options": [
                    "Its coefficient in the cointegrating vector is zero ($\\beta_i = 0$)",
                    "It is stationary",
                    "Its adjustment coefficients are zero ($\\alpha_i = 0$)",
                    "It has no lags in the model"
                ],
                "correctExplanation": "With $\\alpha_i = 0$ the variable does not react to the equilibrium errors: the other variables do all the adjusting, and it drives the common trend; the US 1-year yield behaves this way in the lecture.",
                "incorrectExplanation": "$\\beta_i = 0$ would remove the variable from the long-run relation; stationarity and the lag structure are different questions."
            },
            "ro": {
                "title": "Exogenitatea slabă",
                "text": "O variabilă este slab exogenă pentru vectorii de cointegrare într-un VECM dacă:",
                "options": [
                    "Coeficientul ei în vectorul de cointegrare este nul ($\\beta_i = 0$)",
                    "Este staționară",
                    "Coeficienții ei de ajustare sînt nuli ($\\alpha_i = 0$)",
                    "Nu are decalaje în model"
                ],
                "correctExplanation": "Cu $\\alpha_i = 0$ variabila nu reacționează la erorile de echilibru: celelalte variabile fac toată ajustarea, iar ea antrenează trendul comun; randamentul la 1 an din SUA se comportă astfel în curs.",
                "incorrectExplanation": "$\\beta_i = 0$ ar scoate variabila din relația pe termen lung; staționaritatea și structura decalajelor sînt alte întrebări."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The Johansen method",
                "text": "What is the main advantage of the Johansen method over the Engle-Granger procedure?",
                "options": [
                    "It is faster to compute",
                    "It does not depend on the deterministic terms",
                    "It works only with two variables",
                    "It tests and estimates several cointegrating relations (the rank) in a system, by maximum likelihood"
                ],
                "correctExplanation": "Johansen (1988, 1991) estimates the VECM by maximum likelihood, tests the rank $r$ with the trace and maximum-eigenvalue statistics and does not depend on which variable is normalised.",
                "incorrectExplanation": "Speed is not the point; the critical values depend strongly on the deterministic terms; the method is designed for systems with any number of variables."
            },
            "ro": {
                "title": "Metoda Johansen",
                "text": "Care este principalul avantaj al metodei Johansen față de procedura Engle-Granger?",
                "options": [
                    "Se calculează mai repede",
                    "Nu depinde de termenii determiniști",
                    "Funcționează doar cu două variabile",
                    "Testează și estimează mai multe relații de cointegrare (rangul) într-un sistem, prin verosimilitate maximă"
                ],
                "correctExplanation": "Johansen (1988, 1991) estimează VECM prin verosimilitate maximă, testează rangul $r$ cu statisticile urmei și a valorii proprii maxime și nu depinde de variabila normalizată.",
                "incorrectExplanation": "Viteza nu este argumentul; valorile critice depind puternic de termenii determiniști; metoda este construită pentru sisteme cu orice număr de variabile."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The trace test sequence",
                "text": "For three yields, the trace test rejects $r = 0$ and $r \\le 1$ but not $r \\le 2$. What do you conclude?",
                "options": [
                    "Rank 2: two cointegrating vectors and one common trend",
                    "Rank 0: no cointegration",
                    "Rank 3: the yields are stationary",
                    "Two common trends"
                ],
                "correctExplanation": "The rank is the first $r$ whose null is not rejected: $r = 2$; with $n = 3$ there is $3 - 2 = 1$ common trend, the level of interest rates.",
                "incorrectExplanation": "The rejections of $r = 0$ and $r \\le 1$ rule out ranks 0 and 1; rank 3 would require rejecting $r \\le 2$; two cointegrating vectors leave one common trend, not two."
            },
            "ro": {
                "title": "Secvența testului urmei",
                "text": "Pentru trei randamente, testul urmei respinge $r = 0$ și $r \\le 1$, dar nu și $r \\le 2$. Ce concluzionați?",
                "options": [
                    "Rangul 2: doi vectori de cointegrare și un singur trend comun",
                    "Rangul 0: fără cointegrare",
                    "Rangul 3: randamentele sînt staționare",
                    "Două trenduri comune"
                ],
                "correctExplanation": "Rangul este primul $r$ pentru care ipoteza nulă nu este respinsă: $r = 2$; cu $n = 3$ există $3 - 2 = 1$ trend comun, nivelul ratelor dobînzii.",
                "incorrectExplanation": "Respingerile pentru $r = 0$ și $r \\le 1$ exclud rangurile 0 și 1; rangul 3 ar cere respingerea lui $r \\le 2$; doi vectori de cointegrare lasă un singur trend comun, nu două."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Deterministic terms",
                "text": "In the lecture, the Johansen rank of the US yields changes from 2 to 3 when a linear trend is added. What is the right lesson?",
                "options": [
                    "Always choose the specification with the highest rank",
                    "Choose the deterministic terms from theory and from the plot before testing, not from the result",
                    "Deterministic terms never matter for the Johansen tests",
                    "Interest rates must be modelled with a linear trend"
                ],
                "correctExplanation": "The critical values and the conclusions depend on the deterministic case; a trend in yields over 66 years is not plausible, so the trend case should not be used for them.",
                "incorrectExplanation": "Picking the specification by its result is data snooping; the deterministic terms matter a lot; interest rates have no long-run linear trend."
            },
            "ro": {
                "title": "Termenii determiniști",
                "text": "În curs, rangul Johansen pentru randamentele din SUA se schimbă de la 2 la 3 cînd se adaugă un trend liniar. Care este lecția corectă?",
                "options": [
                    "Alegem întotdeauna specificația cu rangul cel mai mare",
                    "Alegem termenii determiniști după teorie și după grafic, înainte de testare, nu după rezultat",
                    "Termenii determiniști nu contează niciodată pentru testele Johansen",
                    "Ratele dobînzii trebuie modelate cu trend liniar"
                ],
                "correctExplanation": "Valorile critice și concluziile depind de cazul termenilor determiniști; un trend al randamentelor pe 66 de ani nu este plauzibil, deci cazul cu trend nu se folosește pentru ele.",
                "incorrectExplanation": "Alegerea specificației după rezultat este data snooping; termenii determiniști contează mult; ratele dobînzii nu au un trend liniar pe termen lung."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The error-correction term",
                "text": "In a VECM, the term $\\beta^\\top \\mathbf y_{t-1}$ represents:",
                "options": [
                    "The change in the variables",
                    "The forecast error",
                    "The deviation from the long-run equilibrium",
                    "The constant of the model"
                ],
                "correctExplanation": "$\\beta^\\top \\mathbf y_{t-1}$ are the equilibrium errors of the previous period; through $\\alpha$ they pull the changes back towards equilibrium.",
                "incorrectExplanation": "The changes are $\\Delta \\mathbf y_t$; forecast errors are $\\mathbf u_t$; the constants are separate deterministic terms."
            },
            "ro": {
                "title": "Termenul de corecție a erorii",
                "text": "Într-un VECM, termenul $\\beta^\\top \\mathbf y_{t-1}$ reprezintă:",
                "options": [
                    "Variația variabilelor",
                    "Eroarea de prognoză",
                    "Abaterea de la echilibrul pe termen lung",
                    "Constanta modelului"
                ],
                "correctExplanation": "$\\beta^\\top \\mathbf y_{t-1}$ sînt erorile de echilibru din perioada anterioară; prin $\\alpha$, ele readuc variațiile spre echilibru.",
                "incorrectExplanation": "Variațiile sînt $\\Delta \\mathbf y_t$; erorile de prognoză sînt $\\mathbf u_t$; constantele sînt termeni determiniști separați."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Cointegration rank",
                "text": "For 3 $I(1)$ variables, what is the maximum number of cointegrating relations?",
                "options": [
                    "1",
                    "3",
                    "As many as the lags of the VAR",
                    "2"
                ],
                "correctExplanation": "With $n$ $I(1)$ variables there are at most $n - 1$ independent cointegrating vectors; a rank of $n$ would mean that all series are stationary.",
                "incorrectExplanation": "Rank 3 is full rank (stationary series); the number of lags has nothing to do with the rank."
            },
            "ro": {
                "title": "Rangul de cointegrare",
                "text": "Pentru 3 variabile $I(1)$, care este numărul maxim de relații de cointegrare?",
                "options": [
                    "1",
                    "3",
                    "Cîte decalaje are VAR-ul",
                    "2"
                ],
                "correctExplanation": "Cu $n$ variabile $I(1)$ există cel mult $n - 1$ vectori de cointegrare independenți; un rang egal cu $n$ ar însemna că toate seriile sînt staționare.",
                "incorrectExplanation": "Rangul 3 este rangul maxim (serii staționare); numărul de decalaje nu are legătură cu rangul."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Forecasting with a VECM",
                "text": "According to Christoffersen and Diebold (1998) and the out-of-sample experiment of the lecture, where does a VECM help most?",
                "options": [
                    "In forecasting the cointegrating combinations (e.g. a spread) at long horizons",
                    "In forecasting each level at the one-step horizon",
                    "In beating the random walk for every yield at every horizon",
                    "Nowhere: a VECM is always worse than a VAR in differences"
                ],
                "correctExplanation": "The VECM pulls the equilibrium errors back to their means; in the lecture it reduced the RMSE of the 10y-1y spread forecast at 24-36 months, while for the levels the random walk stayed hard to beat.",
                "incorrectExplanation": "The levels are dominated by the unpredictable common trend; the VECM did not beat the random walk at short horizons; but it did help for the spread."
            },
            "ro": {
                "title": "Prognoza cu VECM",
                "text": "Conform lui Christoffersen și Diebold (1998) și experimentului din curs, unde ajută cel mai mult un VECM?",
                "options": [
                    "La prognoza combinațiilor de cointegrare (de exemplu un spread) pe orizonturi lungi",
                    "La prognoza fiecărui nivel cu un pas înainte",
                    "La depășirea mersului aleator pentru orice randament și orice orizont",
                    "Nicăieri: un VECM este întotdeauna mai slab decît un VAR în diferențe"
                ],
                "correctExplanation": "VECM readuce erorile de echilibru la mediile lor; în curs a redus RMSE al prognozei spread-ului 10 ani minus 1 an la 24-36 de luni, în timp ce pentru niveluri mersul aleator a rămas greu de depășit.",
                "incorrectExplanation": "Nivelurile sînt dominate de trendul comun imprevizibil; VECM nu a depășit mersul aleator pe orizonturi scurte; pentru spread însă a ajutat."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Impulse responses of a VECM",
                "text": "How do the impulse responses of the levels in a cointegrated VECM differ from those of a stationary VAR?",
                "options": [
                    "They always return to zero within a few periods",
                    "They need not return to zero: shocks can have permanent effects on the levels",
                    "They are always zero on impact",
                    "They cannot be computed for a VECM"
                ],
                "correctExplanation": "Cointegrated levels share common stochastic trends, so a shock can shift them permanently (in the lecture the three yields settle about 0.33-0.37 points higher), while the equilibrium errors return to their means.",
                "incorrectExplanation": "Only in a stationary VAR do all responses die out; impact responses are generally non-zero; VECM impulse responses are routinely computed, e.g. with statsmodels."
            },
            "ro": {
                "title": "Răspunsul la impuls într-un VECM",
                "text": "Prin ce diferă răspunsurile la impuls ale nivelurilor într-un VECM cointegrat de cele ale unui VAR staționar?",
                "options": [
                    "Revin întotdeauna la zero în cîteva perioade",
                    "Nu revin neapărat la zero: șocurile pot avea efecte permanente asupra nivelurilor",
                    "Sînt întotdeauna nule la impact",
                    "Nu pot fi calculate pentru un VECM"
                ],
                "correctExplanation": "Nivelurile cointegrate au trenduri stochastice comune, deci un șoc le poate deplasa permanent (în curs, cele trei randamente se stabilizează cu circa 0,33-0,37 puncte mai sus), în timp ce erorile de echilibru revin la mediile lor.",
                "incorrectExplanation": "Doar într-un VAR staționar toate răspunsurile se sting; răspunsurile la impact sînt în general nenule; răspunsurile la impuls ale unui VECM se calculează în mod obișnuit, de exemplu cu statsmodels."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Purchasing power parity for the leu",
                "text": "What did the lecture find for PPP between Romania and the euro area over 2005-2026?",
                "options": [
                    "Strong cointegration with $\\beta = (1, -1, 1)$",
                    "PPP holds exactly every month",
                    "No cointegration: the real exchange rate is not stationary and the leu appreciated in real terms",
                    "The EUR/RON rate is stationary"
                ],
                "correctExplanation": "ADF on the real exchange rate, Engle-Granger and Johansen all fail to reject; prices rose more than the depreciation of the leu, a real appreciation consistent with Balassa-Samuelson convergence.",
                "incorrectExplanation": "No test supported PPP; deviations are large and persistent; EUR/RON itself is $I(1)$."
            },
            "ro": {
                "title": "Paritatea puterii de cumpărare pentru leu",
                "text": "Ce a găsit cursul pentru PPP între România și zona euro în perioada 2005-2026?",
                "options": [
                    "O cointegrare puternică, cu $\\beta = (1, -1, 1)$",
                    "PPP este valabilă exact în fiecare lună",
                    "Nicio cointegrare: cursul real nu este staționar, iar leul s-a apreciat în termeni reali",
                    "Cursul EUR/RON este staționar"
                ],
                "correctExplanation": "ADF pe cursul real, Engle-Granger și Johansen nu resping ipoteza nulă; prețurile au crescut mai mult decît s-a depreciat leul, o apreciere reală în acord cu convergența de tip Balassa-Samuelson.",
                "incorrectExplanation": "Niciun test nu a susținut PPP; abaterile sînt mari și persistente; cursul EUR/RON însuși este $I(1)$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Pairs trading out of sample",
                "text": "Why does a pairs-trading backtest with full-sample parameters overstate the profits?",
                "options": [
                    "Because transaction costs are always zero in practice",
                    "Because cointegrated spreads never revert",
                    "Because the full sample has fewer observations",
                    "The hedge ratio, the spread mean and standard deviation, and the choice of the pair use future data"
                ],
                "correctExplanation": "Parameters and pair selection must use only the formation period; in the lecture the in-sample TLV/BRD rule looked attractive, while the rolling rule on 28 BVB pairs earned close to zero after costs.",
                "incorrectExplanation": "Costs are positive and reduce profits further; cointegrated spreads do revert, but slowly; the sample size is not the issue, the look-ahead is."
            },
            "ro": {
                "title": "Pairs trading în afara eșantionului",
                "text": "De ce supraestimează profiturile un test istoric de pairs trading cu parametrii estimați pe întregul eșantion?",
                "options": [
                    "Pentru că în practică costurile de tranzacționare sînt întotdeauna nule",
                    "Pentru că spread-urile cointegrate nu revin niciodată",
                    "Pentru că întregul eșantion are mai puține observații",
                    "Raportul de acoperire, media și abaterea standard a spread-ului și alegerea perechii folosesc date din viitor"
                ],
                "correctExplanation": "Parametrii și alegerea perechilor trebuie să folosească doar perioada de formare; în curs, regula TLV/BRD în eșantion părea atractivă, iar regula pe ferestre mobile pe 28 de perechi de la BVB a cîștigat aproape zero după costuri.",
                "incorrectExplanation": "Costurile sînt pozitive și reduc și mai mult profiturile; spread-urile cointegrate revin, dar lent; problema nu este mărimea eșantionului, ci privirea în viitor."
            }
        }
    ]
};
