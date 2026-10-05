// ============================================================
// Chapter 7 quiz bank: Cointegration and VECM (EN + RO)
// 10 questions ported from the 2025/2026 site; 10 drawn per attempt.
// The bank grows to 24 questions when the chapter is rebuilt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['cointegration'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Spurious regression",
                "text": "What does the rule of thumb $R^2 > DW$ indicate in a regression between time series?",
                "options": [
                    "The model is well specified",
                    "A suspicion of spurious regression",
                    "The variables are stationary",
                    "There is no autocorrelation in the residuals"
                ],
                "correctExplanation": "Granger and Newbold's rule of thumb: a high $R^2$ together with a Durbin-Watson statistic below it signals strongly autocorrelated (possibly I(1)) residuals, the typical symptom of a spurious regression between non-cointegrated I(1) series.",
                "incorrectExplanation": "A low DW means strongly autocorrelated residuals, the opposite of a well-specified model with uncorrelated errors. The high $R^2$ is not evidence of stationarity: it is precisely what trending I(1) series produce even when they are unrelated."
            },
            "ro": {
                "title": "Regresia falsă",
                "text": "Ce indică regula empirică $R^2 > DW$ într-o regresie între serii de timp?",
                "options": [
                    "Modelul este bine specificat",
                    "O suspiciune de regresie falsă",
                    "Variabilele sînt staționare",
                    "Reziduurile nu sînt autocorelate"
                ],
                "correctExplanation": "Regula empirică a lui Granger și Newbold: un $R^2$ ridicat însoțit de o statistică Durbin-Watson mai mică decît acesta semnalează reziduuri puternic autocorelate (posibil I(1)), simptomul tipic al unei regresii false între serii I(1) necointegrate.",
                "incorrectExplanation": "Un DW mic înseamnă reziduuri puternic autocorelate, opusul unui model bine specificat, cu erori necorelate. $R^2$ mare nu dovedește staționaritatea: este exact ce produc seriile I(1) cu trend chiar și atunci cînd nu au nicio legătură între ele."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Definition of cointegration",
                "text": "The variables $Y_{1t}$ and $Y_{2t}$ are cointegrated if:",
                "options": [
                    "Both are stationary, I(0)",
                    "Both are I(1) and some nonzero linear combination of them is I(0)",
                    "Their correlation equals 1",
                    "One is I(1) and the other is I(0)"
                ],
                "correctExplanation": "Cointegration (Engle and Granger, 1987): I(1) variables share a common stochastic trend, so a combination $Y_{1t} - \\beta Y_{2t}$ is stationary.",
                "incorrectExplanation": "Stationary variables need no cointegration concept; an I(1) and an I(0) variable cannot have a stationary combination with nonzero weight on the I(1) one; and cointegration is about a stationary combination, not about correlation, which can be high even in a spurious regression."
            },
            "ro": {
                "title": "Definiția cointegrării",
                "text": "Variabilele $Y_{1t}$ și $Y_{2t}$ sînt cointegrate dacă:",
                "options": [
                    "Ambele sînt staționare, I(0)",
                    "Ambele sînt I(1) și o combinație liniară nenulă a lor este I(0)",
                    "Corelația dintre ele este egală cu 1",
                    "Una este I(1), iar cealaltă este I(0)"
                ],
                "correctExplanation": "Cointegrarea (Engle și Granger, 1987): variabilele I(1) au un trend stochastic comun, astfel încît o combinație $Y_{1t} - \\beta Y_{2t}$ este staționară.",
                "incorrectExplanation": "Pentru variabile staționare conceptul de cointegrare nu este necesar; o variabilă I(1) și una I(0) nu pot avea o combinație staționară cu pondere nenulă pe variabila I(1); iar cointegrarea privește o combinație staționară, nu corelația, care poate fi mare și într-o regresie falsă."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The Engle-Granger procedure",
                "text": "In the Engle-Granger procedure, what is tested in step 2?",
                "options": [
                    "The stationarity of the original variables",
                    "The stationarity of the residuals from the cointegrating regression",
                    "The normality of the residuals",
                    "The absence of autocorrelation in the original variables"
                ],
                "correctExplanation": "Step 1 estimates the long-run regression by OLS; step 2 applies an ADF-type test to its residuals, with Engle-Granger (MacKinnon) critical values. Stationary residuals indicate cointegration.",
                "incorrectExplanation": "Checking that the variables are I(1) is a preliminary step, not step 2. Normality and autocorrelation diagnostics do not answer the cointegration question: what matters is whether the equilibrium error is I(0)."
            },
            "ro": {
                "title": "Procedura Engle-Granger",
                "text": "În procedura Engle-Granger, ce se testează în etapa a doua?",
                "options": [
                    "Staționaritatea variabilelor inițiale",
                    "Staționaritatea reziduurilor din regresia de cointegrare",
                    "Normalitatea reziduurilor",
                    "Absența autocorelației în variabilele inițiale"
                ],
                "correctExplanation": "Prima etapă estimează prin OLS regresia de lungă durată; a doua aplică reziduurilor un test de tip ADF, cu valori critice Engle-Granger (MacKinnon). Reziduurile staționare indică cointegrare.",
                "incorrectExplanation": "Verificarea faptului că variabilele sînt I(1) este o etapă preliminară, nu etapa a doua. Testele de normalitate sau de autocorelație nu răspund la întrebarea despre cointegrare: contează dacă eroarea de echilibru este I(0)."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The Johansen test",
                "text": "What is the main advantage of the Johansen test over the Engle-Granger procedure?",
                "options": [
                    "It is faster to compute",
                    "It can identify several cointegrating relationships (the cointegration rank) in a system",
                    "It does not depend on the choice of deterministic terms",
                    "It works only with two variables"
                ],
                "correctExplanation": "Johansen's maximum likelihood approach in a VAR estimates the cointegration rank $r$: with $k$ variables there can be up to $k-1$ cointegrating vectors, all estimated jointly.",
                "incorrectExplanation": "Engle-Granger finds at most one relationship and depends on which variable is normalised. Johansen's critical values do depend on the deterministic terms (constant, trend), and the method is designed for systems of more than two variables; speed is not its point."
            },
            "ro": {
                "title": "Testul Johansen",
                "text": "Care este principalul avantaj al testului Johansen față de procedura Engle-Granger?",
                "options": [
                    "Este mai rapid de calculat",
                    "Poate identifica mai multe relații de cointegrare (rangul de cointegrare) într-un sistem",
                    "Nu depinde de alegerea termenilor determiniști",
                    "Funcționează doar cu două variabile"
                ],
                "correctExplanation": "Abordarea Johansen, de verosimilitate maximă într-un VAR, estimează rangul de cointegrare $r$: cu $k$ variabile pot exista pînă la $k-1$ vectori de cointegrare, estimați simultan.",
                "incorrectExplanation": "Engle-Granger găsește cel mult o relație și depinde de variabila aleasă pentru normalizare. Valorile critice Johansen depind de termenii determiniști (constantă, trend), iar metoda este gîndită pentru sisteme cu mai mult de două variabile; viteza nu este argumentul ei."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Adjustment coefficients in a VECM",
                "text": "In a VECM $\\Delta Y_t = \\alpha \\beta' Y_{t-1} + \\sum_i \\Gamma_i \\Delta Y_{t-i} + \\varepsilon_t$, what do the coefficients $\\alpha$ represent?",
                "options": [
                    "The cointegrating vectors",
                    "The adjustment coefficients (speed of adjustment)",
                    "The short-run lag coefficients",
                    "The error variance"
                ],
                "correctExplanation": "$\\alpha$ measures how strongly, and how fast, each variable responds to last period's deviation from equilibrium $\\beta' Y_{t-1}$.",
                "incorrectExplanation": "The cointegrating vectors are the columns of $\\beta$, the short-run dynamics are in the $\\Gamma_i$ matrices, and the error variance is the covariance matrix of $\\varepsilon_t$. Only $\\alpha$ links the equilibrium error to the changes."
            },
            "ro": {
                "title": "Coeficienții de ajustare într-un VECM",
                "text": "Într-un VECM $\\Delta Y_t = \\alpha \\beta' Y_{t-1} + \\sum_i \\Gamma_i \\Delta Y_{t-i} + \\varepsilon_t$, ce reprezintă coeficienții $\\alpha$?",
                "options": [
                    "Vectorii de cointegrare",
                    "Coeficienții de ajustare (viteza de ajustare)",
                    "Coeficienții lag-urilor de termen scurt",
                    "Varianța erorilor"
                ],
                "correctExplanation": "$\\alpha$ măsoară cît de puternic și cît de repede răspunde fiecare variabilă la abaterea de la echilibru din perioada anterioară, $\\beta' Y_{t-1}$.",
                "incorrectExplanation": "Vectorii de cointegrare sînt coloanele lui $\\beta$, dinamica de termen scurt se află în matricele $\\Gamma_i$, iar varianța erorilor este matricea de covarianță a lui $\\varepsilon_t$. Doar $\\alpha$ leagă eroarea de echilibru de variații."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Weak exogeneity",
                "text": "A variable is weakly exogenous (for the long-run parameters) in a VECM if:",
                "options": [
                    "Its adjustment coefficients are zero ($\\alpha_i = 0$)",
                    "Its coefficient in the cointegrating vector is zero ($\\beta_i = 0$)",
                    "It is stationary",
                    "It has no lags in the model"
                ],
                "correctExplanation": "If $\\alpha_i = 0$ the variable does not react to the disequilibrium: it does not error-correct and drives the common trend instead.",
                "incorrectExplanation": "$\\beta_i = 0$ means the variable is excluded from the long-run relationship, a different restriction. Stationarity and the lag length have nothing to do with weak exogeneity, which is a restriction on the loadings $\\alpha$."
            },
            "ro": {
                "title": "Exogenitatea slabă",
                "text": "O variabilă este slab exogenă (pentru parametrii de lungă durată) într-un VECM dacă:",
                "options": [
                    "Coeficienții săi de ajustare sînt nuli ($\\alpha_i = 0$)",
                    "Coeficientul său din vectorul de cointegrare este nul ($\\beta_i = 0$)",
                    "Este staționară",
                    "Nu are lag-uri în model"
                ],
                "correctExplanation": "Dacă $\\alpha_i = 0$, variabila nu reacționează la dezechilibru: nu se corectează și, dimpotrivă, antrenează trendul comun.",
                "incorrectExplanation": "$\\beta_i = 0$ înseamnă că variabila este exclusă din relația de lungă durată, o restricție diferită. Staționaritatea și numărul de lag-uri nu au legătură cu exogenitatea slabă, care este o restricție asupra coeficienților $\\alpha$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Economic interpretation",
                "text": "Cointegration between consumption and income represents:",
                "options": [
                    "A short-run equilibrium relationship",
                    "A long-run equilibrium relationship",
                    "The absence of any relationship",
                    "One-way causality"
                ],
                "correctExplanation": "Cointegration means the two series cannot drift apart permanently: the consumption-income ratio is anchored by a long-run equilibrium, while short-run deviations are allowed.",
                "incorrectExplanation": "Short-run deviations from the equilibrium are allowed and are modelled by the error-correction dynamics. Cointegration implies Granger causality in at least one direction, but not necessarily one-way causality."
            },
            "ro": {
                "title": "Interpretarea economică",
                "text": "Cointegrarea dintre consum și venit reprezintă:",
                "options": [
                    "O relație de echilibru pe termen scurt",
                    "O relație de echilibru pe termen lung",
                    "Absența oricărei relații",
                    "O cauzalitate unidirecțională"
                ],
                "correctExplanation": "Cointegrarea înseamnă că cele două serii nu se pot îndepărta permanent una de cealaltă: raportul consum-venit este ancorat de un echilibru pe termen lung, iar abaterile pe termen scurt sînt permise.",
                "incorrectExplanation": "Abaterile pe termen scurt de la echilibru sînt permise și sînt modelate de dinamica de corecție a erorilor. Cointegrarea implică o cauzalitate Granger în cel puțin un sens, dar nu neapărat una unidirecțională."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Cointegration rank",
                "text": "For 3 I(1) variables, what is the maximum number of cointegrating relationships?",
                "options": [
                    "1",
                    "2",
                    "3",
                    "As many as the number of lags in the VAR"
                ],
                "correctExplanation": "With $k$ I(1) variables the cointegration rank satisfies $0 \\le r \\le k-1$, here $3-1=2$. There must be at least one common stochastic trend.",
                "incorrectExplanation": "A rank of $r = k = 3$ would make every combination stationary, i.e. the variables themselves I(0), contradicting the assumption. The rank does not depend on the number of lags, and one relationship is possible but not the maximum."
            },
            "ro": {
                "title": "Rangul de cointegrare",
                "text": "Pentru 3 variabile I(1), care este numărul maxim de relații de cointegrare?",
                "options": [
                    "1",
                    "2",
                    "3",
                    "Atîtea cîte lag-uri are VAR-ul"
                ],
                "correctExplanation": "Cu $k$ variabile I(1), rangul de cointegrare respectă $0 \\le r \\le k-1$, aici $3-1=2$. Trebuie să existe cel puțin un trend stochastic comun.",
                "incorrectExplanation": "Un rang $r = k = 3$ ar face ca orice combinație să fie staționară, adică variabilele însele să fie I(0), în contradicție cu ipoteza. Rangul nu depinde de numărul de lag-uri, iar o singură relație este posibilă, dar nu este maximul."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The error-correction term",
                "text": "In a VECM, the term $\\beta' Y_{t-1}$ represents:",
                "options": [
                    "The change in the variables",
                    "The deviation from the long-run equilibrium",
                    "The forecast error",
                    "The model constant"
                ],
                "correctExplanation": "$\\beta' Y_{t-1}$ is the equilibrium error in the previous period: how far the variables are from the long-run relationship. Multiplied by $\\alpha$, it pulls the changes back towards equilibrium.",
                "incorrectExplanation": "The changes are $\\Delta Y_t$, on the left-hand side; the forecast error is $\\varepsilon_t$; any constant is a separate deterministic term (possibly restricted inside the cointegrating relation)."
            },
            "ro": {
                "title": "Termenul de corecție a erorilor",
                "text": "Într-un VECM, termenul $\\beta' Y_{t-1}$ reprezintă:",
                "options": [
                    "Variația variabilelor",
                    "Abaterea de la echilibrul pe termen lung",
                    "Eroarea de prognoză",
                    "Constanta modelului"
                ],
                "correctExplanation": "$\\beta' Y_{t-1}$ este eroarea de echilibru din perioada anterioară: cît de departe sînt variabilele de relația de lungă durată. Înmulțit cu $\\alpha$, el readuce variațiile spre echilibru.",
                "incorrectExplanation": "Variațiile sînt $\\Delta Y_t$, în membrul stîng; eroarea de prognoză este $\\varepsilon_t$; o eventuală constantă este un termen determinist separat (eventual restricționat în relația de cointegrare)."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Practical applications",
                "text": "Pairs trading in finance exploits:",
                "options": [
                    "A perfect positive correlation between returns",
                    "Cointegration between asset prices",
                    "Identical returns of the two assets",
                    "Constant volatility"
                ],
                "correctExplanation": "Pairs trading bets on the mean reversion of the spread between cointegrated prices: when the spread is unusually wide, sell the expensive asset and buy the cheap one.",
                "incorrectExplanation": "High correlation of returns does not stop prices from drifting apart, so the spread need not revert. Identical returns would leave nothing to trade, and constant volatility is not required. The strategy needs a stationary spread, i.e. cointegrated prices."
            },
            "ro": {
                "title": "Aplicații practice",
                "text": "Strategia pairs trading din finanțe exploatează:",
                "options": [
                    "O corelație perfect pozitivă între randamente",
                    "Cointegrarea dintre prețurile activelor",
                    "Randamente identice ale celor două active",
                    "O volatilitate constantă"
                ],
                "correctExplanation": "Pairs trading mizează pe revenirea la medie a spread-ului dintre prețuri cointegrate: cînd spread-ul este neobișnuit de mare, se vinde activul scump și se cumpără cel ieftin.",
                "incorrectExplanation": "O corelație mare a randamentelor nu împiedică prețurile să se îndepărteze, deci spread-ul nu revine neapărat. Randamentele identice nu ar lăsa nimic de tranzacționat, iar volatilitatea constantă nu este necesară. Strategia are nevoie de un spread staționar, adică de prețuri cointegrate."
            }
        }
    ]
};
