// ============================================================
// Chapter 2 quiz bank: ARMA models (EN + RO)
// 21 questions ported from the 2025/2026 site; 20 drawn per attempt.
// The bank grows to 24 questions when the chapter is rebuilt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['arma'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 2,
            "en": {
                "title": "AR(1) stationarity",
                "text": "For which value of $\\phi$ is the AR(1) process $X_t = c + \\phi X_{t-1} + \\varepsilon_t$ stationary?",
                "options": [
                    "$\\phi = 1.2$",
                    "$\\phi = 1.0$",
                    "$\\phi = -0.8$",
                    "$\\phi = -1.5$"
                ],
                "correctExplanation": "AR(1) is stationary if and only if $|\\phi| < 1$. Only $|-0.8| = 0.8 < 1$.",
                "incorrectExplanation": "The condition is on the modulus, so a negative $\\phi$ is fine as long as $|\\phi|<1$. $\\phi = 1$ is a unit root (random walk), and $|\\phi| > 1$ gives an explosive process, whatever the sign."
            },
            "ro": {
                "title": "Staționaritatea AR(1)",
                "text": "Pentru ce valoare a lui $\\phi$ este staționar procesul AR(1) $X_t = c + \\phi X_{t-1} + \\varepsilon_t$?",
                "options": [
                    "$\\phi = 1{,}2$",
                    "$\\phi = 1{,}0$",
                    "$\\phi = -0{,}8$",
                    "$\\phi = -1{,}5$"
                ],
                "correctExplanation": "AR(1) este staționar dacă și numai dacă $|\\phi| < 1$. Doar $|-0{,}8| = 0{,}8 < 1$.",
                "incorrectExplanation": "Condiția privește modulul, deci un $\\phi$ negativ este acceptabil atîta timp cît $|\\phi|<1$. $\\phi = 1$ înseamnă rădăcină unitară (mers aleator), iar $|\\phi| > 1$ dă un proces exploziv, indiferent de semn."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Recognising ACF and PACF patterns",
                "text": "The ACF has a single significant spike at lag 1 and then cuts off, while the PACF decays gradually. Which model is suggested?",
                "options": [
                    "AR(1)",
                    "MA(1)",
                    "ARMA(1,1)",
                    "White noise"
                ],
                "correctExplanation": "An ACF that cuts off after lag 1 indicates an MA(1); the gradually decaying PACF confirms it.",
                "incorrectExplanation": "For an AR(1) the pattern is reversed (ACF decays, PACF cuts off after lag 1); for an ARMA(1,1) both decay; white noise has no significant spikes at all."
            },
            "ro": {
                "title": "Recunoașterea tiparelor ACF și PACF",
                "text": "ACF-ul are un singur vîrf semnificativ la lag-ul 1 și apoi se anulează, iar PACF-ul scade treptat. Ce model este sugerat?",
                "options": [
                    "AR(1)",
                    "MA(1)",
                    "ARMA(1,1)",
                    "Zgomot alb"
                ],
                "correctExplanation": "Un ACF care se anulează după lag-ul 1 indică un MA(1); PACF-ul care scade treptat confirmă acest lucru.",
                "incorrectExplanation": "Pentru un AR(1) tiparul este inversat (ACF scade treptat, PACF se anulează după lag-ul 1); pentru un ARMA(1,1) ambele scad treptat; zgomotul alb nu are niciun vîrf semnificativ."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "MA(1) invertibility",
                "text": "Is the MA(1) process $X_t = \\varepsilon_t + 1.5\\varepsilon_{t-1}$ invertible?",
                "options": [
                    "Yes, MA processes are always invertible",
                    "Yes, because $\\theta = 1.5 > 0$",
                    "No, because $|\\theta| = 1.5 > 1$",
                    "No, MA processes are never invertible"
                ],
                "correctExplanation": "Invertibility requires $|\\theta| < 1$. Here $|\\theta| = 1.5 > 1$, so the process is NOT invertible (although, like every finite MA, it is stationary).",
                "incorrectExplanation": "MA processes are always STATIONARY, not always invertible; invertibility depends on $|\\theta|$, not on its sign. With $|\\theta|<1$ an MA process is invertible, so it is not true that MA processes are never invertible."
            },
            "ro": {
                "title": "Invertibilitatea MA(1)",
                "text": "Este invertibil procesul MA(1) $X_t = \\varepsilon_t + 1{,}5\\varepsilon_{t-1}$?",
                "options": [
                    "Da, procesele MA sînt întotdeauna invertibile",
                    "Da, deoarece $\\theta = 1{,}5 > 0$",
                    "Nu, deoarece $|\\theta| = 1{,}5 > 1$",
                    "Nu, procesele MA nu sînt niciodată invertibile"
                ],
                "correctExplanation": "Invertibilitatea cere $|\\theta| < 1$. Aici $|\\theta| = 1{,}5 > 1$, deci procesul NU este invertibil (deși, ca orice MA finit, este staționar).",
                "incorrectExplanation": "Procesele MA sînt întotdeauna STAȚIONARE, nu întotdeauna invertibile; invertibilitatea depinde de $|\\theta|$, nu de semnul lui. Pentru $|\\theta|<1$ un proces MA este invertibil, deci nu este adevărat că procesele MA nu sînt niciodată invertibile."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "ARMA representation",
                "text": "Which model does the compact form $\\phi(L)X_t = \\theta(L)\\varepsilon_t$ represent, with $\\phi(L)$ of degree $p \\ge 1$ and $\\theta(L)$ of degree $q \\ge 1$?",
                "options": [
                    "A pure AR model",
                    "A pure MA model",
                    "An ARMA($p,q$) model",
                    "An exponential smoothing model"
                ],
                "correctExplanation": "$\\phi(L)$ is the AR polynomial and $\\theta(L)$ the MA polynomial, so the equation describes an ARMA($p,q$) process.",
                "incorrectExplanation": "A pure AR model has $\\theta(L) = 1$, a pure MA model has $\\phi(L) = 1$. Exponential smoothing corresponds to ARIMA models with a differencing factor $(1-L)$, which this form does not contain."
            },
            "ro": {
                "title": "Reprezentarea ARMA",
                "text": "Ce model reprezintă forma compactă $\\phi(L)X_t = \\theta(L)\\varepsilon_t$, cu $\\phi(L)$ de grad $p \\ge 1$ și $\\theta(L)$ de grad $q \\ge 1$?",
                "options": [
                    "Un model AR pur",
                    "Un model MA pur",
                    "Un model ARMA($p,q$)",
                    "Un model de netezire exponențială"
                ],
                "correctExplanation": "$\\phi(L)$ este polinomul AR, iar $\\theta(L)$ este polinomul MA, deci ecuația descrie un proces ARMA($p,q$).",
                "incorrectExplanation": "Un model AR pur are $\\theta(L) = 1$, iar un model MA pur are $\\phi(L) = 1$. Netezirea exponențială corespunde unor modele ARIMA cu factorul de diferențiere $(1-L)$, pe care această formă nu îl conține."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The lag operator",
                "text": "What is $(1-L)^2 X_t$?",
                "options": [
                    "$X_t - X_{t-1}$",
                    "$X_t - 2X_{t-1} + X_{t-2}$",
                    "$X_t + X_{t-1} + X_{t-2}$",
                    "$X_t - X_{t-2}$"
                ],
                "correctExplanation": "$(1-L)^2 = 1 - 2L + L^2$, so $(1-L)^2 X_t = X_t - 2X_{t-1} + X_{t-2}$ (the second difference).",
                "incorrectExplanation": "$X_t - X_{t-1}$ is only the first difference and $X_t - X_{t-2}$ is $(1-L^2)X_t$. Expanding the square gives the middle term $-2L$."
            },
            "ro": {
                "title": "Operatorul lag",
                "text": "Cît este $(1-L)^2 X_t$?",
                "options": [
                    "$X_t - X_{t-1}$",
                    "$X_t - 2X_{t-1} + X_{t-2}$",
                    "$X_t + X_{t-1} + X_{t-2}$",
                    "$X_t - X_{t-2}$"
                ],
                "correctExplanation": "$(1-L)^2 = 1 - 2L + L^2$, deci $(1-L)^2 X_t = X_t - 2X_{t-1} + X_{t-2}$ (a doua diferență).",
                "incorrectExplanation": "$X_t - X_{t-1}$ este doar prima diferență, iar $X_t - X_{t-2}$ este $(1-L^2)X_t$. Dezvoltarea pătratului dă termenul din mijloc $-2L$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Information criteria",
                "text": "ARMA(1,1) and ARMA(2,1) are compared using BIC. Which statement is correct?",
                "options": [
                    "A lower BIC always means better forecasts",
                    "BIC penalises complexity less than AIC",
                    "The model with the lower BIC is preferred",
                    "BIC can only compare models with the same number of parameters"
                ],
                "correctExplanation": "BIC $= -2\\ln L + k\\ln T$: a lower value means a better trade-off between fit and complexity, so the model with the lower BIC is preferred.",
                "incorrectExplanation": "BIC penalises each parameter by $\\ln T$, MORE than AIC's 2 as soon as $T \\ge 8$. Its purpose is precisely to compare models of different size, and a lower BIC does not guarantee better out-of-sample forecasts."
            },
            "ro": {
                "title": "Criterii informaționale",
                "text": "Modelele ARMA(1,1) și ARMA(2,1) sînt comparate cu ajutorul BIC. Care afirmație este corectă?",
                "options": [
                    "Un BIC mai mic înseamnă întotdeauna prognoze mai bune",
                    "BIC penalizează complexitatea mai puțin decît AIC",
                    "Este preferat modelul cu BIC mai mic",
                    "BIC poate compara doar modele cu același număr de parametri"
                ],
                "correctExplanation": "BIC $= -2\\ln L + k\\ln T$: o valoare mai mică înseamnă un compromis mai bun între ajustare și complexitate, deci este preferat modelul cu BIC mai mic.",
                "incorrectExplanation": "BIC penalizează fiecare parametru cu $\\ln T$, MAI MULT decît penalizarea 2 a AIC îndată ce $T \\ge 8$. Scopul său este tocmai compararea unor modele de dimensiuni diferite, iar un BIC mai mic nu garantează prognoze mai bune în afara eșantionului."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The Ljung-Box test",
                "text": "After fitting an ARMA model, the Ljung-Box test on the residuals gives a $p$-value of 0.03. What does this mean?",
                "options": [
                    "The model is adequate: the residuals are white noise",
                    "The model is inadequate: the residuals are autocorrelated",
                    "The sample size must be increased",
                    "The test is inconclusive"
                ],
                "correctExplanation": "$p < 0.05$ rejects $H_0$ (no autocorrelation up to the chosen lag): autocorrelation remains in the residuals, so the model is inadequate at the 5% level.",
                "incorrectExplanation": "White-noise residuals would give a large $p$-value. A small $p$-value is a clear rejection, not an inconclusive result, and it calls for re-specifying the model rather than collecting more data."
            },
            "ro": {
                "title": "Testul Ljung-Box",
                "text": "După estimarea unui model ARMA, testul Ljung-Box aplicat reziduurilor dă o $p$-valoare de 0,03. Ce înseamnă acest rezultat?",
                "options": [
                    "Modelul este adecvat: reziduurile sînt zgomot alb",
                    "Modelul este inadecvat: reziduurile sînt autocorelate",
                    "Trebuie mărită dimensiunea eșantionului",
                    "Testul este neconcludent"
                ],
                "correctExplanation": "$p < 0{,}05$ respinge $H_0$ (absența autocorelației pînă la lag-ul ales): în reziduuri rămîne autocorelație, deci modelul este inadecvat la pragul de 5%.",
                "incorrectExplanation": "Reziduurile de tip zgomot alb ar da o $p$-valoare mare. O $p$-valoare mică este o respingere clară, nu un rezultat neconcludent, și cere respecificarea modelului, nu colectarea mai multor date."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Long-horizon forecasts",
                "text": "For a stationary AR(1) model, what happens to the forecasts as the horizon $h \\to \\infty$?",
                "options": [
                    "They grow without bound",
                    "They oscillate forever",
                    "They converge to the unconditional mean $\\mu$",
                    "They become more accurate"
                ],
                "correctExplanation": "$\\hat{X}_{n+h|n} = \\mu + \\phi^h(X_n - \\mu) \\to \\mu$ as $h \\to \\infty$, since $|\\phi|<1$: mean reversion.",
                "incorrectExplanation": "Since $\\phi^h \\to 0$, any oscillation (for $\\phi<0$) dies out and nothing explodes. Accuracy worsens with the horizon: the forecast error variance rises towards the unconditional variance."
            },
            "ro": {
                "title": "Prognoze pe orizont lung",
                "text": "Pentru un model AR(1) staționar, ce se întîmplă cu prognozele cînd orizontul $h \\to \\infty$?",
                "options": [
                    "Cresc nelimitat",
                    "Oscilează la nesfîrșit",
                    "Converg la media necondiționată $\\mu$",
                    "Devin mai precise"
                ],
                "correctExplanation": "$\\hat{X}_{n+h|n} = \\mu + \\phi^h(X_n - \\mu) \\to \\mu$ cînd $h \\to \\infty$, deoarece $|\\phi|<1$: revenire la medie.",
                "incorrectExplanation": "Cum $\\phi^h \\to 0$, orice oscilație (pentru $\\phi<0$) se stinge și nimic nu explodează. Precizia scade odată cu orizontul: varianța erorii de prognoză crește spre varianța necondiționată."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "AR(1) variance",
                "text": "For the stationary AR(1) process $X_t = \\phi X_{t-1} + \\varepsilon_t$ ($|\\phi|<1$) with $\\text{Var}(\\varepsilon_t) = \\sigma^2$, what is $\\text{Var}(X_t)$?",
                "options": [
                    "$\\sigma^2$",
                    "$\\sigma^2 / (1 - \\phi)$",
                    "$\\sigma^2 / (1 - \\phi^2)$",
                    "$\\sigma^2 (1 + \\phi^2)$"
                ],
                "correctExplanation": "Taking variances: $\\gamma(0) = \\phi^2 \\gamma(0) + \\sigma^2$, so $\\gamma(0) = \\sigma^2 / (1 - \\phi^2)$.",
                "incorrectExplanation": "$\\sigma^2$ ignores the persistence; $1-\\phi$ appears in the MEAN $c/(1-\\phi)$, not in the variance; $\\sigma^2(1+\\phi^2)$ mimics the MA(1) variance $\\sigma^2(1+\\theta^2)$ and ignores all higher powers of $\\phi$."
            },
            "ro": {
                "title": "Varianța AR(1)",
                "text": "Pentru procesul AR(1) staționar $X_t = \\phi X_{t-1} + \\varepsilon_t$ ($|\\phi|<1$) cu $\\text{Var}(\\varepsilon_t) = \\sigma^2$, cît este $\\text{Var}(X_t)$?",
                "options": [
                    "$\\sigma^2$",
                    "$\\sigma^2 / (1 - \\phi)$",
                    "$\\sigma^2 / (1 - \\phi^2)$",
                    "$\\sigma^2 (1 + \\phi^2)$"
                ],
                "correctExplanation": "Aplicînd varianța: $\\gamma(0) = \\phi^2 \\gamma(0) + \\sigma^2$, deci $\\gamma(0) = \\sigma^2 / (1 - \\phi^2)$.",
                "incorrectExplanation": "$\\sigma^2$ ignoră persistența; $1-\\phi$ apare în MEDIE, $c/(1-\\phi)$, nu în varianță; $\\sigma^2(1+\\phi^2)$ imită varianța unui MA(1), $\\sigma^2(1+\\theta^2)$, și ignoră toate puterile mai mari ale lui $\\phi$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "MA(1) autocorrelation",
                "text": "For an MA(1) process $X_t = \\varepsilon_t + \\theta \\varepsilon_{t-1}$, what is $\\rho(2)$?",
                "options": [
                    "$\\theta / (1 + \\theta^2)$",
                    "$\\theta^2 / (1 + \\theta^2)$",
                    "$\\theta^2$",
                    "0"
                ],
                "correctExplanation": "An MA($q$) has $\\rho(h) = 0$ for $h > q$. For an MA(1), $q = 1$, so $\\rho(2) = 0$.",
                "incorrectExplanation": "$\\theta/(1+\\theta^2)$ is $\\rho(1)$, the only nonzero autocorrelation. $X_t$ and $X_{t-2}$ share no common shock, so their correlation is exactly zero; powers of $\\theta$ belong to AR-type decay."
            },
            "ro": {
                "title": "Autocorelația MA(1)",
                "text": "Pentru un proces MA(1) $X_t = \\varepsilon_t + \\theta \\varepsilon_{t-1}$, cît este $\\rho(2)$?",
                "options": [
                    "$\\theta / (1 + \\theta^2)$",
                    "$\\theta^2 / (1 + \\theta^2)$",
                    "$\\theta^2$",
                    "0"
                ],
                "correctExplanation": "Un MA($q$) are $\\rho(h) = 0$ pentru $h > q$. Pentru MA(1), $q = 1$, deci $\\rho(2) = 0$.",
                "incorrectExplanation": "$\\theta/(1+\\theta^2)$ este $\\rho(1)$, singura autocorelație nenulă. $X_t$ și $X_{t-2}$ nu au niciun șoc comun, deci corelația lor este exact zero; puterile lui $\\theta$ țin de scăderea de tip AR."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "AR(1) mean",
                "text": "For the AR(1) process $X_t = c + \\phi X_{t-1} + \\varepsilon_t$ with $c = 2$ and $\\phi = 0.6$, what is $E[X_t]$?",
                "options": [
                    "2",
                    "3.33",
                    "5",
                    "0.8"
                ],
                "correctExplanation": "$\\mu = c / (1 - \\phi) = 2 / (1 - 0.6) = 2 / 0.4 = 5$.",
                "incorrectExplanation": "The constant $c$ is not the mean: taking expectations, $\\mu = c + \\phi\\mu$, so $\\mu = c/(1-\\phi)$. 3.33 results from dividing $c$ by $\\phi$ instead of $1-\\phi$, and 0.8 from multiplying $c$ by $1-\\phi$."
            },
            "ro": {
                "title": "Media AR(1)",
                "text": "Pentru procesul AR(1) $X_t = c + \\phi X_{t-1} + \\varepsilon_t$ cu $c = 2$ și $\\phi = 0{,}6$, cît este $E[X_t]$?",
                "options": [
                    "2",
                    "3,33",
                    "5",
                    "0,8"
                ],
                "correctExplanation": "$\\mu = c / (1 - \\phi) = 2 / (1 - 0{,}6) = 2 / 0{,}4 = 5$.",
                "incorrectExplanation": "Termenul liber $c$ nu este media: aplicînd media, $\\mu = c + \\phi\\mu$, deci $\\mu = c/(1-\\phi)$. 3,33 rezultă din împărțirea lui $c$ la $\\phi$ în loc de $1-\\phi$, iar 0,8 din înmulțirea lui $c$ cu $1-\\phi$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "AR(2) stationarity",
                "text": "For the AR(2) process $X_t = \\phi_1 X_{t-1} + \\phi_2 X_{t-2} + \\varepsilon_t$, which conditions ensure stationarity?",
                "options": [
                    "$|\\phi_1| < 1$ and $|\\phi_2| < 1$",
                    "$\\phi_1 + \\phi_2 < 1$, $\\phi_2 - \\phi_1 < 1$, $|\\phi_2| < 1$",
                    "$\\phi_1^2 + \\phi_2^2 < 1$",
                    "$|\\phi_1 + \\phi_2| < 1$"
                ],
                "correctExplanation": "The three inequalities $\\phi_1 + \\phi_2 < 1$, $\\phi_2 - \\phi_1 < 1$ and $|\\phi_2| < 1$ define the stationarity triangle, equivalent to both roots of $1 - \\phi_1 z - \\phi_2 z^2$ lying outside the unit circle.",
                "incorrectExplanation": "Bounding each coefficient separately is neither necessary nor sufficient: $\\phi_1 = 1.2$, $\\phi_2 = -0.5$ is stationary, while $\\phi_1 = 0.6$, $\\phi_2 = 0.6$ is not. A unit disc or a single bound on $\\phi_1 + \\phi_2$ also misses parts of the triangle."
            },
            "ro": {
                "title": "Staționaritatea AR(2)",
                "text": "Pentru procesul AR(2) $X_t = \\phi_1 X_{t-1} + \\phi_2 X_{t-2} + \\varepsilon_t$, ce condiții asigură staționaritatea?",
                "options": [
                    "$|\\phi_1| < 1$ și $|\\phi_2| < 1$",
                    "$\\phi_1 + \\phi_2 < 1$, $\\phi_2 - \\phi_1 < 1$, $|\\phi_2| < 1$",
                    "$\\phi_1^2 + \\phi_2^2 < 1$",
                    "$|\\phi_1 + \\phi_2| < 1$"
                ],
                "correctExplanation": "Cele trei inegalități $\\phi_1 + \\phi_2 < 1$, $\\phi_2 - \\phi_1 < 1$ și $|\\phi_2| < 1$ definesc triunghiul de staționaritate, echivalent cu faptul că ambele rădăcini ale lui $1 - \\phi_1 z - \\phi_2 z^2$ se află în afara cercului unitate.",
                "incorrectExplanation": "Limitarea separată a fiecărui coeficient nu este nici necesară, nici suficientă: $\\phi_1 = 1{,}2$, $\\phi_2 = -0{,}5$ este staționar, în timp ce $\\phi_1 = 0{,}6$, $\\phi_2 = 0{,}6$ nu este. Un disc unitate sau o singură limită pentru $\\phi_1 + \\phi_2$ ratează, de asemenea, părți din triunghi."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Yule-Walker equations",
                "text": "The Yule-Walker equations are used to:",
                "options": [
                    "Test for stationarity",
                    "Estimate AR parameters from the autocorrelations",
                    "Estimate MA parameters",
                    "Compute forecasts"
                ],
                "correctExplanation": "Yule-Walker: $\\boldsymbol{\\phi} = \\mathbf{R}^{-1} \\boldsymbol{\\rho}$ links the AR parameters to the ACF; replacing the theoretical autocorrelations with sample ones gives a method-of-moments estimator.",
                "incorrectExplanation": "The equations are linear only for AR models; for MA parameters the moment equations are nonlinear and inefficient. They do not test stationarity and are not a forecasting formula."
            },
            "ro": {
                "title": "Ecuațiile Yule-Walker",
                "text": "Ecuațiile Yule-Walker sînt folosite pentru:",
                "options": [
                    "Testarea staționarității",
                    "Estimarea parametrilor AR pe baza autocorelațiilor",
                    "Estimarea parametrilor MA",
                    "Calculul prognozelor"
                ],
                "correctExplanation": "Yule-Walker: $\\boldsymbol{\\phi} = \\mathbf{R}^{-1} \\boldsymbol{\\rho}$ leagă parametrii AR de ACF; înlocuirea autocorelațiilor teoretice cu cele de selecție dă un estimator prin metoda momentelor.",
                "incorrectExplanation": "Ecuațiile sînt liniare doar pentru modelele AR; pentru parametrii MA, ecuațiile de momente sînt neliniare și ineficiente. Ele nu testează staționaritatea și nu sînt o formulă de prognoză."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Interpreting the PACF",
                "text": "The PACF at lag $k$ measures:",
                "options": [
                    "The total correlation between $X_t$ and $X_{t-k}$",
                    "The correlation between $X_t$ and $X_{t-k}$ after removing the effect of the intermediate lags",
                    "The MA coefficient at lag $k$",
                    "The variance at lag $k$"
                ],
                "correctExplanation": "The PACF is the direct correlation between $X_t$ and $X_{t-k}$ after controlling for $X_{t-1}, \\dots, X_{t-k+1}$; it equals the last coefficient of an AR($k$) regression.",
                "incorrectExplanation": "The total correlation, including indirect links through intermediate lags, is the ACF. The PACF is not an MA coefficient (it relates to AR regressions) and has nothing to do with a variance at lag $k$."
            },
            "ro": {
                "title": "Interpretarea PACF",
                "text": "PACF la lag-ul $k$ măsoară:",
                "options": [
                    "Corelația totală dintre $X_t$ și $X_{t-k}$",
                    "Corelația dintre $X_t$ și $X_{t-k}$ după eliminarea efectului lag-urilor intermediare",
                    "Coeficientul MA la lag-ul $k$",
                    "Varianța la lag-ul $k$"
                ],
                "correctExplanation": "PACF este corelația directă dintre $X_t$ și $X_{t-k}$ după controlul pentru $X_{t-1}, \\dots, X_{t-k+1}$; este egală cu ultimul coeficient al unei regresii AR($k$).",
                "incorrectExplanation": "Corelația totală, inclusiv legăturile indirecte prin lag-urile intermediare, este ACF. PACF nu este un coeficient MA (ține de regresiile AR) și nu are legătură cu o varianță la lag-ul $k$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The parsimony principle",
                "text": "In ARMA modelling, the parsimony principle recommends:",
                "options": [
                    "Always using the highest-order model",
                    "Choosing the simplest adequate model",
                    "Using only AR models",
                    "Always including seasonal terms"
                ],
                "correctExplanation": "Parsimony: among adequate models prefer the one with fewest parameters, to avoid overfitting and to improve out-of-sample performance.",
                "incorrectExplanation": "High-order models fit noise and forecast worse; restricting oneself to AR models can require many lags where a short ARMA would do; seasonal terms belong in the model only when the data are seasonal."
            },
            "ro": {
                "title": "Principiul parcimoniei",
                "text": "În modelarea ARMA, principiul parcimoniei recomandă:",
                "options": [
                    "Folosirea întotdeauna a modelului de ordin maxim",
                    "Alegerea celui mai simplu model adecvat",
                    "Folosirea doar a modelelor AR",
                    "Includerea întotdeauna a termenilor sezonieri"
                ],
                "correctExplanation": "Parcimonie: dintre modelele adecvate se preferă cel cu cei mai puțini parametri, pentru a evita supraajustarea și a îmbunătăți performanța în afara eșantionului.",
                "incorrectExplanation": "Modelele de ordin mare ajustează zgomotul și prognozează mai slab; limitarea la modele AR poate cere multe lag-uri acolo unde un ARMA scurt ar fi suficient; termenii sezonieri își au locul doar dacă datele au sezonalitate."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Characteristic roots",
                "text": "For an AR(1) with $\\phi = 0.9$, the root of the characteristic equation $1 - \\phi z = 0$ is:",
                "options": [
                    "0.9",
                    "$1/0.9 \\approx 1.11$",
                    "-0.9",
                    "0.81"
                ],
                "correctExplanation": "$1 - 0.9z = 0 \\Rightarrow z = 1/0.9 \\approx 1.11$. Since $|z| > 1$ (outside the unit circle), the process is stationary.",
                "incorrectExplanation": "0.9 is the inverse root (the root of $z - \\phi = 0$), a different convention; $-0.9$ has the wrong sign and 0.81 is $\\phi^2$, i.e. $\\rho(2)$. In the lag-polynomial convention stationarity requires the root to lie outside the unit circle."
            },
            "ro": {
                "title": "Rădăcini caracteristice",
                "text": "Pentru un AR(1) cu $\\phi = 0{,}9$, rădăcina ecuației caracteristice $1 - \\phi z = 0$ este:",
                "options": [
                    "0,9",
                    "$1/0{,}9 \\approx 1{,}11$",
                    "-0,9",
                    "0,81"
                ],
                "correctExplanation": "$1 - 0{,}9z = 0 \\Rightarrow z = 1/0{,}9 \\approx 1{,}11$. Deoarece $|z| > 1$ (în afara cercului unitate), procesul este staționar.",
                "incorrectExplanation": "0,9 este rădăcina inversă (rădăcina lui $z - \\phi = 0$), o altă convenție; $-0{,}9$ are semnul greșit, iar 0,81 este $\\phi^2$, adică $\\rho(2)$. În convenția polinomului de lag, staționaritatea cere ca rădăcina să fie în afara cercului unitate."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Estimation methods",
                "text": "Which estimation method is generally preferred for ARMA models?",
                "options": [
                    "Ordinary least squares (OLS)",
                    "Maximum likelihood estimation (MLE)",
                    "The method of moments only",
                    "Simple averaging"
                ],
                "correctExplanation": "MLE is asymptotically efficient and handles the AR and MA parts jointly, including the unobserved past errors.",
                "incorrectExplanation": "OLS works for pure AR models but cannot handle the unobserved MA errors directly; the method of moments (Yule-Walker) is inefficient for MA terms; simple averaging estimates only the mean."
            },
            "ro": {
                "title": "Metode de estimare",
                "text": "Ce metodă de estimare este preferată, în general, pentru modelele ARMA?",
                "options": [
                    "Metoda celor mai mici pătrate (OLS)",
                    "Metoda verosimilității maxime (MLE)",
                    "Exclusiv metoda momentelor",
                    "Medierea simplă"
                ],
                "correctExplanation": "MLE este asimptotic eficientă și tratează simultan partea AR și partea MA, inclusiv erorile trecute neobservate.",
                "incorrectExplanation": "OLS funcționează pentru modelele AR pure, dar nu poate trata direct erorile MA neobservate; metoda momentelor (Yule-Walker) este ineficientă pentru termenii MA; medierea simplă estimează doar media."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Forecast error variance",
                "text": "For a stationary ARMA model, as the forecast horizon increases, the forecast error variance:",
                "options": [
                    "Decreases to zero",
                    "Increases without bound",
                    "Converges to the unconditional variance",
                    "Remains constant"
                ],
                "correctExplanation": "$\\text{Var}(e_{n+h}) = \\sigma^2 \\sum_{j=0}^{h-1} \\psi_j^2 \\to \\sigma^2 \\sum_{j\\ge 0} \\psi_j^2 = \\gamma(0)$ as $h \\to \\infty$.",
                "incorrectExplanation": "The variance rises with $h$ because more future shocks enter the error, so it neither falls nor stays constant. Unbounded growth is the I(1) case; for a stationary model the sum of squared $\\psi$ weights is finite."
            },
            "ro": {
                "title": "Varianța erorii de prognoză",
                "text": "Pentru un model ARMA staționar, pe măsură ce orizontul de prognoză crește, varianța erorii de prognoză:",
                "options": [
                    "Scade la zero",
                    "Crește nelimitat",
                    "Converge la varianța necondiționată",
                    "Rămîne constantă"
                ],
                "correctExplanation": "$\\text{Var}(e_{n+h}) = \\sigma^2 \\sum_{j=0}^{h-1} \\psi_j^2 \\to \\sigma^2 \\sum_{j\\ge 0} \\psi_j^2 = \\gamma(0)$ cînd $h \\to \\infty$.",
                "incorrectExplanation": "Varianța crește odată cu $h$, deoarece în eroare intră tot mai multe șocuri viitoare, deci nici nu scade, nici nu rămîne constantă. Creșterea nelimitată este cazul I(1); pentru un model staționar, suma pătratelor ponderilor $\\psi$ este finită."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The Box-Jenkins methodology",
                "text": "What is the correct order of steps in the Box-Jenkins methodology?",
                "options": [
                    "Estimation → Identification → Diagnostics",
                    "Identification → Estimation → Diagnostics",
                    "Diagnostics → Identification → Estimation",
                    "Estimation → Diagnostics → Identification"
                ],
                "correctExplanation": "Box-Jenkins: (1) identify the model order from the ACF/PACF, (2) estimate the parameters, (3) check the residuals; if the checks fail, go back to step 1.",
                "incorrectExplanation": "Nothing can be estimated before the orders are chosen, and diagnostics need the residuals of an estimated model. The loop returns to identification only after the diagnostic step."
            },
            "ro": {
                "title": "Metodologia Box-Jenkins",
                "text": "Care este ordinea corectă a etapelor în metodologia Box-Jenkins?",
                "options": [
                    "Estimare → Identificare → Diagnosticare",
                    "Identificare → Estimare → Diagnosticare",
                    "Diagnosticare → Identificare → Estimare",
                    "Estimare → Diagnosticare → Identificare"
                ],
                "correctExplanation": "Box-Jenkins: (1) identificarea ordinelor modelului pe baza ACF/PACF, (2) estimarea parametrilor, (3) verificarea reziduurilor; dacă verificările eșuează, se revine la etapa 1.",
                "incorrectExplanation": "Nimic nu poate fi estimat înainte de alegerea ordinelor, iar diagnosticarea are nevoie de reziduurile unui model estimat. Ciclul revine la identificare doar după etapa de diagnosticare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Wold's theorem",
                "text": "According to Wold's decomposition theorem, any covariance-stationary process can be written as:",
                "options": [
                    "A finite-order AR process",
                    "An infinite-order MA process, MA($\\infty$), plus a deterministic component",
                    "A random walk",
                    "Pure white noise"
                ],
                "correctExplanation": "Wold: $X_t = \\sum_{j=0}^{\\infty} \\psi_j \\varepsilon_{t-j} + D_t$, with $\\psi_0 = 1$, $\\sum_j \\psi_j^2 < \\infty$, $\\varepsilon_t$ white noise and $D_t$ deterministic.",
                "incorrectExplanation": "Finite-order AR models are only approximations of the general case; a random walk is not stationary at all; white noise is the special case where all $\\psi_j$ beyond $j=0$ are zero. The theorem justifies using ARMA models as parsimonious approximations of MA($\\infty$)."
            },
            "ro": {
                "title": "Teorema lui Wold",
                "text": "Conform teoremei de descompunere a lui Wold, orice proces staționar în covarianță poate fi scris ca:",
                "options": [
                    "Un proces AR de ordin finit",
                    "Un proces MA de ordin infinit, MA($\\infty$), plus o componentă deterministă",
                    "Un mers aleator",
                    "Un zgomot alb pur"
                ],
                "correctExplanation": "Wold: $X_t = \\sum_{j=0}^{\\infty} \\psi_j \\varepsilon_{t-j} + D_t$, cu $\\psi_0 = 1$, $\\sum_j \\psi_j^2 < \\infty$, $\\varepsilon_t$ zgomot alb și $D_t$ determinist.",
                "incorrectExplanation": "Modelele AR de ordin finit sînt doar aproximări ale cazului general; mersul aleator nu este deloc staționar; zgomotul alb este cazul particular în care toți $\\psi_j$ cu $j \\ge 1$ sînt nuli. Teorema justifică folosirea modelelor ARMA ca aproximări parcimonioase ale MA($\\infty$)."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ACF of an AR(1) process",
                "text": "For an AR(1) with $\\phi = 0.7$, what is the autocorrelation at lag 3, $\\rho(3)$?",
                "options": [
                    "0.7",
                    "0.343",
                    "2.1",
                    "0.49"
                ],
                "correctExplanation": "For an AR(1), $\\rho(k) = \\phi^k$, so $\\rho(3) = 0.7^3 = 0.343$.",
                "incorrectExplanation": "The ACF of an AR(1) decays geometrically: 0.7 is $\\rho(1)$ and 0.49 is $\\rho(2)$, while 2.1 ($3\\phi$) is not even a valid correlation, since $|\\rho| \\le 1$."
            },
            "ro": {
                "title": "ACF-ul procesului AR(1)",
                "text": "Pentru un AR(1) cu $\\phi = 0{,}7$, cît este autocorelația la lag-ul 3, $\\rho(3)$?",
                "options": [
                    "0,7",
                    "0,343",
                    "2,1",
                    "0,49"
                ],
                "correctExplanation": "Pentru un AR(1), $\\rho(k) = \\phi^k$, deci $\\rho(3) = 0{,}7^3 = 0{,}343$.",
                "incorrectExplanation": "ACF-ul unui AR(1) scade geometric: 0,7 este $\\rho(1)$, iar 0,49 este $\\rho(2)$, în timp ce 2,1 ($3\\phi$) nici nu este o corelație validă, deoarece $|\\rho| \\le 1$."
            }
        }
    ]
};
