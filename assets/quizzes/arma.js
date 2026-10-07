// ============================================================
// Chapter 2 quiz bank: ARMA models (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['arma'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "AR(1) stationarity",
                "text": "For which value of $\\phi$ is the AR(1) process $X_t = c + \\phi X_{t-1} + \\varepsilon_t$ stationary?",
                "options": [
                    "$\\phi = -0.8$",
                    "$\\phi = 1.2$",
                    "$\\phi = 1.0$",
                    "$\\phi = -1.5$"
                ],
                "correctExplanation": "AR(1) is stationary if and only if $|\\phi| < 1$; only $|-0.8| = 0.8 < 1$.",
                "incorrectExplanation": "The condition is on the modulus, so a negative $\\phi$ is fine as long as $|\\phi| < 1$. $\\phi = 1$ is a unit root (random walk), and $|\\phi| > 1$ gives an explosive process, whatever the sign."
            },
            "ro": {
                "title": "Staționaritatea AR(1)",
                "text": "Pentru ce valoare a lui $\\phi$ este staționar procesul AR(1) $X_t = c + \\phi X_{t-1} + \\varepsilon_t$?",
                "options": [
                    "$\\phi = -0{,}8$",
                    "$\\phi = 1{,}2$",
                    "$\\phi = 1{,}0$",
                    "$\\phi = -1{,}5$"
                ],
                "correctExplanation": "AR(1) este staționar dacă și numai dacă $|\\phi| < 1$; doar $|-0{,}8| = 0{,}8 < 1$.",
                "incorrectExplanation": "Condiția privește modulul, deci un $\\phi$ negativ este acceptabil atîta timp cît $|\\phi| < 1$. $\\phi = 1$ înseamnă rădăcină unitară (mers aleator), iar $|\\phi| > 1$ dă un proces exploziv, indiferent de semn."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The mean of an AR(1)",
                "text": "For $X_t = 2 + 0.6X_{t-1} + \\varepsilon_t$, what is $E[X_t]$?",
                "options": [
                    "2",
                    "1.2",
                    "3.33",
                    "5"
                ],
                "correctExplanation": "Taking expectations with $E[X_t] = E[X_{t-1}] = \\mu$: $\\mu = 2 + 0.6\\mu$, so $\\mu = 2/0.4 = 5$.",
                "incorrectExplanation": "The intercept 2 is not the mean; $1.2 = 2 \\times 0.6$ and $3.33 = 2/0.6$ use the wrong formula. The mean is $c/(1 - \\phi)$."
            },
            "ro": {
                "title": "Media unui AR(1)",
                "text": "Pentru $X_t = 2 + 0{,}6X_{t-1} + \\varepsilon_t$, cît este $E[X_t]$?",
                "options": [
                    "2",
                    "1,2",
                    "3,33",
                    "5"
                ],
                "correctExplanation": "Aplicînd media, cu $E[X_t] = E[X_{t-1}] = \\mu$: $\\mu = 2 + 0{,}6\\mu$, deci $\\mu = 2/0{,}4 = 5$.",
                "incorrectExplanation": "Termenul liber 2 nu este media; $1{,}2 = 2 \\times 0{,}6$ și $3{,}33 = 2/0{,}6$ folosesc o formulă greșită. Media este $c/(1 - \\phi)$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The variance of an AR(1)",
                "text": "For a stationary AR(1) $X_t = \\phi X_{t-1} + \\varepsilon_t$ with $\\mathrm{Var}(\\varepsilon_t) = \\sigma^2$, what is $\\gamma(0)$?",
                "options": [
                    "$\\sigma^2/(1 - \\phi)$",
                    "$\\sigma^2$",
                    "$\\sigma^2/(1 - \\phi^2)$",
                    "$\\phi^2\\sigma^2$"
                ],
                "correctExplanation": "$\\gamma(0) = \\phi^2\\gamma(0) + \\sigma^2$, because $\\varepsilon_t$ is uncorrelated with $X_{t-1}$; hence $\\gamma(0) = \\sigma^2/(1 - \\phi^2)$.",
                "incorrectExplanation": "$\\sigma^2$ ignores the propagation of past shocks, $\\phi^2\\sigma^2$ ignores the new shock, and $\\sigma^2/(1 - \\phi)$ confuses the variance with the mean formula."
            },
            "ro": {
                "title": "Varianța unui AR(1)",
                "text": "Pentru un AR(1) staționar $X_t = \\phi X_{t-1} + \\varepsilon_t$, cu $\\mathrm{Var}(\\varepsilon_t) = \\sigma^2$, cît este $\\gamma(0)$?",
                "options": [
                    "$\\sigma^2/(1 - \\phi)$",
                    "$\\sigma^2$",
                    "$\\sigma^2/(1 - \\phi^2)$",
                    "$\\phi^2\\sigma^2$"
                ],
                "correctExplanation": "$\\gamma(0) = \\phi^2\\gamma(0) + \\sigma^2$, deoarece $\\varepsilon_t$ este necorelat cu $X_{t-1}$; deci $\\gamma(0) = \\sigma^2/(1 - \\phi^2)$.",
                "incorrectExplanation": "$\\sigma^2$ ignoră propagarea șocurilor trecute, $\\phi^2\\sigma^2$ ignoră șocul nou, iar $\\sigma^2/(1 - \\phi)$ confundă varianța cu formula mediei."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The ACF of an AR(1)",
                "text": "For an AR(1) with $\\phi = 0.7$, what is $\\rho(3)$?",
                "options": [
                    "0.7",
                    "0.343",
                    "0.49",
                    "2.1"
                ],
                "correctExplanation": "For an AR(1), $\\rho(h) = \\phi^h$, so $\\rho(3) = 0.7^3 = 0.343$.",
                "incorrectExplanation": "0.7 is $\\rho(1)$ and 0.49 is $\\rho(2)$; 2.1 is $3\\phi$, which cannot be an autocorrelation since $|\\rho(h)| \\le 1$."
            },
            "ro": {
                "title": "ACF a unui AR(1)",
                "text": "Pentru un AR(1) cu $\\phi = 0{,}7$, cît este $\\rho(3)$?",
                "options": [
                    "0,7",
                    "0,343",
                    "0,49",
                    "2,1"
                ],
                "correctExplanation": "Pentru un AR(1), $\\rho(h) = \\phi^h$, deci $\\rho(3) = 0{,}7^3 = 0{,}343$.",
                "incorrectExplanation": "0,7 este $\\rho(1)$, iar 0,49 este $\\rho(2)$; 2,1 este $3\\phi$, care nu poate fi o autocorelație, deoarece $|\\rho(h)| \\le 1$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Roots of an AR(2)",
                "text": "Is $X_t = 1.2X_{t-1} - 0.32X_{t-2} + \\varepsilon_t$ stationary?",
                "options": [
                    "No: $\\phi_1 = 1.2 > 1$",
                    "No: the roots 0.8 and 0.4 lie inside the unit circle",
                    "Yes: $1 - 1.2z + 0.32z^2 = (1 - 0.8z)(1 - 0.4z)$ has roots 1.25 and 2.5, outside the unit circle",
                    "It cannot be decided without data"
                ],
                "correctExplanation": "Stationarity requires all roots of $\\phi(z)$ outside the unit circle; here they are 1.25 and 2.5 (the inverse roots 0.8 and 0.4 are inside, which is the same statement).",
                "incorrectExplanation": "A single coefficient above 1 does not decide stationarity: only the roots do. 0.8 and 0.4 are the inverse roots, which must lie inside the circle, and the model is known, so no data are needed."
            },
            "ro": {
                "title": "Rădăcinile unui AR(2)",
                "text": "Este staționar procesul $X_t = 1{,}2X_{t-1} - 0{,}32X_{t-2} + \\varepsilon_t$?",
                "options": [
                    "Nu: $\\phi_1 = 1{,}2 > 1$",
                    "Nu: rădăcinile 0,8 și 0,4 sînt în interiorul cercului unitate",
                    "Da: $1 - 1{,}2z + 0{,}32z^2 = (1 - 0{,}8z)(1 - 0{,}4z)$ are rădăcinile 1,25 și 2,5, în afara cercului unitate",
                    "Nu se poate decide fără date"
                ],
                "correctExplanation": "Staționaritatea cere ca toate rădăcinile lui $\\phi(z)$ să fie în afara cercului unitate; aici ele sînt 1,25 și 2,5 (rădăcinile inverse 0,8 și 0,4 sînt în interior, ceea ce înseamnă același lucru).",
                "incorrectExplanation": "Un singur coeficient mai mare decît 1 nu decide staționaritatea: o decid doar rădăcinile. 0,8 și 0,4 sînt rădăcinile inverse, care trebuie să fie în interiorul cercului, iar modelul este cunoscut, deci nu sînt necesare date."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Complex roots",
                "text": "The AR(2) $X_t = 1.0X_{t-1} - 0.6X_{t-2} + \\varepsilon_t$ has complex roots. What does its ACF look like?",
                "options": [
                    "A damped wave (pseudo-cycles)",
                    "It cuts off after lag 2",
                    "A geometric decay without sign changes",
                    "It does not decay (a unit root)"
                ],
                "correctExplanation": "Since $\\phi_1^2 + 4\\phi_2 = 1 - 2.4 < 0$, the roots are complex and the ACF is a damped cosine; the damping factor per lag is $\\sqrt{0.6} \\approx 0.77$.",
                "incorrectExplanation": "A cut-off after lag 2 is the ACF of an MA(2), not of an AR(2). A geometric decay without sign changes corresponds to real positive roots, and the process is stationary, so the ACF does decay."
            },
            "ro": {
                "title": "Rădăcini complexe",
                "text": "Procesul AR(2) $X_t = 1{,}0X_{t-1} - 0{,}6X_{t-2} + \\varepsilon_t$ are rădăcini complexe. Cum arată ACF?",
                "options": [
                    "O undă amortizată (pseudo-cicluri)",
                    "Se anulează după lagul 2",
                    "O descreștere geometrică fără schimbări de semn",
                    "Nu descrește (rădăcină unitară)"
                ],
                "correctExplanation": "Deoarece $\\phi_1^2 + 4\\phi_2 = 1 - 2{,}4 < 0$, rădăcinile sînt complexe, iar ACF este un cosinus amortizat; factorul de amortizare pe lag este $\\sqrt{0{,}6} \\approx 0{,}77$.",
                "incorrectExplanation": "Anularea după lagul 2 caracterizează ACF a unui MA(2), nu a unui AR(2). O descreștere geometrică fără schimbări de semn corespunde unor rădăcini reale pozitive, iar procesul este staționar, deci ACF descrește."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Identifying an AR model",
                "text": "The PACF has significant values at lags 1 and 2 and none after; the ACF decays gradually. Which model is suggested?",
                "options": [
                    "MA(2)",
                    "ARMA(1,1)",
                    "AR(2)",
                    "White noise"
                ],
                "correctExplanation": "A PACF that cuts off after lag $p$ together with a decaying ACF is the signature of an AR($p$), here AR(2).",
                "incorrectExplanation": "For an MA(2) the roles are reversed (the ACF cuts off); for an ARMA(1,1) both functions decay; white noise has no significant lags."
            },
            "ro": {
                "title": "Identificarea unui model AR",
                "text": "PACF are valori semnificative la lagurile 1 și 2 și niciuna după aceea; ACF descrește treptat. Ce model este sugerat?",
                "options": [
                    "MA(2)",
                    "ARMA(1,1)",
                    "AR(2)",
                    "Zgomot alb"
                ],
                "correctExplanation": "O PACF care se anulează după lagul $p$, împreună cu o ACF care descrește, este semnătura unui AR($p$), aici AR(2).",
                "incorrectExplanation": "Pentru un MA(2) rolurile sînt inversate (ACF se anulează); pentru un ARMA(1,1) ambele funcții descresc; zgomotul alb nu are laguri semnificative."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Identifying an MA model",
                "text": "The ACF has a single significant value at lag 1 and then cuts off, while the PACF decays gradually. Which model is suggested?",
                "options": [
                    "MA(1)",
                    "AR(1)",
                    "ARMA(1,1)",
                    "White noise"
                ],
                "correctExplanation": "An ACF that cuts off after lag 1 indicates an MA(1); the decaying PACF confirms it.",
                "incorrectExplanation": "For an AR(1) the pattern is reversed (the ACF decays, the PACF cuts off after lag 1); for an ARMA(1,1) both decay; white noise has no significant values."
            },
            "ro": {
                "title": "Identificarea unui model MA",
                "text": "ACF are o singură valoare semnificativă, la lagul 1, și apoi se anulează, iar PACF descrește treptat. Ce model este sugerat?",
                "options": [
                    "MA(1)",
                    "AR(1)",
                    "ARMA(1,1)",
                    "Zgomot alb"
                ],
                "correctExplanation": "O ACF care se anulează după lagul 1 indică un MA(1); PACF care descrește treptat confirmă acest lucru.",
                "incorrectExplanation": "Pentru un AR(1) tiparul este inversat (ACF descrește, PACF se anulează după lagul 1); pentru un ARMA(1,1) ambele descresc; zgomotul alb nu are nicio valoare semnificativă."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The largest MA(1) autocorrelation",
                "text": "A series has $\\hat\\rho(1) = 0.7$ and no other significant autocorrelation, with a large sample. Can it be an MA(1)?",
                "options": [
                    "Yes, with $\\theta = 0.7$",
                    "No: an MA(1) always has $|\\rho(1)| \\le 0.5$",
                    "Yes, with $\\theta = 1/0.7$",
                    "Yes, but only if $\\sigma^2 > 1$"
                ],
                "correctExplanation": "$\\rho(1) = \\theta/(1 + \\theta^2)$ reaches its maximum 0.5 at $\\theta = 1$, so no MA(1) produces 0.7; try an MA($q$) with more terms or another model.",
                "incorrectExplanation": "$\\theta = 0.7$ gives $\\rho(1) = 0.47$, and $\\theta = 1/0.7$ gives the same 0.47; the noise variance $\\sigma^2$ does not enter $\\rho(1)$ at all."
            },
            "ro": {
                "title": "Cea mai mare autocorelație a unui MA(1)",
                "text": "O serie are $\\hat\\rho(1) = 0{,}7$ și nicio altă autocorelație semnificativă, pe un eșantion mare. Poate fi un MA(1)?",
                "options": [
                    "Da, cu $\\theta = 0{,}7$",
                    "Nu: un MA(1) are întotdeauna $|\\rho(1)| \\le 0{,}5$",
                    "Da, cu $\\theta = 1/0{,}7$",
                    "Da, dar doar dacă $\\sigma^2 > 1$"
                ],
                "correctExplanation": "$\\rho(1) = \\theta/(1 + \\theta^2)$ își atinge maximul 0,5 în $\\theta = 1$, deci niciun MA(1) nu produce 0,7; încercați un MA($q$) cu mai mulți termeni sau alt model.",
                "incorrectExplanation": "$\\theta = 0{,}7$ dă $\\rho(1) = 0{,}47$, iar $\\theta = 1/0{,}7$ dă același 0,47; varianța zgomotului $\\sigma^2$ nu intră deloc în $\\rho(1)$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "MA(1) invertibility",
                "text": "Is the MA(1) process $X_t = \\varepsilon_t + 1.5\\varepsilon_{t-1}$ invertible?",
                "options": [
                    "No, because $|\\theta| = 1.5 > 1$",
                    "Yes, MA processes are always invertible",
                    "Yes, because $\\theta = 1.5 > 0$",
                    "No, MA processes are never invertible"
                ],
                "correctExplanation": "Invertibility requires $|\\theta| < 1$ (the root of $1 + \\theta z$ outside the unit circle). Here $|\\theta| = 1.5$, so the process is not invertible, although, like every finite MA, it is stationary.",
                "incorrectExplanation": "MA processes are always stationary, not always invertible; invertibility depends on $|\\theta|$, not on its sign, and with $|\\theta| < 1$ an MA(1) is invertible."
            },
            "ro": {
                "title": "Invertibilitatea MA(1)",
                "text": "Este invertibil procesul MA(1) $X_t = \\varepsilon_t + 1{,}5\\varepsilon_{t-1}$?",
                "options": [
                    "Nu, deoarece $|\\theta| = 1{,}5 > 1$",
                    "Da, procesele MA sînt întotdeauna invertibile",
                    "Da, deoarece $\\theta = 1{,}5 > 0$",
                    "Nu, procesele MA nu sînt niciodată invertibile"
                ],
                "correctExplanation": "Invertibilitatea cere $|\\theta| < 1$ (rădăcina lui $1 + \\theta z$ în afara cercului unitate). Aici $|\\theta| = 1{,}5$, deci procesul nu este invertibil, deși, ca orice MA finit, este staționar.",
                "incorrectExplanation": "Procesele MA sînt întotdeauna staționare, nu întotdeauna invertibile; invertibilitatea depinde de $|\\theta|$, nu de semnul lui, iar pentru $|\\theta| < 1$ un MA(1) este invertibil."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "$\\theta$ and $1/\\theta$",
                "text": "Compare $X_t = \\varepsilon_t + 2\\varepsilon_{t-1}$, $\\sigma^2 = 1$, with $Y_t = u_t + 0.5u_{t-1}$, $\\mathrm{Var}(u_t) = 4$. Which statement is correct?",
                "options": [
                    "They differ in variance: 5 against 4",
                    "Only $X_t$ is stationary",
                    "They have the same autocovariances and both are invertible",
                    "They have the same autocovariances; only $Y_t$ is invertible"
                ],
                "correctExplanation": "Both have $\\gamma(0) = 5$ and $\\gamma(1) = 2$; with Gaussian shocks they have the same distribution. Only $|0.5| < 1$ is invertible, so we report $Y_t$.",
                "incorrectExplanation": "$\\gamma(0)$ of $Y_t$ is $4 \\times 1.25 = 5$, not 4; every finite MA is stationary; $|\\theta| = 2$ is not invertible."
            },
            "ro": {
                "title": "$\\theta$ și $1/\\theta$",
                "text": "Comparați $X_t = \\varepsilon_t + 2\\varepsilon_{t-1}$, $\\sigma^2 = 1$, cu $Y_t = u_t + 0{,}5u_{t-1}$, $\\mathrm{Var}(u_t) = 4$. Ce afirmație este corectă?",
                "options": [
                    "Diferă prin varianță: 5 față de 4",
                    "Doar $X_t$ este staționar",
                    "Au aceleași autocovarianțe și ambele sînt invertibile",
                    "Au aceleași autocovarianțe; doar $Y_t$ este invertibil"
                ],
                "correctExplanation": "Ambele au $\\gamma(0) = 5$ și $\\gamma(1) = 2$; cu șocuri gaussiene au aceeași distribuție. Doar $|0{,}5| < 1$ dă un proces invertibil, deci raportăm $Y_t$.",
                "incorrectExplanation": "$\\gamma(0)$ pentru $Y_t$ este $4 \\times 1{,}25 = 5$, nu 4; orice MA finit este staționar; $|\\theta| = 2$ nu este invertibil."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Common factors",
                "text": "What is the process $X_t = 0.5X_{t-1} + \\varepsilon_t - 0.5\\varepsilon_{t-1}$?",
                "options": [
                    "An ARMA(1,1) with a slowly decaying ACF",
                    "A non-stationary process",
                    "White noise: $(1 - 0.5L)$ cancels on both sides",
                    "An MA(1) with $\\theta = -0.5$"
                ],
                "correctExplanation": "$(1 - 0.5L)X_t = (1 - 0.5L)\\varepsilon_t$, so $X_t = \\varepsilon_t$. Estimating an ARMA(1,1) on such data gives $\\hat\\phi \\approx -\\hat\\theta$ with huge standard errors.",
                "incorrectExplanation": "The AR and MA factors are identical and cancel, so there is no ARMA dynamics left; $|0.5| < 1$, so nothing is non-stationary; the MA part does not survive alone."
            },
            "ro": {
                "title": "Factori comuni",
                "text": "Ce proces este $X_t = 0{,}5X_{t-1} + \\varepsilon_t - 0{,}5\\varepsilon_{t-1}$?",
                "options": [
                    "Un ARMA(1,1) cu o ACF care descrește lent",
                    "Un proces nestaționar",
                    "Zgomot alb: $(1 - 0{,}5L)$ se simplifică în ambii membri",
                    "Un MA(1) cu $\\theta = -0{,}5$"
                ],
                "correctExplanation": "$(1 - 0{,}5L)X_t = (1 - 0{,}5L)\\varepsilon_t$, deci $X_t = \\varepsilon_t$. Estimarea unui ARMA(1,1) pe astfel de date dă $\\hat\\phi \\approx -\\hat\\theta$, cu erori standard foarte mari.",
                "incorrectExplanation": "Factorii AR și MA sînt identici și se simplifică, deci nu rămîne nicio dinamică ARMA; $|0{,}5| < 1$, deci nimic nu este nestaționar; partea MA nu rămîne singură."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "$\\psi$ weights",
                "text": "For the ARMA(1,1) $X_t = 0.5X_{t-1} + \\varepsilon_t + 0.3\\varepsilon_{t-1}$, what is $\\psi_1$, the effect of a unit shock after one period?",
                "options": [
                    "0.5",
                    "0.8",
                    "0.3",
                    "0.15"
                ],
                "correctExplanation": "From $\\psi_j = \\theta_j + \\phi\\psi_{j-1}$ with $\\psi_0 = 1$: $\\psi_1 = 0.3 + 0.5 = 0.8$; then $\\psi_j = 0.5^{j-1} \\times 0.8$.",
                "incorrectExplanation": "0.5 counts only the AR part, 0.3 only the MA part, and 0.15 is their product; the first response adds them."
            },
            "ro": {
                "title": "Ponderile $\\psi$",
                "text": "Pentru ARMA(1,1) $X_t = 0{,}5X_{t-1} + \\varepsilon_t + 0{,}3\\varepsilon_{t-1}$, cît este $\\psi_1$, efectul unui șoc unitar după o perioadă?",
                "options": [
                    "0,5",
                    "0,8",
                    "0,3",
                    "0,15"
                ],
                "correctExplanation": "Din $\\psi_j = \\theta_j + \\phi\\psi_{j-1}$, cu $\\psi_0 = 1$: $\\psi_1 = 0{,}3 + 0{,}5 = 0{,}8$; apoi $\\psi_j = 0{,}5^{j-1} \\times 0{,}8$.",
                "incorrectExplanation": "0,5 ține seama doar de partea AR, 0,3 doar de partea MA, iar 0,15 este produsul lor; primul răspuns le adună."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Yule–Walker for an AR(2)",
                "text": "A series has $\\hat\\rho(1) = 0.5$ and $\\hat\\rho(2) = 0.4$. What is the Yule–Walker estimate $\\hat\\phi_2$?",
                "options": [
                    "0.4",
                    "0.15",
                    "0.2",
                    "-0.2"
                ],
                "correctExplanation": "$\\hat\\phi_2 = (\\hat\\rho_2 - \\hat\\rho_1^2)/(1 - \\hat\\rho_1^2) = (0.4 - 0.25)/0.75 = 0.2$; it equals the sample PACF at lag 2.",
                "incorrectExplanation": "0.4 is $\\hat\\rho(2)$ itself, 0.15 is only the numerator, and the sign is positive because $\\hat\\rho_2 > \\hat\\rho_1^2$."
            },
            "ro": {
                "title": "Yule–Walker pentru un AR(2)",
                "text": "O serie are $\\hat\\rho(1) = 0{,}5$ și $\\hat\\rho(2) = 0{,}4$. Cît este estimarea Yule–Walker $\\hat\\phi_2$?",
                "options": [
                    "0,4",
                    "0,15",
                    "0,2",
                    "−0,2"
                ],
                "correctExplanation": "$\\hat\\phi_2 = (\\hat\\rho_2 - \\hat\\rho_1^2)/(1 - \\hat\\rho_1^2) = (0{,}4 - 0{,}25)/0{,}75 = 0{,}2$; este egală cu PACF de selecție la lagul 2.",
                "incorrectExplanation": "0,4 este chiar $\\hat\\rho(2)$, 0,15 este doar numărătorul, iar semnul este pozitiv deoarece $\\hat\\rho_2 > \\hat\\rho_1^2$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Annual growth of quarterly data",
                "text": "Quarterly GDP growth $g_t$ is close to white noise. Which model fits the annual growth $y_t = g_t + g_{t-1} + g_{t-2} + g_{t-3}$?",
                "options": [
                    "MA(3): consecutive annual rates share three quarters",
                    "AR(1): growth is persistent",
                    "White noise: a sum of white noises is white noise",
                    "AR(4): one lag per quarter"
                ],
                "correctExplanation": "$y_t$ and $y_{t-h}$ share $4 - h$ quarterly shocks for $h \\le 3$ and none for $h \\ge 4$, so the ACF cuts off after lag 3: an MA(3). For Romania, BIC chooses exactly this model.",
                "incorrectExplanation": "The persistence is created by the overlap, not by an autoregression; a moving sum of white noise is correlated, not white; the ACF cuts off, which rules out an AR."
            },
            "ro": {
                "title": "Creșterea anuală a datelor trimestriale",
                "text": "Creșterea trimestrială a PIB-ului, $g_t$, este aproape de un zgomot alb. Ce model se potrivește creșterii anuale $y_t = g_t + g_{t-1} + g_{t-2} + g_{t-3}$?",
                "options": [
                    "MA(3): ratele anuale consecutive au trei trimestre comune",
                    "AR(1): creșterea este persistentă",
                    "Zgomot alb: o sumă de zgomote albe este zgomot alb",
                    "AR(4): cîte un lag pentru fiecare trimestru"
                ],
                "correctExplanation": "$y_t$ și $y_{t-h}$ au $4 - h$ șocuri trimestriale comune pentru $h \\le 3$ și niciunul pentru $h \\ge 4$, deci ACF se anulează după lagul 3: un MA(3). Pentru România, BIC alege exact acest model.",
                "incorrectExplanation": "Persistența este creată de suprapunere, nu de o autoregresie; o sumă mobilă de zgomote albe este corelată, nu este zgomot alb; ACF se anulează, ceea ce exclude un AR."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "AIC and BIC",
                "text": "Which statement about AIC $= -2\\ln L + 2k$ and BIC $= -2\\ln L + k\\ln T$ is correct?",
                "options": [
                    "AIC always selects the true model in large samples",
                    "The model with the largest criterion is preferred",
                    "AIC and BIC can compare models fitted to different samples",
                    "BIC penalises parameters more once $T \\ge 8$ and is consistent; AIC may overfit even in large samples"
                ],
                "correctExplanation": "$\\ln T > 2$ for $T \\ge 8$, so BIC prefers smaller models; it finds the true order with probability tending to 1, while AIC keeps a positive probability of choosing too large a model.",
                "incorrectExplanation": "It is BIC, not AIC, that is consistent; the smallest value wins; and the criteria are comparable only on the same observations."
            },
            "ro": {
                "title": "AIC și BIC",
                "text": "Ce afirmație despre AIC $= -2\\ln L + 2k$ și BIC $= -2\\ln L + k\\ln T$ este corectă?",
                "options": [
                    "AIC alege întotdeauna modelul adevărat în eșantioane mari",
                    "Este preferat modelul cu valoarea cea mai mare a criteriului",
                    "AIC și BIC pot compara modele estimate pe eșantioane diferite",
                    "BIC penalizează mai mult parametrii de îndată ce $T \\ge 8$ și este consistent; AIC poate supraparametriza chiar în eșantioane mari"
                ],
                "correctExplanation": "$\\ln T > 2$ pentru $T \\ge 8$, deci BIC preferă modele mai mici; găsește ordinul adevărat cu o probabilitate care tinde la 1, pe cînd AIC păstrează o probabilitate pozitivă de a alege un model prea mare.",
                "incorrectExplanation": "BIC, nu AIC, este consistent; cîștigă valoarea cea mai mică; iar criteriile sînt comparabile doar pe aceleași observații."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Computing AIC",
                "text": "An AR(2) with a constant is estimated on $T = 100$ observations: $\\ln L = -210.9$ and $k = 4$ parameters. What is its AIC?",
                "options": [
                    "421.8",
                    "429.8",
                    "440.2",
                    "214.9"
                ],
                "correctExplanation": "AIC $= -2(-210.9) + 2 \\times 4 = 421.8 + 8 = 429.8$.",
                "incorrectExplanation": "421.8 omits the penalty, 440.2 is the BIC ($421.8 + 4\\ln 100$), and 214.9 forgets the factor 2 on the log-likelihood."
            },
            "ro": {
                "title": "Calculul AIC",
                "text": "Un AR(2) cu termen liber este estimat pe $T = 100$ de observații: $\\ln L = -210{,}9$ și $k = 4$ parametri. Cît este AIC?",
                "options": [
                    "421,8",
                    "429,8",
                    "440,2",
                    "214,9"
                ],
                "correctExplanation": "AIC $= -2(-210{,}9) + 2 \\times 4 = 421{,}8 + 8 = 429{,}8$.",
                "incorrectExplanation": "421,8 omite penalizarea, 440,2 este BIC ($421{,}8 + 4\\ln 100$), iar 214,9 uită factorul 2 al log-verosimilității."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Ljung–Box degrees of freedom",
                "text": "The Ljung–Box statistic $Q^*(10)$ is computed on the residuals of an ARMA(2,1). How many degrees of freedom does its $\\chi^2$ distribution have?",
                "options": [
                    "10",
                    "7",
                    "9",
                    "13"
                ],
                "correctExplanation": "Estimating $p + q = 3$ coefficients uses up 3 degrees of freedom: $m - p - q = 10 - 3 = 7$ (Box and Pierce, 1970).",
                "incorrectExplanation": "10 ignores the estimated coefficients and makes the test too lenient; 9 subtracts only one coefficient; 13 adds instead of subtracting."
            },
            "ro": {
                "title": "Gradele de libertate Ljung–Box",
                "text": "Statistica Ljung–Box $Q^*(10)$ se calculează pe reziduurile unui ARMA(2,1). Cîte grade de libertate are distribuția $\\chi^2$?",
                "options": [
                    "10",
                    "7",
                    "9",
                    "13"
                ],
                "correctExplanation": "Estimarea a $p + q = 3$ coeficienți consumă 3 grade de libertate: $m - p - q = 10 - 3 = 7$ (Box și Pierce, 1970).",
                "incorrectExplanation": "10 ignoră coeficienții estimați și face ca testul să respingă prea rar; 9 scade doar un coeficient; 13 adună în loc să scadă."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Reading a Ljung–Box test",
                "text": "After fitting an ARMA model, Ljung–Box on the residuals gives a p-value of 0.03. What does this mean at the 5% level?",
                "options": [
                    "The residuals are white noise",
                    "The test is inconclusive, so more data are needed",
                    "The residuals are Normal",
                    "Autocorrelation remains in the residuals: the model is inadequate"
                ],
                "correctExplanation": "$p < 0.05$ rejects $H_0$ of no autocorrelation up to lag $m$: the model has not captured all the dependence and must be re-specified.",
                "incorrectExplanation": "White-noise residuals would give a large p-value; a small p-value is a clear rejection, not an inconclusive result; normality is tested by Jarque–Bera, not by Ljung–Box."
            },
            "ro": {
                "title": "Interpretarea testului Ljung–Box",
                "text": "După estimarea unui model ARMA, testul Ljung–Box pentru reziduuri dă un p-value de 0,03. Ce înseamnă aceasta la pragul de 5%?",
                "options": [
                    "Reziduurile sînt zgomot alb",
                    "Testul este neconcludent, deci sînt necesare mai multe date",
                    "Reziduurile urmează distribuția Normală",
                    "În reziduuri rămîne autocorelație: modelul este inadecvat"
                ],
                "correctExplanation": "$p < 0{,}05$ respinge $H_0$ (absența autocorelației pînă la lagul $m$): modelul nu a surprins toată dependența și trebuie respecificat.",
                "incorrectExplanation": "Reziduurile de tip zgomot alb ar da un p-value mare; un p-value mic este o respingere clară, nu un rezultat neconcludent; normalitatea se testează cu Jarque–Bera, nu cu Ljung–Box."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Forecasting an AR(1)",
                "text": "$X_t = 2 + 0.8X_{t-1} + \\varepsilon_t$ (mean 10) and $X_T = 12$. What is the forecast $\\hat X_{T+2|T}$?",
                "options": [
                    "11.6",
                    "12",
                    "10",
                    "11.28"
                ],
                "correctExplanation": "$\\hat X_{T+h|T} = \\mu + \\phi^h(X_T - \\mu) = 10 + 0.64 \\times 2 = 11.28$: the forecast moves back towards the mean.",
                "incorrectExplanation": "11.6 is the one-step forecast, 12 is the no-change forecast, and 10 is the limit as $h \\to \\infty$."
            },
            "ro": {
                "title": "Prognoza unui AR(1)",
                "text": "$X_t = 2 + 0{,}8X_{t-1} + \\varepsilon_t$ (media 10) și $X_T = 12$. Cît este prognoza $\\hat X_{T+2|T}$?",
                "options": [
                    "11,6",
                    "12",
                    "10",
                    "11,28"
                ],
                "correctExplanation": "$\\hat X_{T+h|T} = \\mu + \\phi^h(X_T - \\mu) = 10 + 0{,}64 \\times 2 = 11{,}28$: prognoza revine spre medie.",
                "incorrectExplanation": "11,6 este prognoza cu un pas, 12 este prognoza fără schimbare, iar 10 este limita pentru $h \\to \\infty$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Forecasting an MA(2)",
                "text": "What is the forecast of an MA(2) process three steps ahead, $\\hat X_{T+3|T}$?",
                "options": [
                    "The last observation $X_T$",
                    "The mean $\\mu$",
                    "$\\mu + \\theta_2\\varepsilon_T$",
                    "It cannot be computed"
                ],
                "correctExplanation": "$X_{T+3} = \\mu + \\varepsilon_{T+3} + \\theta_1\\varepsilon_{T+2} + \\theta_2\\varepsilon_{T+1}$ contains only future shocks, all forecast by 0; beyond $q$ steps the forecast is $\\mu$.",
                "incorrectExplanation": "An MA(2) does not carry $X_T$ forward; $\\theta_2\\varepsilon_T$ enters the forecast for $T + 2$, not $T + 3$; and the forecast is perfectly well defined."
            },
            "ro": {
                "title": "Prognoza unui MA(2)",
                "text": "Cît este prognoza unui proces MA(2) cu trei pași înainte, $\\hat X_{T+3|T}$?",
                "options": [
                    "Ultima observație $X_T$",
                    "Media $\\mu$",
                    "$\\mu + \\theta_2\\varepsilon_T$",
                    "Nu se poate calcula"
                ],
                "correctExplanation": "$X_{T+3} = \\mu + \\varepsilon_{T+3} + \\theta_1\\varepsilon_{T+2} + \\theta_2\\varepsilon_{T+1}$ conține doar șocuri viitoare, toate prognozate cu 0; după $q$ pași prognoza este $\\mu$.",
                "incorrectExplanation": "Un MA(2) nu transmite mai departe valoarea $X_T$; $\\theta_2\\varepsilon_T$ intră în prognoza pentru $T + 2$, nu pentru $T + 3$; iar prognoza este bine definită."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Forecast error variance",
                "text": "For a stationary ARMA model, what happens to the forecast error variance as the horizon $h$ grows?",
                "options": [
                    "It increases towards the unconditional variance $\\gamma(0)$",
                    "It stays equal to $\\sigma^2$",
                    "It grows without bound",
                    "It decreases to 0"
                ],
                "correctExplanation": "$\\sigma_h^2 = \\sigma^2(1 + \\psi_1^2 + \\dots + \\psi_{h-1}^2)$ increases with $h$ and converges to $\\sigma^2\\sum_j\\psi_j^2 = \\gamma(0)$.",
                "incorrectExplanation": "$\\sigma^2$ is the one-step value only; unbounded growth is the case of a unit root (Chapter 3); uncertainty never falls with the horizon."
            },
            "ro": {
                "title": "Varianța erorii de prognoză",
                "text": "Pentru un model ARMA staționar, ce se întîmplă cu varianța erorii de prognoză cînd orizontul $h$ crește?",
                "options": [
                    "Crește spre varianța necondiționată $\\gamma(0)$",
                    "Rămîne egală cu $\\sigma^2$",
                    "Crește nelimitat",
                    "Scade la 0"
                ],
                "correctExplanation": "$\\sigma_h^2 = \\sigma^2(1 + \\psi_1^2 + \\dots + \\psi_{h-1}^2)$ crește cu $h$ și converge la $\\sigma^2\\sum_j\\psi_j^2 = \\gamma(0)$.",
                "incorrectExplanation": "$\\sigma^2$ este doar valoarea pentru un pas; creșterea nelimitată este cazul unei rădăcini unitare (Capitolul 3); incertitudinea nu scade niciodată cu orizontul."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The Box–Jenkins method",
                "text": "What is the order of the steps in the Box–Jenkins method?",
                "options": [
                    "Estimation, identification, forecasting, diagnostic checking",
                    "Forecasting, estimation, identification",
                    "Identification, estimation, diagnostic checking, forecasting (and back to identification if the check fails)",
                    "Diagnostic checking, identification, estimation"
                ],
                "correctExplanation": "Box and Jenkins proposed a loop: identify candidates from the ACF and PACF of a stationary series, estimate them, check the residuals, and forecast only with a model that passes.",
                "incorrectExplanation": "A model cannot be estimated before its orders are chosen, and forecasting before checking would use a model that may leave structure in its residuals."
            },
            "ro": {
                "title": "Metoda Box–Jenkins",
                "text": "Care este ordinea pașilor în metoda Box–Jenkins?",
                "options": [
                    "Estimare, identificare, prognoză, verificarea diagnosticelor",
                    "Prognoză, estimare, identificare",
                    "Identificare, estimare, verificarea diagnosticelor, prognoză (și revenire la identificare dacă verificarea eșuează)",
                    "Verificarea diagnosticelor, identificare, estimare"
                ],
                "correctExplanation": "Box și Jenkins au propus o buclă: identificăm candidații din ACF și PACF ale unei serii staționare, îi estimăm, verificăm reziduurile și prognozăm doar cu un model acceptat.",
                "incorrectExplanation": "Un model nu poate fi estimat înainte de alegerea ordinelor, iar prognoza înainte de verificare ar folosi un model care poate lăsa structură în reziduuri."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Significant or useful?",
                "text": "An AR(1) for BET daily returns gives $\\hat\\phi = 0.11$ with $t = 21$. What follows?",
                "options": [
                    "Yesterday's return explains most of today's return",
                    "The coefficient is not significant",
                    "Returns are a random walk",
                    "The effect is statistically real but explains only about 1% of the variance"
                ],
                "correctExplanation": "$t = 21$ makes $\\phi \\ne 0$ certain, but $R^2 \\approx \\hat\\phi^2 \\approx 1.2\\%$: too small to trade on after costs. Significance is not size.",
                "incorrectExplanation": "A high $t$-ratio reflects the many observations, not a large effect; the coefficient is clearly significant; and the random walk describes prices, not returns."
            },
            "ro": {
                "title": "Semnificativ sau util?",
                "text": "Un AR(1) pentru randamentele zilnice ale BET dă $\\hat\\phi = 0{,}11$, cu $t = 21$. Ce rezultă?",
                "options": [
                    "Randamentul de ieri explică cea mai mare parte a randamentului de azi",
                    "Coeficientul nu este semnificativ",
                    "Randamentele sînt un mers aleator",
                    "Efectul este real din punct de vedere statistic, dar explică doar aproximativ 1% din varianță"
                ],
                "correctExplanation": "$t = 21$ face ca $\\phi \\ne 0$ să fie sigur, dar $R^2 \\approx \\hat\\phi^2 \\approx 1{,}2\\%$: prea puțin pentru tranzacționare după costuri. Semnificația nu este mărime.",
                "incorrectExplanation": "Un raport $t$ mare reflectă numărul mare de observații, nu un efect mare; coeficientul este clar semnificativ; iar mersul aleator descrie prețurile, nu randamentele."
            }
        }
    ]
};
