// ============================================================
// Chapter 14 quiz bank: Multivariate GARCH models (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['mgarch'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "Portfolio volatility",
                "text": "Two assets with volatility 20% each, weights 50% and 50%, correlation $\\rho = 0.8$. What is the portfolio volatility?",
                "options": [
                    "About 19.0%",
                    "20.0%",
                    "About 15.5%",
                    "16.0%"
                ],
                "correctExplanation": "$\\sigma_p^2 = 0.25\\cdot400 + 0.25\\cdot400 + 2\\cdot0.25\\cdot0.8\\cdot400 = 360$, so $\\sigma_p = \\sqrt{360} \\approx 19.0\\%$.",
                "incorrectExplanation": "20% would need $\\rho = 1$, 15.5% corresponds to $\\rho = 0.2$, and 16% multiplies 20% by 0.8 instead of computing the variance. The variance is 360, so $\\sigma_p \\approx 19.0\\%$."
            },
            "ro": {
                "title": "Volatilitatea portofoliului",
                "text": "Două active cu volatilitatea de 20% fiecare, ponderi 50% și 50%, corelația $\\rho = 0{,}8$. Cît este volatilitatea portofoliului?",
                "options": [
                    "Aproximativ 19,0%",
                    "20,0%",
                    "Aproximativ 15,5%",
                    "16,0%"
                ],
                "correctExplanation": "$\\sigma_p^2 = 0{,}25\\cdot400 + 0{,}25\\cdot400 + 2\\cdot0{,}25\\cdot0{,}8\\cdot400 = 360$, deci $\\sigma_p = \\sqrt{360} \\approx 19{,}0\\%$.",
                "incorrectExplanation": "20% ar cere $\\rho = 1$, 15,5% corespunde lui $\\rho = 0{,}2$, iar 16% înmulțește 20% cu 0,8 în loc să calculeze varianța. Varianța este 360, deci $\\sigma_p \\approx 19{,}0\\%$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Positive definiteness",
                "text": "A $2 \\times 2$ conditional covariance matrix has $h_{11} = 4$ and $h_{22} = 1$. For which value of $h_{12}$ is it NOT positive definite?",
                "options": [
                    "$h_{12} = 1.5$",
                    "$h_{12} = -1$",
                    "$h_{12} = 0$",
                    "$h_{12} = 2.5$"
                ],
                "correctExplanation": "Positive definiteness needs $h_{11}h_{22} - h_{12}^2 > 0$, i.e. $|h_{12}| < 2$. With $h_{12} = 2.5$ the implied correlation would be $2.5/2 = 1.25$, which is impossible.",
                "incorrectExplanation": "The values 1.5, $-1$ and 0 all satisfy $h_{12}^2 < 4$, i.e. a correlation between $-1$ and 1. Only $h_{12} = 2.5$ violates the condition."
            },
            "ro": {
                "title": "Caracterul pozitiv definit",
                "text": "O matrice de covarianță condiționată $2 \\times 2$ are $h_{11} = 4$ și $h_{22} = 1$. Pentru ce valoare a lui $h_{12}$ NU este pozitiv definită?",
                "options": [
                    "$h_{12} = 1{,}5$",
                    "$h_{12} = -1$",
                    "$h_{12} = 0$",
                    "$h_{12} = 2{,}5$"
                ],
                "correctExplanation": "Caracterul pozitiv definit cere $h_{11}h_{22} - h_{12}^2 > 0$, adică $|h_{12}| < 2$. Cu $h_{12} = 2{,}5$ corelația implicită ar fi $2{,}5/2 = 1{,}25$, ceea ce este imposibil.",
                "incorrectExplanation": "Valorile 1,5, $-1$ și 0 respectă toate $h_{12}^2 < 4$, adică o corelație între $-1$ și 1. Doar $h_{12} = 2{,}5$ încalcă condiția."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The vech operator",
                "text": "How many distinct elements does the conditional covariance matrix of $N = 10$ assets have?",
                "options": [
                    "100",
                    "55",
                    "45",
                    "10"
                ],
                "correctExplanation": "The matrix is symmetric, so only the lower triangle counts: $N(N+1)/2 = 10\\cdot11/2 = 55$ elements (10 variances and 45 covariances). This is the length of vech($\\mathbf{H}_t$).",
                "incorrectExplanation": "100 counts every element twice off the diagonal, 45 counts only the covariances and 10 only the variances. The symmetric matrix has $N(N+1)/2 = 55$ distinct elements."
            },
            "ro": {
                "title": "Operatorul vech",
                "text": "Cîte elemente distincte are matricea de covarianță condiționată a $N = 10$ active?",
                "options": [
                    "100",
                    "55",
                    "45",
                    "10"
                ],
                "correctExplanation": "Matricea este simetrică, deci contează doar triunghiul inferior: $N(N+1)/2 = 10\\cdot11/2 = 55$ de elemente (10 varianțe și 45 de covarianțe). Aceasta este lungimea lui vech($\\mathbf{H}_t$).",
                "incorrectExplanation": "100 numără de două ori elementele din afara diagonalei, 45 numără doar covarianțele, iar 10 doar varianțele. Matricea simetrică are $N(N+1)/2 = 55$ de elemente distincte."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Parameters of the VEC model",
                "text": "How many parameters does the variance equation of a full VEC(1,1) model have for $N = 2$ assets?",
                "options": [
                    "21",
                    "11",
                    "9",
                    "7"
                ],
                "correctExplanation": "With $k = N(N+1)/2 = 3$: $\\mathbf{c}$ has 3 elements and $\\mathbf{A}$, $\\mathbf{B}$ are $3 \\times 3$, so $3 + 2\\cdot9 = 21$ parameters.",
                "incorrectExplanation": "11 is the full BEKK(1,1), 9 the diagonal VEC and 7 the CCC model for two assets. The full VEC has $k + 2k^2 = 21$ parameters."
            },
            "ro": {
                "title": "Parametrii modelului VEC",
                "text": "Cîți parametri are ecuația de varianță a unui model VEC(1,1) complet pentru $N = 2$ active?",
                "options": [
                    "21",
                    "11",
                    "9",
                    "7"
                ],
                "correctExplanation": "Cu $k = N(N+1)/2 = 3$: $\\mathbf{c}$ are 3 elemente, iar $\\mathbf{A}$, $\\mathbf{B}$ sînt $3 \\times 3$, deci $3 + 2\\cdot9 = 21$ de parametri.",
                "incorrectExplanation": "11 este BEKK(1,1) complet, 9 este VEC diagonal, iar 7 este modelul CCC pentru două active. VEC complet are $k + 2k^2 = 21$ de parametri."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The advantage of BEKK",
                "text": "What is the main advantage of the BEKK model over the VEC model?",
                "options": [
                    "It has no parameters to estimate",
                    "$\\mathbf{H}_t$ is positive definite by construction",
                    "Its correlations are constant",
                    "It does not need the past shocks"
                ],
                "correctExplanation": "Each term of $\\mathbf{H}_t = \\mathbf{C}\\mathbf{C}^\\top + \\mathbf{A}^\\top\\boldsymbol{\\varepsilon}_{t-1}\\boldsymbol{\\varepsilon}_{t-1}^\\top\\mathbf{A} + \\mathbf{B}^\\top\\mathbf{H}_{t-1}\\mathbf{B}$ is a quadratic form, so $\\mathbf{H}_t$ is positive definite without extra restrictions.",
                "incorrectExplanation": "BEKK still has many parameters, its correlations move over time and it uses the past shocks through $\\mathbf{A}$. Its advantage is positive definiteness by construction."
            },
            "ro": {
                "title": "Avantajul modelului BEKK",
                "text": "Care este principalul avantaj al modelului BEKK față de modelul VEC?",
                "options": [
                    "Nu are parametri de estimat",
                    "$\\mathbf{H}_t$ este pozitiv definită prin construcție",
                    "Corelațiile lui sînt constante",
                    "Nu are nevoie de șocurile trecute"
                ],
                "correctExplanation": "Fiecare termen din $\\mathbf{H}_t = \\mathbf{C}\\mathbf{C}^\\top + \\mathbf{A}^\\top\\boldsymbol{\\varepsilon}_{t-1}\\boldsymbol{\\varepsilon}_{t-1}^\\top\\mathbf{A} + \\mathbf{B}^\\top\\mathbf{H}_{t-1}\\mathbf{B}$ este o formă pătratică, deci $\\mathbf{H}_t$ este pozitiv definită fără restricții suplimentare.",
                "incorrectExplanation": "BEKK are tot mulți parametri, corelațiile lui se mișcă în timp și folosește șocurile trecute prin $\\mathbf{A}$. Avantajul lui este caracterul pozitiv definit prin construcție."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Parameters of BEKK",
                "text": "How many parameters does a full BEKK(1,1) variance equation have for $N = 2$ assets?",
                "options": [
                    "21",
                    "7",
                    "5",
                    "11"
                ],
                "correctExplanation": "$\\mathbf{C}$ lower triangular has $N(N+1)/2 = 3$ elements and $\\mathbf{A}$, $\\mathbf{B}$ have $N^2 = 4$ each: $3 + 8 = 11$.",
                "incorrectExplanation": "21 is the full VEC, 7 the diagonal BEKK and 5 the scalar BEKK. The full BEKK has $N(N+1)/2 + 2N^2 = 11$ parameters."
            },
            "ro": {
                "title": "Parametrii modelului BEKK",
                "text": "Cîți parametri are ecuația de varianță a unui BEKK(1,1) complet pentru $N = 2$ active?",
                "options": [
                    "21",
                    "7",
                    "5",
                    "11"
                ],
                "correctExplanation": "$\\mathbf{C}$, inferior triunghiulară, are $N(N+1)/2 = 3$ elemente, iar $\\mathbf{A}$ și $\\mathbf{B}$ au cîte $N^2 = 4$: $3 + 8 = 11$.",
                "incorrectExplanation": "21 este VEC complet, 7 este BEKK diagonal, iar 5 este BEKK scalar. BEKK complet are $N(N+1)/2 + 2N^2 = 11$ parametri."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Diagonal BEKK",
                "text": "What does a diagonal BEKK model lose compared with a full BEKK?",
                "options": [
                    "Positive definiteness of $\\mathbf{H}_t$",
                    "Time-varying covariances",
                    "The GARCH dynamics of each variance",
                    "Volatility spillovers between assets"
                ],
                "correctExplanation": "Spillovers are measured by the off-diagonal elements $a_{ij}$, $b_{ij}$; a diagonal BEKK sets them to zero, so a shock to asset 1 does not enter the variance of asset 2.",
                "incorrectExplanation": "The diagonal BEKK stays positive definite, keeps time-varying covariances ($h_{12,t}$ still moves) and keeps a GARCH-type dynamics for each variance. Only the spillovers disappear."
            },
            "ro": {
                "title": "BEKK diagonal",
                "text": "Ce pierde un model BEKK diagonal față de un BEKK complet?",
                "options": [
                    "Caracterul pozitiv definit al lui $\\mathbf{H}_t$",
                    "Covarianțele variabile în timp",
                    "Dinamica GARCH a fiecărei varianțe",
                    "Transmiterea volatilității între active"
                ],
                "correctExplanation": "Transmiterea este măsurată de elementele din afara diagonalei $a_{ij}$, $b_{ij}$; BEKK diagonal le fixează la zero, deci un șoc al activului 1 nu intră în varianța activului 2.",
                "incorrectExplanation": "BEKK diagonal rămîne pozitiv definit, păstrează covarianțele variabile ($h_{12,t}$ se mișcă în continuare) și dinamica de tip GARCH a fiecărei varianțe. Dispare doar transmiterea volatilității."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The CCC model",
                "text": "In the CCC model of Bollerslev (1990), which part of $\\mathbf{H}_t = \\mathbf{D}_t\\mathbf{R}_t\\mathbf{D}_t$ is constant?",
                "options": [
                    "The volatilities in $\\mathbf{D}_t$",
                    "The correlation matrix $\\mathbf{R}$",
                    "The whole matrix $\\mathbf{H}_t$",
                    "Nothing: everything changes over time"
                ],
                "correctExplanation": "CCC keeps the conditional correlations constant, $\\mathbf{R}_t = \\mathbf{R}$, while each volatility follows its own GARCH model; covariances move only through the volatilities.",
                "incorrectExplanation": "In CCC the volatilities change (univariate GARCH), so $\\mathbf{H}_t$ changes too; the constant part is the correlation matrix."
            },
            "ro": {
                "title": "Modelul CCC",
                "text": "În modelul CCC al lui Bollerslev (1990), care parte din $\\mathbf{H}_t = \\mathbf{D}_t\\mathbf{R}_t\\mathbf{D}_t$ este constantă?",
                "options": [
                    "Volatilitățile din $\\mathbf{D}_t$",
                    "Matricea corelațiilor $\\mathbf{R}$",
                    "Întreaga matrice $\\mathbf{H}_t$",
                    "Nimic: totul se schimbă în timp"
                ],
                "correctExplanation": "CCC ține constante corelațiile condiționate, $\\mathbf{R}_t = \\mathbf{R}$, iar fiecare volatilitate urmează propriul model GARCH; covarianțele se mișcă doar prin volatilități.",
                "incorrectExplanation": "În CCC volatilitățile se schimbă (GARCH univariat), deci se schimbă și $\\mathbf{H}_t$; partea constantă este matricea corelațiilor."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The DCC recursion",
                "text": "In the DCC model, which shocks enter $\\mathbf{Q}_t$?",
                "options": [
                    "The raw returns $\\mathbf{r}_t$ of the same day",
                    "The standardised residuals $\\mathbf{z}_{t-1}$ of the previous day",
                    "The standardised residuals $\\mathbf{z}_t$ of the same day",
                    "Only the volatilities $\\sigma_{i,t}$"
                ],
                "correctExplanation": "$\\mathbf{Q}_t = (1-a-b)\\bar{\\mathbf{Q}} + a\\,\\mathbf{z}_{t-1}\\mathbf{z}_{t-1}^\\top + b\\,\\mathbf{Q}_{t-1}$ uses yesterday's standardised residuals, so $\\mathbf{R}_t$ is a forecast made with information up to $t-1$.",
                "incorrectExplanation": "Using day-$t$ values would use information that is not yet known when the forecast is made, and raw returns mix volatility with correlation. DCC uses $\\mathbf{z}_{t-1}$."
            },
            "ro": {
                "title": "Recurența DCC",
                "text": "În modelul DCC, ce șocuri intră în $\\mathbf{Q}_t$?",
                "options": [
                    "Randamentele brute $\\mathbf{r}_t$ din aceeași zi",
                    "Reziduurile standardizate $\\mathbf{z}_{t-1}$ din ziua precedentă",
                    "Reziduurile standardizate $\\mathbf{z}_t$ din aceeași zi",
                    "Doar volatilitățile $\\sigma_{i,t}$"
                ],
                "correctExplanation": "$\\mathbf{Q}_t = (1-a-b)\\bar{\\mathbf{Q}} + a\\,\\mathbf{z}_{t-1}\\mathbf{z}_{t-1}^\\top + b\\,\\mathbf{Q}_{t-1}$ folosește reziduurile standardizate de ieri, deci $\\mathbf{R}_t$ este o prognoză făcută cu informația pînă la $t-1$.",
                "incorrectExplanation": "Valorile din ziua $t$ ar folosi informații încă necunoscute în momentul prognozei, iar randamentele brute amestecă volatilitatea cu corelația. DCC folosește $\\mathbf{z}_{t-1}$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Rescaling in DCC",
                "text": "Why is $\\mathbf{Q}_t$ rescaled to $\\rho_{ij,t} = q_{ij,t}/\\sqrt{q_{ii,t}q_{jj,t}}$?",
                "options": [
                    "To make the model faster to estimate",
                    "To remove the GARCH volatilities",
                    "So that $\\mathbf{R}_t$ has ones on the diagonal and is a correlation matrix",
                    "To make the correlations constant"
                ],
                "correctExplanation": "The recursion does not keep $q_{ii,t} = 1$; dividing by $\\sqrt{q_{ii,t}q_{jj,t}}$ turns $\\mathbf{Q}_t$ into a proper correlation matrix with unit diagonal and $|\\rho| < 1$.",
                "incorrectExplanation": "The volatilities are already removed in step 1, the rescaling does not make correlations constant and has nothing to do with speed. Its role is to produce a valid correlation matrix."
            },
            "ro": {
                "title": "Rescalarea în DCC",
                "text": "De ce se rescalează $\\mathbf{Q}_t$ la $\\rho_{ij,t} = q_{ij,t}/\\sqrt{q_{ii,t}q_{jj,t}}$?",
                "options": [
                    "Pentru ca modelul să se estimeze mai repede",
                    "Pentru a elimina volatilitățile GARCH",
                    "Pentru ca $\\mathbf{R}_t$ să aibă 1 pe diagonală și să fie o matrice de corelații",
                    "Pentru ca corelațiile să fie constante"
                ],
                "correctExplanation": "Recurența nu păstrează $q_{ii,t} = 1$; împărțirea la $\\sqrt{q_{ii,t}q_{jj,t}}$ transformă $\\mathbf{Q}_t$ într-o matrice de corelații corectă, cu 1 pe diagonală și $|\\rho| < 1$.",
                "incorrectExplanation": "Volatilitățile sînt deja eliminate în pasul 1, rescalarea nu face corelațiile constante și nu are legătură cu viteza. Rolul ei este să producă o matrice de corelații validă."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Two-step estimation",
                "text": "What happens in step 1 of the two-step DCC estimation?",
                "options": [
                    "The parameters $a$ and $b$ are estimated",
                    "A VAR model is fitted to all series jointly",
                    "A univariate GARCH model is fitted to each series separately",
                    "The sample correlation of the returns is computed"
                ],
                "correctExplanation": "Step 1 fits a GARCH (e.g. GARCH(1,1)) to each series and keeps the standardised residuals $\\hat z_{i,t} = \\hat\\varepsilon_{i,t}/\\hat\\sigma_{i,t}$; step 2 uses them to estimate $(a, b)$.",
                "incorrectExplanation": "$a$ and $b$ are estimated in step 2, DCC does not require a VAR, and the correlation target is computed from the residuals of step 1, not from the raw returns."
            },
            "ro": {
                "title": "Estimarea în doi pași",
                "text": "Ce se face în pasul 1 al estimării DCC în doi pași?",
                "options": [
                    "Se estimează parametrii $a$ și $b$",
                    "Se estimează un model VAR pentru toate seriile împreună",
                    "Se estimează cîte un model GARCH univariat pentru fiecare serie",
                    "Se calculează corelația de selecție a randamentelor"
                ],
                "correctExplanation": "Pasul 1 estimează un GARCH (de exemplu GARCH(1,1)) pentru fiecare serie și păstrează reziduurile standardizate $\\hat z_{i,t} = \\hat\\varepsilon_{i,t}/\\hat\\sigma_{i,t}$; pasul 2 le folosește pentru a estima $(a, b)$.",
                "incorrectExplanation": "$a$ și $b$ se estimează în pasul 2, DCC nu cere un VAR, iar ținta corelației se calculează din reziduurile pasului 1, nu din randamentele brute."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Correlation targeting",
                "text": "In DCC with correlation targeting, how is $\\bar{\\mathbf{Q}}$ obtained?",
                "options": [
                    "By numerical maximisation together with $a$ and $b$",
                    "As the sample correlation matrix of the standardised residuals $\\hat{\\mathbf{z}}_t$",
                    "It is set to the identity matrix",
                    "As the average of the GARCH volatilities"
                ],
                "correctExplanation": "$\\bar{\\mathbf{Q}}$ is fixed at the sample correlation of $\\hat{\\mathbf{z}}_t$, so the optimiser only searches over $(a, b)$, whatever the number of assets.",
                "incorrectExplanation": "Estimating $\\bar{\\mathbf{Q}}$ numerically would bring back the curse of dimensionality, the identity would assume zero correlations, and volatilities are not correlations."
            },
            "ro": {
                "title": "Țintirea corelației",
                "text": "În DCC cu țintirea corelației, cum se obține $\\bar{\\mathbf{Q}}$?",
                "options": [
                    "Prin maximizare numerică, împreună cu $a$ și $b$",
                    "Ca matricea corelațiilor de selecție a reziduurilor standardizate $\\hat{\\mathbf{z}}_t$",
                    "Este fixată la matricea unitate",
                    "Ca media volatilităților GARCH"
                ],
                "correctExplanation": "$\\bar{\\mathbf{Q}}$ este fixată la corelația de selecție a lui $\\hat{\\mathbf{z}}_t$, deci optimizarea caută doar după $(a, b)$, oricare ar fi numărul de active.",
                "incorrectExplanation": "Estimarea numerică a lui $\\bar{\\mathbf{Q}}$ ar readuce problema dimensionalității, matricea unitate ar presupune corelații nule, iar volatilitățile nu sînt corelații."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Half-life of correlation shocks",
                "text": "A DCC model has $a + b = 0.99$ on daily data. What is the half-life of a correlation shock?",
                "options": [
                    "About 69 days",
                    "About 34 days",
                    "About 99 days",
                    "About 7 days"
                ],
                "correctExplanation": "Deviations shrink by the factor $a + b$ per day: half-life $= \\ln 0.5/\\ln 0.99 \\approx 69$ days.",
                "incorrectExplanation": "34 days corresponds to $a + b = 0.98$; 99 and 7 do not follow from the formula $\\ln 0.5/\\ln(a+b)$, which gives about 69 days here."
            },
            "ro": {
                "title": "Timpul de înjumătățire al șocurilor de corelație",
                "text": "Un model DCC are $a + b = 0{,}99$ pe date zilnice. Cît este timpul de înjumătățire al unui șoc de corelație?",
                "options": [
                    "Aproximativ 69 de zile",
                    "Aproximativ 34 de zile",
                    "Aproximativ 99 de zile",
                    "Aproximativ 7 zile"
                ],
                "correctExplanation": "Abaterile se reduc zilnic cu factorul $a + b$: timpul de înjumătățire $= \\ln 0{,}5/\\ln 0{,}99 \\approx 69$ de zile.",
                "incorrectExplanation": "34 de zile corespund lui $a + b = 0{,}98$; 99 și 7 nu rezultă din formula $\\ln 0{,}5/\\ln(a+b)$, care dă aici aproximativ 69 de zile."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "One DCC step",
                "text": "DCC with $a = 0.05$, $b = 0.93$, $\\bar q_{11} = \\bar q_{22} = 1$, $\\bar q_{12} = 0.5$, $\\mathbf{Q}_{t-1} = \\bar{\\mathbf{Q}}$ and $z_{t-1} = (1, 1)$. What is $\\rho_{12,t}$?",
                "options": [
                    "0.500",
                    "0.550",
                    "0.525",
                    "0.465"
                ],
                "correctExplanation": "$q_{12,t} = 0.02\\cdot0.5 + 0.05\\cdot1 + 0.93\\cdot0.5 = 0.525$ and $q_{11,t} = q_{22,t} = 0.02 + 0.05 + 0.93 = 1$, so $\\rho_{12,t} = 0.525$.",
                "incorrectExplanation": "0.500 ignores the shock, 0.550 adds $a$ twice and 0.465 keeps only the $b$ term. A joint move of $+1$ raises the correlation slightly, to 0.525."
            },
            "ro": {
                "title": "Un pas DCC",
                "text": "DCC cu $a = 0{,}05$, $b = 0{,}93$, $\\bar q_{11} = \\bar q_{22} = 1$, $\\bar q_{12} = 0{,}5$, $\\mathbf{Q}_{t-1} = \\bar{\\mathbf{Q}}$ și $z_{t-1} = (1, 1)$. Cît este $\\rho_{12,t}$?",
                "options": [
                    "0,500",
                    "0,550",
                    "0,525",
                    "0,465"
                ],
                "correctExplanation": "$q_{12,t} = 0{,}02\\cdot0{,}5 + 0{,}05\\cdot1 + 0{,}93\\cdot0{,}5 = 0{,}525$ și $q_{11,t} = q_{22,t} = 0{,}02 + 0{,}05 + 0{,}93 = 1$, deci $\\rho_{12,t} = 0{,}525$.",
                "incorrectExplanation": "0,500 ignoră șocul, 0,550 adună de două ori $a$, iar 0,465 păstrează doar termenul cu $b$. O mișcare comună de $+1$ crește ușor corelația, la 0,525."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Testing DCC against CCC",
                "text": "How can constant correlation (CCC) be tested against DCC?",
                "options": [
                    "A Dickey-Fuller test on the correlations",
                    "A Ljung-Box test on the returns",
                    "Likelihood ratio $2(\\ell^{DCC} - \\ell^{CCC})$ compared with $\\chi^2(2)$",
                    "Comparing the means of the two series"
                ],
                "correctExplanation": "CCC is DCC with $a = b = 0$: two restrictions, so the LR statistic is compared with $\\chi^2(2)$ (5% critical value 5.99, conservative because the null is on the boundary).",
                "incorrectExplanation": "Unit-root tests, autocorrelation tests of returns and comparisons of means do not test the restriction $a = b = 0$. The likelihood ratio does."
            },
            "ro": {
                "title": "Testarea DCC față de CCC",
                "text": "Cum se poate testa corelația constantă (CCC) față de DCC?",
                "options": [
                    "Printr-un test Dickey-Fuller pe corelații",
                    "Printr-un test Ljung-Box pe randamente",
                    "Prin raportul de verosimilitate $2(\\ell^{DCC} - \\ell^{CCC})$ comparat cu $\\chi^2(2)$",
                    "Prin compararea mediilor celor două serii"
                ],
                "correctExplanation": "CCC este DCC cu $a = b = 0$: două restricții, deci statistica LR se compară cu $\\chi^2(2)$ (valoarea critică la 5%: 5,99, conservatoare, deoarece ipoteza nulă este pe frontieră).",
                "incorrectExplanation": "Testele de rădăcină unitară, testele de autocorelare a randamentelor și compararea mediilor nu testează restricția $a = b = 0$. Raportul de verosimilitate o testează."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Asynchronous trading",
                "text": "Daily returns of the S&P 500 (close 22:00 CET) and the BET (close 16:45 CET) are correlated. What does asynchronous trading do to their daily correlation?",
                "options": [
                    "It biases the correlation towards one",
                    "It has no effect on the correlation",
                    "It biases the correlation towards zero",
                    "It makes the correlation negative by construction"
                ],
                "correctExplanation": "News that arrives after the Bucharest close enters the BET only the next day, so the same news falls on different days: the same-day correlation is understated; weekly returns reduce the problem.",
                "incorrectExplanation": "The mismatch splits common news across two days, which weakens the same-day co-movement; it neither inflates it nor makes it negative by construction."
            },
            "ro": {
                "title": "Tranzacționarea asincronă",
                "text": "Randamentele zilnice S&P 500 (închidere la 22:00 CET) și BET (închidere la 16:45 CET) sînt corelate. Ce efect are tranzacționarea asincronă asupra corelației zilnice?",
                "options": [
                    "Deplasează corelația spre unu",
                    "Nu are niciun efect asupra corelației",
                    "Deplasează corelația spre zero",
                    "Face corelația negativă prin construcție"
                ],
                "correctExplanation": "Știrile care apar după închiderea de la București intră în BET abia a doua zi, deci aceeași știre cade în zile diferite: corelația din aceeași zi este subestimată; randamentele săptămînale reduc problema.",
                "incorrectExplanation": "Diferența dintre orele de închidere împarte știrile comune pe două zile, ceea ce slăbește mișcarea comună din aceeași zi; nu o mărește și nu o face negativă prin construcție."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Correlations in crises",
                "text": "Forbes and Rigobon (2002) warn about comparing sample correlations of calm and crisis periods. Why?",
                "options": [
                    "Correlations cannot be computed in crises",
                    "Higher volatility raises the sample correlation even if the link is unchanged",
                    "Crisis periods always have negative correlations",
                    "Sample correlations ignore the mean"
                ],
                "correctExplanation": "Conditioning on a high-volatility period inflates the sample correlation mechanically, so a higher crisis correlation is not by itself proof of contagion; DCC works with standardised residuals, which reduces this effect.",
                "incorrectExplanation": "Correlations can be computed in crises, they are not always negative there, and the issue is not the mean. The problem is the mechanical effect of higher volatility."
            },
            "ro": {
                "title": "Corelațiile în crize",
                "text": "Forbes și Rigobon (2002) atrag atenția asupra comparării corelațiilor de selecție din perioade calme și din crize. De ce?",
                "options": [
                    "Corelațiile nu pot fi calculate în crize",
                    "Volatilitatea mai mare crește corelația de selecție chiar dacă legătura nu s-a schimbat",
                    "Perioadele de criză au întotdeauna corelații negative",
                    "Corelațiile de selecție ignoră media"
                ],
                "correctExplanation": "Condiționarea pe o perioadă cu volatilitate mare crește mecanic corelația de selecție, deci o corelație mai mare în criză nu dovedește singură contagiunea; DCC lucrează cu reziduuri standardizate, ceea ce reduce acest efect.",
                "incorrectExplanation": "Corelațiile se pot calcula în crize, nu sînt întotdeauna negative atunci, iar problema nu este media. Problema este efectul mecanic al volatilității mai mari."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Portfolio VaR 1%",
                "text": "Today $\\sigma_{p,t} = 1.5\\%$ and the mean is 0. What is the Normal VaR 1% of the portfolio?",
                "options": [
                    "About 2.47%",
                    "About 1.50%",
                    "About 3.49%",
                    "About 2.94%"
                ],
                "correctExplanation": "$\\mathrm{VaR}_{0.01} = -(\\mu_p + z_{0.01}\\sigma_{p,t}) = 2.326 \\cdot 1.5 \\approx 3.49\\%$.",
                "incorrectExplanation": "2.47% uses the 5% quantile 1.645, 2.94% uses 1.96 and 1.50% ignores the quantile. VaR 1% needs $z_{0.01} = -2.326$."
            },
            "ro": {
                "title": "VaR 1% al portofoliului",
                "text": "Astăzi $\\sigma_{p,t} = 1{,}5\\%$, iar media este 0. Cît este VaR 1% al portofoliului, pentru distribuția Normală?",
                "options": [
                    "Aproximativ 2,47%",
                    "Aproximativ 1,50%",
                    "Aproximativ 3,49%",
                    "Aproximativ 2,94%"
                ],
                "correctExplanation": "$\\mathrm{VaR}_{0{,}01} = -(\\mu_p + z_{0{,}01}\\sigma_{p,t}) = 2{,}326 \\cdot 1{,}5 \\approx 3{,}49\\%$.",
                "incorrectExplanation": "2,47% folosește cuantila de 5% (1,645), 2,94% folosește 1,96, iar 1,50% ignoră cuantila. VaR 1% cere $z_{0{,}01} = -2{,}326$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The Kupiec test",
                "text": "What does the Kupiec (1995) test check in a VaR backtest?",
                "options": [
                    "Whether the share of violations equals the VaR level",
                    "Whether violations cluster in time",
                    "Whether returns follow the Normal distribution",
                    "Whether the VaR is too volatile"
                ],
                "correctExplanation": "The Kupiec test compares the observed violation rate $x/n$ with the nominal level (1%) by a likelihood ratio with $\\chi^2(1)$ distribution.",
                "incorrectExplanation": "Clustering is tested by Christoffersen (1998), normality by other tests, and the volatility of the VaR is not a test criterion. Kupiec tests the frequency of violations."
            },
            "ro": {
                "title": "Testul Kupiec",
                "text": "Ce verifică testul Kupiec (1995) în backtesting-ul unui VaR?",
                "options": [
                    "Dacă ponderea încălcărilor este egală cu nivelul VaR",
                    "Dacă încălcările se grupează în timp",
                    "Dacă randamentele urmează distribuția Normală",
                    "Dacă VaR-ul este prea volatil"
                ],
                "correctExplanation": "Testul Kupiec compară rata observată a încălcărilor $x/n$ cu nivelul nominal (1%) printr-un raport de verosimilitate cu distribuția $\\chi^2(1)$.",
                "incorrectExplanation": "Gruparea se testează cu Christoffersen (1998), normalitatea cu alte teste, iar volatilitatea VaR-ului nu este un criteriu de testare. Kupiec testează frecvența încălcărilor."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Clustering of violations",
                "text": "A static VaR 1% has the right number of violations over ten years, but ten of them fall in two months of 2020. What is wrong?",
                "options": [
                    "Nothing: the frequency is correct",
                    "The VaR level is wrong",
                    "The portfolio weights are wrong",
                    "The violations are not independent: the VaR does not adapt to volatility"
                ],
                "correctExplanation": "A good VaR has violations at random times. Clustering in a crisis means the risk forecast did not rise with volatility; the Christoffersen (1998) independence test detects it.",
                "incorrectExplanation": "A correct frequency is not enough, the level is fine, and the weights are not the issue. The problem is the dependence of the violations."
            },
            "ro": {
                "title": "Gruparea încălcărilor",
                "text": "Un VaR 1% static are numărul corect de încălcări în zece ani, dar zece dintre ele cad în două luni din 2020. Care este problema?",
                "options": [
                    "Niciuna: frecvența este corectă",
                    "Nivelul VaR este greșit",
                    "Ponderile portofoliului sînt greșite",
                    "Încălcările nu sînt independente: VaR-ul nu se adaptează volatilității"
                ],
                "correctExplanation": "Un VaR bun are încălcări la momente aleatoare. Gruparea într-o criză înseamnă că prognoza riscului nu a crescut odată cu volatilitatea; testul de independență Christoffersen (1998) o detectează.",
                "incorrectExplanation": "O frecvență corectă nu este suficientă, nivelul este corect, iar ponderile nu sînt problema. Problema este dependența încălcărilor."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The minimum-variance hedge ratio",
                "text": "$\\sigma_s = 2\\%$, $\\sigma_f = 1\\%$ and $\\rho = 0.5$. What is the minimum-variance hedge ratio $h^*$?",
                "options": [
                    "1.0",
                    "0.5",
                    "0.25",
                    "2.0"
                ],
                "correctExplanation": "$h^* = \\sigma_{sf}/\\sigma_f^2 = \\rho\\,\\sigma_s/\\sigma_f = 0.5 \\cdot 2/1 = 1.0$ unit of $f$ sold per unit of $s$.",
                "incorrectExplanation": "0.5 is the correlation alone, 0.25 is $\\rho^2$ (the hedging effectiveness) and 2.0 ignores the correlation. The ratio is $\\rho\\sigma_s/\\sigma_f = 1.0$."
            },
            "ro": {
                "title": "Raportul de acoperire cu varianță minimă",
                "text": "$\\sigma_s = 2\\%$, $\\sigma_f = 1\\%$ și $\\rho = 0{,}5$. Cît este raportul de acoperire cu varianță minimă $h^*$?",
                "options": [
                    "1,0",
                    "0,5",
                    "0,25",
                    "2,0"
                ],
                "correctExplanation": "$h^* = \\sigma_{sf}/\\sigma_f^2 = \\rho\\,\\sigma_s/\\sigma_f = 0{,}5 \\cdot 2/1 = 1{,}0$ unități din $f$ vîndute pentru o unitate din $s$.",
                "incorrectExplanation": "0,5 este doar corelația, 0,25 este $\\rho^2$ (eficiența acoperirii), iar 2,0 ignoră corelația. Raportul este $\\rho\\sigma_s/\\sigma_f = 1{,}0$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Hedging effectiveness",
                "text": "With the constant minimum-variance hedge ratio and correlation $\\rho = 0.6$, what share of the variance is removed?",
                "options": [
                    "36%",
                    "60%",
                    "40%",
                    "64%"
                ],
                "correctExplanation": "With $h = h^*$: $\\Var(r_s - h^*r_f) = \\sigma_s^2(1 - \\rho^2)$, so the effectiveness is $\\rho^2 = 0.36$.",
                "incorrectExplanation": "60% is $\\rho$, 64% is $1 - \\rho^2$ (the share of variance left) and 40% is $1 - \\rho$. The removed share is $\\rho^2 = 36\\%$."
            },
            "ro": {
                "title": "Eficiența acoperirii",
                "text": "Cu raportul de acoperire constant cu varianță minimă și corelația $\\rho = 0{,}6$, ce pondere din varianță este eliminată?",
                "options": [
                    "36%",
                    "60%",
                    "40%",
                    "64%"
                ],
                "correctExplanation": "Cu $h = h^*$: $\\Var(r_s - h^*r_f) = \\sigma_s^2(1 - \\rho^2)$, deci eficiența este $\\rho^2 = 0{,}36$.",
                "incorrectExplanation": "60% este $\\rho$, 64% este $1 - \\rho^2$ (ponderea varianței rămase), iar 40% este $1 - \\rho$. Ponderea eliminată este $\\rho^2 = 36\\%$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Mean and variance spillovers",
                "text": "How does a VAR model (Chapter 6) differ from a full BEKK model in the spillovers it measures?",
                "options": [
                    "VAR: spillovers in the variance; BEKK: spillovers in the mean",
                    "Both measure only long-run equilibria",
                    "Neither can measure spillovers",
                    "VAR: spillovers in the mean; BEKK: spillovers in the variance"
                ],
                "correctExplanation": "A VAR lets past returns of one market predict the returns of another (the mean); the off-diagonal elements of a full BEKK let past shocks of one market raise the variance of another.",
                "incorrectExplanation": "The roles are not reversed, and long-run equilibria are the subject of cointegration (Chapter 7). VAR describes mean spillovers, BEKK variance spillovers."
            },
            "ro": {
                "title": "Transmiterea în medie și în varianță",
                "text": "Prin ce diferă un model VAR (Capitolul 6) de un model BEKK complet în privința transmiterii pe care o măsoară?",
                "options": [
                    "VAR: transmiterea în varianță; BEKK: transmiterea în medie",
                    "Ambele măsoară doar echilibre pe termen lung",
                    "Niciunul nu poate măsura transmiterea",
                    "VAR: transmiterea în medie; BEKK: transmiterea în varianță"
                ],
                "correctExplanation": "Un VAR permite randamentelor trecute ale unei piețe să prognozeze randamentele alteia (media); elementele din afara diagonalei ale unui BEKK complet permit șocurilor trecute ale unei piețe să crească varianța alteia.",
                "incorrectExplanation": "Rolurile nu sînt inversate, iar echilibrele pe termen lung sînt tema cointegrării (Capitolul 7). VAR descrie transmiterea în medie, BEKK transmiterea în varianță."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Correlation and cointegration",
                "text": "Which statement about correlation and cointegration (Chapter 7) is correct?",
                "options": [
                    "Cointegration implies a daily return correlation close to one",
                    "High return correlation implies cointegration",
                    "Cointegration and correlation measure the same thing",
                    "Two prices can have highly correlated returns and still not be cointegrated"
                ],
                "correctExplanation": "Correlation is a short-run link between returns; cointegration is a long-run link between price levels. Prices with correlated returns can drift apart for ever, and cointegrated prices can have low daily correlation.",
                "incorrectExplanation": "Neither concept implies the other: one describes short-run comovement of returns, the other a long-run equilibrium of levels."
            },
            "ro": {
                "title": "Corelație și cointegrare",
                "text": "Care afirmație despre corelație și cointegrare (Capitolul 7) este corectă?",
                "options": [
                    "Cointegrarea implică o corelație zilnică a randamentelor apropiată de unu",
                    "O corelație mare a randamentelor implică cointegrarea",
                    "Cointegrarea și corelația măsoară același lucru",
                    "Două prețuri pot avea randamente puternic corelate fără să fie cointegrate"
                ],
                "correctExplanation": "Corelația este o legătură pe termen scurt între randamente; cointegrarea este o legătură pe termen lung între nivelurile prețurilor. Prețuri cu randamente corelate se pot îndepărta definitiv, iar prețuri cointegrate pot avea o corelație zilnică mică.",
                "incorrectExplanation": "Niciun concept nu îl implică pe celălalt: unul descrie mișcarea comună pe termen scurt a randamentelor, celălalt un echilibru pe termen lung al nivelurilor."
            }
        }
    ]
};
