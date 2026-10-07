// ============================================================
// Chapter 6 quiz bank: VAR models and Granger causality (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['var'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "Coefficient matrices in a VAR",
                "text": "In a VAR(p) model with $K$ variables, what is the dimension of each coefficient matrix $\\mathbf{A}_i$?",
                "options": [
                    "$K \\times K$",
                    "$K \\times 1$",
                    "$K \\times p$",
                    "$Kp \\times Kp$"
                ],
                "correctExplanation": "Each $\\mathbf{A}_i$ maps the $K$-vector $\\mathbf{Y}_{t-i}$ into a contribution to the $K$-vector $\\mathbf{Y}_t$, so it is $K \\times K$. A VAR(p) has $p$ such matrices, one for each lag, whatever the value of $K$.",
                "incorrectExplanation": "$K \\times 1$ is the size of the constant vector and $Kp \\times Kp$ is the size of the companion matrix; $K \\times p$ mixes variables and lags. Each lag matrix $\\mathbf{A}_i$ is $K \\times K$."
            },
            "ro": {
                "title": "Matricele de coeficienți ale unui VAR",
                "text": "Într-un model VAR(p) cu $K$ variabile, care este dimensiunea fiecărei matrice de coeficienți $\\mathbf{A}_i$?",
                "options": [
                    "$K \\times K$",
                    "$K \\times 1$",
                    "$K \\times p$",
                    "$Kp \\times Kp$"
                ],
                "correctExplanation": "Fiecare matrice $\\mathbf{A}_i$ transformă vectorul $\\mathbf{Y}_{t-i}$, de dimensiune $K$, într-o contribuție la vectorul $\\mathbf{Y}_t$, tot de dimensiune $K$; prin urmare, are dimensiunea $K \\times K$. Un VAR(p) are $p$ astfel de matrice, cîte una pentru fiecare lag, indiferent de $K$.",
                "incorrectExplanation": "$K \\times 1$ este dimensiunea vectorului de constante, iar $Kp \\times Kp$ este dimensiunea matricei companion; $K \\times p$ amestecă variabilele cu lagurile. Fiecare matrice $\\mathbf{A}_i$ are dimensiunea $K \\times K$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Number of coefficients: general formula",
                "text": "A VAR(p) with $K$ variables has a constant in each equation. How many coefficients (constants and lag coefficients, excluding the error covariance matrix) does it have in total?",
                "options": [
                    "$K^2 p$",
                    "$K(1 + Kp)$",
                    "$1 + Kp$",
                    "$K + p + K^2$"
                ],
                "correctExplanation": "Each of the $K$ equations has one constant and $Kp$ lag coefficients ($K$ variables times $p$ lags), so the total is $K(1 + Kp) = K + pK^2$.",
                "incorrectExplanation": "$K^2 p$ forgets the $K$ constants, $1 + Kp$ counts a single equation only, and $K + p + K^2$ adds the lag order instead of multiplying by it. The total is $K(1 + Kp)$, which grows with $K^2$."
            },
            "ro": {
                "title": "Numărul de coeficienți: formula generală",
                "text": "Un VAR(p) cu $K$ variabile are o constantă în fiecare ecuație. Cîți coeficienți (constante și coeficienți ai lagurilor, fără matricea de covarianță a erorilor) are în total?",
                "options": [
                    "$K^2 p$",
                    "$K(1 + Kp)$",
                    "$1 + Kp$",
                    "$K + p + K^2$"
                ],
                "correctExplanation": "Fiecare dintre cele $K$ ecuații are o constantă și $Kp$ coeficienți ai lagurilor ($K$ variabile înmulțite cu $p$ laguri), deci totalul este $K(1 + Kp) = K + pK^2$.",
                "incorrectExplanation": "$K^2 p$ omite cele $K$ constante, $1 + Kp$ numără o singură ecuație, iar $K + p + K^2$ adună numărul de laguri în loc să înmulțească cu el. Totalul este $K(1 + Kp)$ și crește cu $K^2$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Stability of a VAR(1)",
                "text": "A VAR(1) model $\\mathbf{Y}_t = \\mathbf{c} + \\mathbf{A}_1 \\mathbf{Y}_{t-1} + \\boldsymbol{\\varepsilon}_t$ is stable (and hence stationary) if:",
                "options": [
                    "All diagonal elements of $\\mathbf{A}_1$ are less than 1",
                    "The determinant of $\\mathbf{A}_1$ is less than 1",
                    "All eigenvalues of $\\mathbf{A}_1$ are less than 1 in absolute value",
                    "The trace of $\\mathbf{A}_1$ equals zero"
                ],
                "correctExplanation": "Stability requires all eigenvalues inside the unit circle, $|\\lambda_i| < 1$. Then $\\mathbf{A}_1^h \\to \\mathbf{0}$ and the effect of every shock dies out.",
                "incorrectExplanation": "Diagonal elements, the determinant or the trace summarise $\\mathbf{A}_1$ only partially: a matrix with small diagonal entries or a determinant below 1 can still have an eigenvalue of modulus above 1. The condition is on the moduli of all eigenvalues."
            },
            "ro": {
                "title": "Stabilitatea unui VAR(1)",
                "text": "Un model VAR(1) $\\mathbf{Y}_t = \\mathbf{c} + \\mathbf{A}_1 \\mathbf{Y}_{t-1} + \\boldsymbol{\\varepsilon}_t$ este stabil (deci staționar) dacă:",
                "options": [
                    "Toate elementele de pe diagonala lui $\\mathbf{A}_1$ sînt mai mici decît 1",
                    "Determinantul lui $\\mathbf{A}_1$ este mai mic decît 1",
                    "Toate valorile proprii ale lui $\\mathbf{A}_1$ au modulul strict mai mic decît 1",
                    "Urma lui $\\mathbf{A}_1$ este egală cu zero"
                ],
                "correctExplanation": "Stabilitatea cere ca toate valorile proprii să fie în interiorul cercului unitate, $|\\lambda_i| < 1$. Atunci $\\mathbf{A}_1^h \\to \\mathbf{0}$, iar efectul oricărui șoc se stinge.",
                "incorrectExplanation": "Elementele diagonale, determinantul sau urma descriu matricea $\\mathbf{A}_1$ doar parțial: o matrice cu elemente diagonale mici sau cu determinantul sub 1 poate avea totuși o valoare proprie cu modulul peste 1. Condiția privește modulele tuturor valorilor proprii."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Stability condition for a VAR(p)",
                "text": "For a general VAR(p), the stability condition requires that all roots of $\\det(\\mathbf{I}_K - \\mathbf{A}_1 z - \\cdots - \\mathbf{A}_p z^p) = 0$ lie:",
                "options": [
                    "Inside the unit circle",
                    "On the unit circle",
                    "At the origin",
                    "Outside the unit circle"
                ],
                "correctExplanation": "In the lag-polynomial form the roots must lie outside the unit circle, $|z| > 1$. This is equivalent to all eigenvalues of the companion matrix lying inside the unit circle, since the roots are the reciprocals of those eigenvalues.",
                "incorrectExplanation": "\"Inside the unit circle\" is the condition on the eigenvalues of the companion matrix, not on the roots of the lag polynomial; roots on the unit circle mean unit roots (nonstationarity). For the polynomial $\\det(\\mathbf{I}_K - \\mathbf{A}_1 z - \\cdots - \\mathbf{A}_p z^p)$ all roots must satisfy $|z| > 1$."
            },
            "ro": {
                "title": "Condiția de stabilitate pentru un VAR(p)",
                "text": "Pentru un VAR(p) general, condiția de stabilitate cere ca toate rădăcinile ecuației $\\det(\\mathbf{I}_K - \\mathbf{A}_1 z - \\cdots - \\mathbf{A}_p z^p) = 0$ să se afle:",
                "options": [
                    "În interiorul cercului unitate",
                    "Pe cercul unitate",
                    "În origine",
                    "În afara cercului unitate"
                ],
                "correctExplanation": "În forma cu polinomul lag, rădăcinile trebuie să fie în afara cercului unitate, $|z| > 1$. Condiția este echivalentă cu aceea ca toate valorile proprii ale matricei companion să fie în interiorul cercului unitate, deoarece rădăcinile sînt inversele acestor valori proprii.",
                "incorrectExplanation": "„În interiorul cercului unitate” este condiția pentru valorile proprii ale matricei companion, nu pentru rădăcinile polinomului lag; rădăcini pe cercul unitate înseamnă rădăcini unitare (nestaționaritate). Pentru polinomul $\\det(\\mathbf{I}_K - \\mathbf{A}_1 z - \\cdots - \\mathbf{A}_p z^p)$, toate rădăcinile trebuie să verifice $|z| > 1$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Purpose of the companion form",
                "text": "The companion form of a VAR(p) is useful because it:",
                "options": [
                    "Rewrites any VAR(p) as a VAR(1), which simplifies the stability analysis, forecasting and the computation of impulse responses",
                    "Reduces the number of parameters to estimate",
                    "Eliminates the need for the Cholesky decomposition",
                    "Makes the error covariance matrix diagonal"
                ],
                "correctExplanation": "Stacking $\\mathbf{Y}_t, \\dots, \\mathbf{Y}_{t-p+1}$ into one $Kp$-vector turns a VAR(p) into a VAR(1). Stability is then checked through the eigenvalues of a single $Kp \\times Kp$ matrix, and forecasts and impulse responses follow from its powers.",
                "incorrectExplanation": "The companion form is only a rewriting: it estimates the same coefficients, it does not orthogonalise the shocks (that still needs Cholesky or another identification) and it leaves the error covariance unchanged. Its value is the first-order structure."
            },
            "ro": {
                "title": "Rolul formei companion",
                "text": "Forma companion a unui VAR(p) este utilă deoarece:",
                "options": [
                    "Rescrie orice VAR(p) ca VAR(1), ceea ce simplifică analiza stabilității, prognoza și calculul funcțiilor de răspuns la impuls",
                    "Reduce numărul de parametri care trebuie estimați",
                    "Elimină nevoia descompunerii Cholesky",
                    "Face diagonală matricea de covarianță a erorilor"
                ],
                "correctExplanation": "Prin stivuirea vectorilor $\\mathbf{Y}_t, \\dots, \\mathbf{Y}_{t-p+1}$ într-un singur vector de dimensiune $Kp$, un VAR(p) devine un VAR(1). Stabilitatea se verifică apoi prin valorile proprii ale unei singure matrice $Kp \\times Kp$, iar prognozele și răspunsurile la impuls se obțin din puterile acesteia.",
                "incorrectExplanation": "Forma companion este doar o rescriere: estimează aceiași coeficienți, nu ortogonalizează șocurile (pentru aceasta este nevoie în continuare de Cholesky sau de altă schemă de identificare) și nu modifică matricea de covarianță a erorilor. Avantajul ei este structura de ordinul întîi."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Moving-average representation of a VAR(1)",
                "text": "The impulse response matrices $\\boldsymbol{\\Phi}_h$ in the MA($\\infty$) representation of a stable VAR(1) $\\mathbf{Y}_t = \\mathbf{A}\\mathbf{Y}_{t-1} + \\boldsymbol{\\varepsilon}_t$ are given by:",
                "options": [
                    "$\\boldsymbol{\\Phi}_h = h \\cdot \\mathbf{A}$",
                    "$\\boldsymbol{\\Phi}_h = \\mathbf{A}^h$",
                    "$\\boldsymbol{\\Phi}_h = \\mathbf{A}^{-h}$",
                    "$\\boldsymbol{\\Phi}_h = \\boldsymbol{\\Sigma}^h$"
                ],
                "correctExplanation": "Recursive substitution gives $\\mathbf{Y}_t = \\sum_{h \\ge 0} \\mathbf{A}^h \\boldsymbol{\\varepsilon}_{t-h}$, so the response at horizon $h$ to a shock at time $t$ is $\\boldsymbol{\\Phi}_h = \\mathbf{A}^h$.",
                "incorrectExplanation": "The responses are not linear in $h$, they do not involve the inverse of $\\mathbf{A}$ (which would make them explode in a stable model) and they do not depend on $\\boldsymbol{\\Sigma}$; the covariance enters only when the shocks are orthogonalised. For a VAR(1), $\\boldsymbol{\\Phi}_h = \\mathbf{A}^h$."
            },
            "ro": {
                "title": "Reprezentarea de medie mobilă a unui VAR(1)",
                "text": "Matricele de răspuns la impuls $\\boldsymbol{\\Phi}_h$ din reprezentarea MA($\\infty$) a unui VAR(1) stabil $\\mathbf{Y}_t = \\mathbf{A}\\mathbf{Y}_{t-1} + \\boldsymbol{\\varepsilon}_t$ sînt date de:",
                "options": [
                    "$\\boldsymbol{\\Phi}_h = h \\cdot \\mathbf{A}$",
                    "$\\boldsymbol{\\Phi}_h = \\mathbf{A}^h$",
                    "$\\boldsymbol{\\Phi}_h = \\mathbf{A}^{-h}$",
                    "$\\boldsymbol{\\Phi}_h = \\boldsymbol{\\Sigma}^h$"
                ],
                "correctExplanation": "Prin substituție recursivă se obține $\\mathbf{Y}_t = \\sum_{h \\ge 0} \\mathbf{A}^h \\boldsymbol{\\varepsilon}_{t-h}$, deci răspunsul la orizontul $h$ la un șoc din momentul $t$ este $\\boldsymbol{\\Phi}_h = \\mathbf{A}^h$.",
                "incorrectExplanation": "Răspunsurile nu sînt liniare în $h$, nu folosesc inversa lui $\\mathbf{A}$ (care le-ar face explozive într-un model stabil) și nu depind de $\\boldsymbol{\\Sigma}$; matricea de covarianță intervine doar la ortogonalizarea șocurilor. Pentru un VAR(1), $\\boldsymbol{\\Phi}_h = \\mathbf{A}^h$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Orthogonalised impulse responses",
                "text": "The purpose of orthogonalising impulse responses with the Cholesky decomposition is to:",
                "options": [
                    "Increase the number of parameters in the model",
                    "Make the model stationary",
                    "Obtain uncorrelated shocks, so that the effect of each shock can be isolated",
                    "Remove the constant term from the VAR"
                ],
                "correctExplanation": "Reduced-form errors are correlated ($\\boldsymbol{\\Sigma}$ is not diagonal), so \"a shock to one variable alone\" is not well defined. Writing $\\boldsymbol{\\Sigma} = \\mathbf{P}\\mathbf{P}^\\prime$ and using $\\mathbf{u}_t = \\mathbf{P}^{-1}\\boldsymbol{\\varepsilon}_t$ gives unit-variance, mutually uncorrelated shocks whose effects can be traced one at a time.",
                "incorrectExplanation": "Orthogonalisation does not add parameters, does not change stationarity (that depends on $\\mathbf{A}_i$ only) and does not touch the constant. It only transforms correlated errors into uncorrelated shocks."
            },
            "ro": {
                "title": "Funcții de răspuns la impuls ortogonalizate",
                "text": "Scopul ortogonalizării funcțiilor de răspuns la impuls prin descompunerea Cholesky este:",
                "options": [
                    "Creșterea numărului de parametri ai modelului",
                    "Transformarea modelului într-unul staționar",
                    "Obținerea unor șocuri necorelate, astfel încît efectul fiecărui șoc să poată fi izolat",
                    "Eliminarea termenului liber din VAR"
                ],
                "correctExplanation": "Erorile din forma redusă sînt corelate ($\\boldsymbol{\\Sigma}$ nu este diagonală), deci „un șoc doar asupra unei variabile” nu este bine definit. Scriind $\\boldsymbol{\\Sigma} = \\mathbf{P}\\mathbf{P}^\\prime$ și folosind $\\mathbf{u}_t = \\mathbf{P}^{-1}\\boldsymbol{\\varepsilon}_t$, se obțin șocuri cu varianță unitară, necorelate între ele, ale căror efecte pot fi urmărite pe rînd.",
                "incorrectExplanation": "Ortogonalizarea nu adaugă parametri, nu schimbă staționaritatea (aceasta depinde doar de matricele $\\mathbf{A}_i$) și nu afectează termenul liber. Ea doar transformă erorile corelate în șocuri necorelate."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Cholesky decomposition",
                "text": "In the Cholesky decomposition $\\boldsymbol{\\Sigma} = \\mathbf{P}\\mathbf{P}^\\prime$ used for orthogonalised impulse responses, the matrix $\\mathbf{P}$ is:",
                "options": [
                    "Orthogonal",
                    "Diagonal",
                    "Upper triangular",
                    "Lower triangular"
                ],
                "correctExplanation": "$\\mathbf{P}$ is lower triangular with positive diagonal. Its zeros above the diagonal mean that the first variable reacts on impact only to its own shock, the second to the first two shocks, and so on.",
                "incorrectExplanation": "An orthogonal matrix would give $\\mathbf{P}\\mathbf{P}^\\prime = \\mathbf{I}$, and a diagonal $\\mathbf{P}$ would only be possible if $\\boldsymbol{\\Sigma}$ were already diagonal. In the convention $\\boldsymbol{\\Sigma} = \\mathbf{P}\\mathbf{P}^\\prime$ the Cholesky factor is lower triangular (the upper triangular factor appears in the form $\\mathbf{R}^\\prime\\mathbf{R}$)."
            },
            "ro": {
                "title": "Descompunerea Cholesky",
                "text": "În descompunerea Cholesky $\\boldsymbol{\\Sigma} = \\mathbf{P}\\mathbf{P}^\\prime$ folosită pentru funcțiile de răspuns la impuls ortogonalizate, matricea $\\mathbf{P}$ este:",
                "options": [
                    "Ortogonală",
                    "Diagonală",
                    "Superior triunghiulară",
                    "Inferior triunghiulară"
                ],
                "correctExplanation": "$\\mathbf{P}$ este inferior triunghiulară, cu diagonala pozitivă. Zerourile de deasupra diagonalei înseamnă că prima variabilă reacționează la impact doar la propriul șoc, a doua la primele două șocuri și așa mai departe.",
                "incorrectExplanation": "O matrice ortogonală ar da $\\mathbf{P}\\mathbf{P}^\\prime = \\mathbf{I}$, iar o matrice $\\mathbf{P}$ diagonală ar fi posibilă doar dacă $\\boldsymbol{\\Sigma}$ ar fi deja diagonală. În convenția $\\boldsymbol{\\Sigma} = \\mathbf{P}\\mathbf{P}^\\prime$, factorul Cholesky este inferior triunghiular (factorul superior triunghiular apare în forma $\\mathbf{R}^\\prime\\mathbf{R}$)."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Variable ordering in Cholesky identification",
                "text": "Why does the ordering of the variables matter when orthogonalised impulse responses are computed with the Cholesky decomposition?",
                "options": [
                    "Because the lower triangular matrix P forces the first variable not to respond contemporaneously to the shocks of the variables ordered after it",
                    "Because OLS estimates depend on the order of the variables",
                    "Because AIC and BIC give different results depending on the ordering",
                    "It does not matter: the results are identical for every ordering"
                ],
                "correctExplanation": "Cholesky imposes a recursive structure: a variable ordered earlier can affect the later ones within the same period, but not the other way round. A different ordering is a different identifying assumption and, unless the errors are uncorrelated, gives different impulse responses.",
                "incorrectExplanation": "The reduced-form VAR (OLS estimates, information criteria, residuals) is invariant to the ordering; only the orthogonalisation step depends on it. The results coincide across orderings only when the reduced-form errors are uncorrelated."
            },
            "ro": {
                "title": "Ordinea variabilelor în identificarea Cholesky",
                "text": "De ce contează ordinea variabilelor atunci cînd funcțiile de răspuns la impuls ortogonalizate se calculează prin descompunerea Cholesky?",
                "options": [
                    "Deoarece matricea inferior triunghiulară P impune ca prima variabilă să nu răspundă contemporan la șocurile variabilelor așezate după ea",
                    "Deoarece estimațiile OLS depind de ordinea variabilelor",
                    "Deoarece AIC și BIC dau rezultate diferite în funcție de ordine",
                    "Nu contează: rezultatele sînt identice pentru orice ordine"
                ],
                "correctExplanation": "Descompunerea Cholesky impune o structură recursivă: o variabilă așezată mai devreme le poate afecta pe cele de după ea în aceeași perioadă, dar nu și invers. O altă ordine reprezintă o altă ipoteză de identificare și, dacă erorile nu sînt necorelate, conduce la alte răspunsuri la impuls.",
                "incorrectExplanation": "VAR-ul în formă redusă (estimațiile OLS, criteriile informaționale, reziduurile) nu depinde de ordinea variabilelor; doar etapa de ortogonalizare depinde de ea. Rezultatele coincid pentru orice ordine doar atunci cînd erorile din forma redusă sînt necorelate."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Interpretation of the FEVD",
                "text": "The forecast error variance decomposition (FEVD) shows that 70% of the forecast error variance of unemployment at horizon 8 is due to GDP shocks. This means that:",
                "options": [
                    "GDP directly causes 70% of unemployment",
                    "70% of the uncertainty in forecasting unemployment 8 periods ahead is attributable to GDP shocks",
                    "Unemployment will increase by 70% after 8 periods",
                    "The correlation between GDP and unemployment is 0.70"
                ],
                "correctExplanation": "The FEVD splits the $h$-step forecast error variance of each variable into the shares contributed by each (orthogonalised) shock. Here GDP shocks account for 70% of the 8-step forecast uncertainty of unemployment.",
                "incorrectExplanation": "The FEVD is a statement about forecast uncertainty, not about the level of unemployment, a growth rate or a correlation coefficient; and, like any Cholesky-based result, it depends on the identifying assumptions."
            },
            "ro": {
                "title": "Interpretarea FEVD",
                "text": "Descompunerea varianței erorii de prognoză (FEVD) arată că 70% din varianța erorii de prognoză a șomajului la orizontul 8 se datorează șocurilor PIB. Aceasta înseamnă că:",
                "options": [
                    "PIB-ul determină direct 70% din șomaj",
                    "70% din incertitudinea prognozei șomajului cu 8 perioade înainte se datorează șocurilor PIB",
                    "Șomajul va crește cu 70% după 8 perioade",
                    "Corelația dintre PIB și șomaj este 0,70"
                ],
                "correctExplanation": "FEVD împarte varianța erorii de prognoză la orizontul $h$ a fiecărei variabile în ponderile datorate fiecărui șoc (ortogonalizat). Aici, șocurile PIB explică 70% din incertitudinea prognozei șomajului la 8 perioade.",
                "incorrectExplanation": "FEVD descrie incertitudinea prognozei, nu nivelul șomajului, o rată de creștere sau un coeficient de corelație; în plus, ca orice rezultat obținut prin Cholesky, depinde de ipotezele de identificare."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Granger causality",
                "text": "What does the statement \"$X$ Granger-causes $Y$\" mean?",
                "options": [
                    "$X$ is the structural economic cause of $Y$",
                    "$X$ and $Y$ are contemporaneously correlated",
                    "Past values of $X$ help forecast $Y$ beyond the information contained in the past of $Y$",
                    "$X$ and $Y$ share the same long-run trend"
                ],
                "correctExplanation": "Granger causality is about incremental predictive content: lagged $X$ reduces the forecast error of $Y$ once lagged $Y$ is already used. It can be spurious (omitted variables, anticipation effects), so it does not establish economic causation.",
                "incorrectExplanation": "Predictability is not structural causation; contemporaneous correlation is instantaneous causality, which involves no lags; a common long-run trend is cointegration. Granger causality concerns lagged $X$ in the equation of $Y$."
            },
            "ro": {
                "title": "Cauzalitatea Granger",
                "text": "Ce înseamnă afirmația „$X$ cauzează Granger pe $Y$”?",
                "options": [
                    "$X$ este cauza economică structurală a lui $Y$",
                    "$X$ și $Y$ sînt corelate contemporan",
                    "Valorile trecute ale lui $X$ ajută la prognoza lui $Y$, dincolo de informația din trecutul lui $Y$",
                    "$X$ și $Y$ au aceeași tendință pe termen lung"
                ],
                "correctExplanation": "Cauzalitatea Granger privește conținutul predictiv suplimentar: lagurile lui $X$ reduc eroarea de prognoză a lui $Y$ după ce sînt folosite deja lagurile lui $Y$. Ea poate fi falsă (variabile omise, efecte de anticipare), deci nu dovedește o cauzalitate economică.",
                "incorrectExplanation": "Predictibilitatea nu înseamnă cauzalitate structurală; corelația contemporană ține de cauzalitatea instantanee, care nu implică laguri; o tendință comună pe termen lung înseamnă cointegrare. Cauzalitatea Granger privește lagurile lui $X$ din ecuația lui $Y$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Granger causality test in a VAR",
                "text": "In a bivariate VAR(p), $Y_2$ does not Granger-cause $Y_1$ if:",
                "options": [
                    "All coefficients in the equation of $Y_2$ are zero",
                    "The eigenvalues of the matrix $\\mathbf{A}$ are zero",
                    "The contemporaneous error covariance $\\sigma_{12} = 0$",
                    "$a_{12}^{(1)} = a_{12}^{(2)} = \\cdots = a_{12}^{(p)} = 0$"
                ],
                "correctExplanation": "$a_{12}^{(i)}$ is the coefficient of $Y_{2,t-i}$ in the equation of $Y_1$. Non-causality is the joint null $H_0\\colon a_{12}^{(1)} = \\cdots = a_{12}^{(p)} = 0$, tested with a Wald or $F$ test.",
                "incorrectExplanation": "The equation of $Y_2$ is where one tests whether $Y_1$ Granger-causes $Y_2$; $\\sigma_{12} = 0$ concerns instantaneous causality; eigenvalues concern stability. Only the lags of $Y_2$ in the equation of $Y_1$ matter here."
            },
            "ro": {
                "title": "Testul Granger într-un VAR",
                "text": "Într-un VAR(p) bivariat, $Y_2$ nu cauzează Granger pe $Y_1$ dacă:",
                "options": [
                    "Toți coeficienții din ecuația lui $Y_2$ sînt zero",
                    "Valorile proprii ale matricei $\\mathbf{A}$ sînt nule",
                    "Covarianța contemporană a erorilor $\\sigma_{12} = 0$",
                    "$a_{12}^{(1)} = a_{12}^{(2)} = \\cdots = a_{12}^{(p)} = 0$"
                ],
                "correctExplanation": "$a_{12}^{(i)}$ este coeficientul lui $Y_{2,t-i}$ din ecuația lui $Y_1$. Lipsa cauzalității este ipoteza nulă comună $H_0\\colon a_{12}^{(1)} = \\cdots = a_{12}^{(p)} = 0$, testată cu un test Wald sau $F$.",
                "incorrectExplanation": "În ecuația lui $Y_2$ se testează dacă $Y_1$ cauzează Granger pe $Y_2$; $\\sigma_{12} = 0$ privește cauzalitatea instantanee; valorile proprii privesc stabilitatea. Aici contează doar lagurile lui $Y_2$ din ecuația lui $Y_1$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Granger causality test: numerical example",
                "text": "A bivariate VAR(2) is estimated on $T = 100$ observations. In the equation of $Y_1$, the unrestricted residual sum of squares is $RSS_U = 45.2$ and the restricted one (without the lags of $Y_2$) is $RSS_R = 52.8$. The $F$ statistic of the Granger causality test is approximately:",
                "options": [
                    "7.99",
                    "5.42",
                    "3.09",
                    "12.35"
                ],
                "correctExplanation": "$F = \\dfrac{(RSS_R - RSS_U)/p}{RSS_U/(T - 2p - 1)} = \\dfrac{(52.8 - 45.2)/2}{45.2/95} = \\dfrac{3.8}{0.4758} \\approx 7.99$, with $(2, 95)$ degrees of freedom; at the 5% level $H_0$ is rejected.",
                "incorrectExplanation": "The other values come from wrong degrees of freedom or from dividing by the wrong sum of squares. The numerator has $p = 2$ restrictions and the denominator $T - 2p - 1 = 95$ degrees of freedom (five coefficients in the unrestricted equation)."
            },
            "ro": {
                "title": "Testul de cauzalitate Granger: exemplu numeric",
                "text": "Un VAR(2) bivariat este estimat pe $T = 100$ de observații. În ecuația lui $Y_1$, suma pătratelor reziduurilor modelului nerestricționat este $RSS_U = 45{,}2$, iar a celui restricționat (fără lagurile lui $Y_2$) este $RSS_R = 52{,}8$. Statistica $F$ a testului de cauzalitate Granger este aproximativ:",
                "options": [
                    "7,98",
                    "5,42",
                    "3,09",
                    "12,35"
                ],
                "correctExplanation": "$F = \\dfrac{(RSS_R - RSS_U)/p}{RSS_U/(T - 2p - 1)} = \\dfrac{(52{,}8 - 45{,}2)/2}{45{,}2/95} = \\dfrac{3{,}8}{0{,}4758} \\approx 7{,}99$, cu $(2, 95)$ grade de libertate; la pragul de 5%, $H_0$ se respinge.",
                "incorrectExplanation": "Celelalte valori provin din grade de libertate greșite sau din împărțirea la altă sumă de pătrate. Numărătorul are $p = 2$ restricții, iar numitorul $T - 2p - 1 = 95$ de grade de libertate (cinci coeficienți în ecuația nerestricționată)."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Pitfalls of Granger causality",
                "text": "Which of the following can produce a spurious finding that $X$ Granger-causes $Y$?",
                "options": [
                    "Adding a further relevant variable to the VAR",
                    "An omitted third variable $Z$ that drives both $X$ and $Y$, with $X$ reacting earlier",
                    "Estimating the VAR by OLS equation by equation",
                    "Testing the restrictions with an $F$ test instead of a Wald test"
                ],
                "correctExplanation": "If $Z$ moves $X$ before it moves $Y$ and $Z$ is not in the system, lagged $X$ acts as a proxy for $Z$ and appears to predict $Y$ even though $X$ has no effect on $Y$.",
                "incorrectExplanation": "Adding relevant variables reduces, rather than creates, this omitted-variable problem; OLS equation by equation is the standard estimator of a VAR; $F$ and Wald tests are asymptotically equivalent ways of testing the same restrictions."
            },
            "ro": {
                "title": "Capcanele cauzalității Granger",
                "text": "Care dintre următoarele poate produce o concluzie falsă că $X$ cauzează Granger pe $Y$?",
                "options": [
                    "Adăugarea în VAR a încă unei variabile relevante",
                    "O a treia variabilă omisă $Z$, care le influențează atît pe $X$, cît și pe $Y$, iar $X$ reacționează mai devreme",
                    "Estimarea VAR prin OLS, ecuație cu ecuație",
                    "Testarea restricțiilor cu un test $F$ în loc de un test Wald"
                ],
                "correctExplanation": "Dacă $Z$ îl influențează pe $X$ înaintea lui $Y$ și nu este inclusă în sistem, lagurile lui $X$ țin locul lui $Z$ și par să prognozeze $Y$, deși $X$ nu are niciun efect asupra lui $Y$.",
                "incorrectExplanation": "Adăugarea unor variabile relevante reduce, nu creează, problema variabilelor omise; estimarea OLS ecuație cu ecuație este estimatorul standard al unui VAR; testele $F$ și Wald sînt asimptotic echivalente și testează aceleași restricții."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Instantaneous causality",
                "text": "Instantaneous causality between two variables of a VAR is tested by checking whether:",
                "options": [
                    "Lagged $X$ helps predict $Y$",
                    "$X$ and $Y$ share a common stochastic trend",
                    "The reduced-form errors are correlated within the same period, $\\sigma_{12} \\neq 0$",
                    "The VAR is stable"
                ],
                "correctExplanation": "Instantaneous causality concerns the same-period correlation of the errors, $H_0\\colon \\sigma_{12} = \\operatorname{Cov}(\\varepsilon_{1t}, \\varepsilon_{2t}) = 0$. It is symmetric and has no direction.",
                "incorrectExplanation": "Lagged predictive content is Granger causality; a common stochastic trend is cointegration; stability is an eigenvalue condition. Instantaneous causality involves no lags at all."
            },
            "ro": {
                "title": "Cauzalitate instantanee",
                "text": "Cauzalitatea instantanee dintre două variabile ale unui VAR se testează verificînd dacă:",
                "options": [
                    "Lagurile lui $X$ ajută la prognoza lui $Y$",
                    "$X$ și $Y$ au o tendință stochastică comună",
                    "Erorile din forma redusă sînt corelate în aceeași perioadă, $\\sigma_{12} \\neq 0$",
                    "VAR-ul este stabil"
                ],
                "correctExplanation": "Cauzalitatea instantanee privește corelația erorilor din aceeași perioadă, $H_0\\colon \\sigma_{12} = \\operatorname{Cov}(\\varepsilon_{1t}, \\varepsilon_{2t}) = 0$. Ea este simetrică și nu are o direcție.",
                "incorrectExplanation": "Conținutul predictiv al lagurilor ține de cauzalitatea Granger; o tendință stochastică comună înseamnă cointegrare; stabilitatea este o condiție asupra valorilor proprii. Cauzalitatea instantanee nu implică laguri."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Toda-Yamamoto procedure",
                "text": "What is the main purpose of the Toda-Yamamoto procedure for testing Granger causality?",
                "options": [
                    "To improve the power of the test in small samples",
                    "To avoid estimating a VAR altogether",
                    "To reduce the number of estimated parameters",
                    "To allow valid Granger causality tests when the data are nonstationary or cointegrated"
                ],
                "correctExplanation": "With integrated or cointegrated variables the usual Wald test in a levels VAR has a nonstandard distribution. Toda and Yamamoto (1995) estimate a VAR($p + d_{max}$) in levels, where $d_{max}$ is the maximal order of integration, and test only the first $p$ lags, which restores the asymptotic $\\chi^2$ distribution.",
                "incorrectExplanation": "The procedure still estimates a VAR, it adds parameters (the extra $d_{max}$ lags) rather than removing them, and it slightly reduces power. Its point is validity of the Wald test regardless of unit roots or cointegration."
            },
            "ro": {
                "title": "Procedura Toda-Yamamoto",
                "text": "Care este scopul principal al procedurii Toda-Yamamoto pentru testarea cauzalității Granger?",
                "options": [
                    "Creșterea puterii testului în eșantioane mici",
                    "Evitarea completă a estimării unui VAR",
                    "Reducerea numărului de parametri estimați",
                    "Testarea validă a cauzalității Granger atunci cînd datele sînt nestaționare sau cointegrate"
                ],
                "correctExplanation": "Cînd variabilele sînt integrate sau cointegrate, testul Wald obișnuit într-un VAR în niveluri are o distribuție nestandard. Toda și Yamamoto (1995) estimează un VAR($p + d_{max}$) în niveluri, unde $d_{max}$ este ordinul maxim de integrare, și testează doar primele $p$ laguri, ceea ce restabilește distribuția asimptotică $\\chi^2$.",
                "incorrectExplanation": "Procedura estimează în continuare un VAR, adaugă parametri (cele $d_{max}$ laguri suplimentare) în loc să îi elimine și reduce ușor puterea testului. Scopul ei este validitatea testului Wald indiferent de rădăcinile unitare sau de cointegrare."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Information criteria for the lag order",
                "text": "Comparing AIC and BIC for selecting the lag order of a VAR, which statement is correct?",
                "options": [
                    "BIC penalises complexity more heavily than AIC and tends to select smaller models",
                    "AIC penalises complexity more heavily than BIC and selects smaller models",
                    "AIC and BIC always select the same model",
                    "BIC cannot be used for VAR models"
                ],
                "correctExplanation": "The penalty per parameter is $2$ for AIC and $\\ln T$ for BIC; since $\\ln T > 2$ for $T \\ge 8$, BIC favours more parsimonious models (and is consistent), while AIC tends to choose longer lags.",
                "incorrectExplanation": "The ordering of the penalties is the other way round, the two criteria often disagree, and BIC is routinely reported for VAR lag selection together with AIC, HQ and FPE."
            },
            "ro": {
                "title": "Criterii informaționale pentru numărul de laguri",
                "text": "Comparînd AIC și BIC pentru alegerea numărului de laguri al unui VAR, care afirmație este corectă?",
                "options": [
                    "BIC penalizează complexitatea mai sever decît AIC și tinde să aleagă modele mai mici",
                    "AIC penalizează complexitatea mai sever decît BIC și alege modele mai mici",
                    "AIC și BIC aleg întotdeauna același model",
                    "BIC nu se poate folosi pentru modele VAR"
                ],
                "correctExplanation": "Penalizarea pentru fiecare parametru este $2$ pentru AIC și $\\ln T$ pentru BIC; deoarece $\\ln T > 2$ pentru $T \\ge 8$, BIC favorizează modele mai parcimonioase (și este consistent), în timp ce AIC tinde să aleagă mai multe laguri.",
                "incorrectExplanation": "Ordinea penalizărilor este inversă, cele două criterii dau adesea rezultate diferite, iar BIC se raportează în mod obișnuit la alegerea numărului de laguri unui VAR, alături de AIC, HQ și FPE."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Estimating a VAR by OLS",
                "text": "Why is OLS applied equation by equation an efficient estimator of a VAR?",
                "options": [
                    "Because the errors are always normally distributed",
                    "Because all equations have the same regressors",
                    "Because the error covariance matrix of a VAR is always diagonal",
                    "Because the number of observations is always large"
                ],
                "correctExplanation": "When every equation contains the same regressors (a constant and the same lags of all variables), GLS on the system (SUR) reduces to OLS equation by equation (Zellner). Under Gaussian errors OLS also coincides with maximum likelihood.",
                "incorrectExplanation": "Normality is not required for this result, the errors of a VAR are typically correlated across equations (Σ is not diagonal), and the argument holds in any sample size. The key is that the regressors are identical across equations."
            },
            "ro": {
                "title": "Estimarea unui VAR prin OLS",
                "text": "De ce OLS aplicat ecuație cu ecuație este un estimator eficient al unui VAR?",
                "options": [
                    "Deoarece erorile au întotdeauna distribuția Normală",
                    "Deoarece toate ecuațiile au aceiași regresori",
                    "Deoarece matricea de covarianță a erorilor unui VAR este întotdeauna diagonală",
                    "Deoarece numărul de observații este întotdeauna mare"
                ],
                "correctExplanation": "Cînd fiecare ecuație conține aceiași regresori (termenul liber și aceleași laguri ale tuturor variabilelor), GLS pe sistem (SUR) se reduce la OLS ecuație cu ecuație (Zellner). Dacă erorile au distribuția Normală, OLS coincide și cu estimatorul de verosimilitate maximă.",
                "incorrectExplanation": "Rezultatul nu cere normalitate, erorile unui VAR sînt de regulă corelate între ecuații (Σ nu este diagonală), iar argumentul este valabil pentru orice volum al eșantionului. Esențial este că regresorii sînt aceiași în toate ecuațiile."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Portmanteau test",
                "text": "What does the multivariate portmanteau (Ljung-Box) test applied to the residuals of a VAR check?",
                "options": [
                    "Normality of the residuals",
                    "Conditional heteroskedasticity",
                    "Absence of serial correlation in the residuals",
                    "Stationarity of the original data"
                ],
                "correctExplanation": "The multivariate portmanteau statistic sums the squared residual autocorrelation matrices up to lag $h$; under $H_0$ of no residual autocorrelation it is approximately $\\chi^2$. It checks that the VAR has captured the serial dependence. Residuals may still be correlated across equations in the same period; this is captured by $\\boldsymbol{\\Sigma}$.",
                "incorrectExplanation": "Normality is checked with a (multivariate) Jarque-Bera test, conditional heteroskedasticity with an ARCH-LM test, and stationarity with unit root tests before estimation. The portmanteau test concerns residual autocorrelation."
            },
            "ro": {
                "title": "Testul portmanteau",
                "text": "Ce verifică testul portmanteau (Ljung-Box) multivariat aplicat reziduurilor unui VAR?",
                "options": [
                    "Normalitatea reziduurilor",
                    "Heteroscedasticitatea condiționată",
                    "Absența corelației seriale a reziduurilor",
                    "Staționaritatea datelor inițiale"
                ],
                "correctExplanation": "Statistica portmanteau multivariată însumează matricele de autocorelație ale reziduurilor pînă la lagul $h$; în ipoteza $H_0$ (fără autocorelație reziduală) are aproximativ distribuția $\\chi^2$. Testul verifică dacă VAR-ul a captat dependența serială. Reziduurile pot rămîne corelate între ecuații în aceeași perioadă; această corelație este descrisă de $\\boldsymbol{\\Sigma}$.",
                "incorrectExplanation": "Normalitatea se verifică printr-un test Jarque-Bera (multivariat), heteroscedasticitatea condiționată printr-un test ARCH-LM, iar staționaritatea prin teste de rădăcină unitară înainte de estimare. Testul portmanteau privește autocorelația reziduurilor."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Structural VAR",
                "text": "The main difference between a structural VAR (SVAR) and a reduced-form VAR is that:",
                "options": [
                    "The SVAR uses more lags",
                    "The SVAR cannot be used for forecasting",
                    "The SVAR requires more data",
                    "The SVAR identifies structural shocks that have an economic interpretation"
                ],
                "correctExplanation": "An SVAR imposes identifying restrictions to recover orthogonal structural shocks with an economic meaning (for example, a monetary policy shock) from the correlated reduced-form errors.",
                "incorrectExplanation": "The lag length, the data requirements and the forecasts are the same as for the underlying reduced form. What the SVAR adds is identification: a mapping from reduced-form errors to economically meaningful shocks."
            },
            "ro": {
                "title": "VAR structural",
                "text": "Principala diferență dintre un VAR structural (SVAR) și un VAR în formă redusă este că:",
                "options": [
                    "SVAR folosește mai multe laguri",
                    "SVAR nu poate fi folosit pentru prognoză",
                    "SVAR necesită mai multe date",
                    "SVAR identifică șocuri structurale care au o interpretare economică"
                ],
                "correctExplanation": "Un SVAR impune restricții de identificare pentru a obține, din erorile corelate ale formei reduse, șocuri structurale ortogonale cu semnificație economică (de exemplu, un șoc de politică monetară).",
                "incorrectExplanation": "Numărul de laguri, necesarul de date și prognozele sînt aceleași ca pentru forma redusă de la bază. Ceea ce adaugă SVAR este identificarea: o legătură între erorile formei reduse și șocuri cu semnificație economică."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Long-horizon VAR forecasts",
                "text": "For a stable VAR(1) $\\mathbf{Y}_t = \\mathbf{c} + \\mathbf{A}\\mathbf{Y}_{t-1} + \\boldsymbol{\\varepsilon}_t$, forecasts at long horizons converge to:",
                "options": [
                    "The unconditional mean $\\boldsymbol{\\mu} = (\\mathbf{I} - \\mathbf{A})^{-1}\\mathbf{c}$",
                    "The last observation $\\mathbf{Y}_T$",
                    "Zero",
                    "The constant vector $\\mathbf{c}$"
                ],
                "correctExplanation": "The unconditional mean solves $\\boldsymbol{\\mu} = \\mathbf{c} + \\mathbf{A}\\boldsymbol{\\mu}$, so $\\boldsymbol{\\mu} = (\\mathbf{I} - \\mathbf{A})^{-1}\\mathbf{c}$. Since $\\mathbf{Y}_{T+h|T} - \\boldsymbol{\\mu} = \\mathbf{A}^h(\\mathbf{Y}_T - \\boldsymbol{\\mu}) \\to \\mathbf{0}$, the information in the starting point fades and forecasts revert to $\\boldsymbol{\\mu}$.",
                "incorrectExplanation": "Zero is the limit only when $\\mathbf{c} = \\mathbf{0}$; the last observation is the long-run forecast of a random walk, not of a stable VAR; $\\mathbf{c}$ equals the mean only if $\\mathbf{A} = \\mathbf{0}$."
            },
            "ro": {
                "title": "Prognoze VAR pe orizonturi lungi",
                "text": "Pentru un VAR(1) stabil $\\mathbf{Y}_t = \\mathbf{c} + \\mathbf{A}\\mathbf{Y}_{t-1} + \\boldsymbol{\\varepsilon}_t$, prognozele pe orizonturi lungi converg către:",
                "options": [
                    "Media necondiționată $\\boldsymbol{\\mu} = (\\mathbf{I} - \\mathbf{A})^{-1}\\mathbf{c}$",
                    "Ultima observație $\\mathbf{Y}_T$",
                    "Zero",
                    "Vectorul de constante $\\mathbf{c}$"
                ],
                "correctExplanation": "Media necondiționată verifică $\\boldsymbol{\\mu} = \\mathbf{c} + \\mathbf{A}\\boldsymbol{\\mu}$, deci $\\boldsymbol{\\mu} = (\\mathbf{I} - \\mathbf{A})^{-1}\\mathbf{c}$. Deoarece $\\mathbf{Y}_{T+h|T} - \\boldsymbol{\\mu} = \\mathbf{A}^h(\\mathbf{Y}_T - \\boldsymbol{\\mu}) \\to \\mathbf{0}$, informația din punctul de plecare se estompează, iar prognozele revin la $\\boldsymbol{\\mu}$.",
                "incorrectExplanation": "Zero este limita doar cînd $\\mathbf{c} = \\mathbf{0}$; ultima observație este prognoza pe termen lung a unui mers aleator, nu a unui VAR stabil; $\\mathbf{c}$ este egal cu media doar dacă $\\mathbf{A} = \\mathbf{0}$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Interpreting a VAR(1) coefficient",
                "text": "In the VAR(1) model with $\\mathbf{A} = \\begin{pmatrix} 0.7 & 0.2 \\\\ -0.1 & 0.6 \\end{pmatrix}$, the coefficient $a_{21} = -0.1$ means that:",
                "options": [
                    "A one-unit increase in $Y_2$ at $t-1$ lowers $Y_1$ at $t$ by 0.1",
                    "A one-unit increase in $Y_1$ at $t-1$ lowers $Y_2$ at $t$ by 0.1, other things equal",
                    "A one-unit increase in $Y_1$ at $t-1$ raises $Y_2$ at $t$ by 0.1",
                    "A one-unit increase in $Y_2$ at $t-1$ raises $Y_2$ at $t$ by 0.1"
                ],
                "correctExplanation": "Element $(i, j)$ of $\\mathbf{A}$ is the effect of $Y_{j,t-1}$ on $Y_{i,t}$. So $a_{21} = -0.1$ is the effect of $Y_{1,t-1}$ on $Y_{2,t}$: a one-unit increase in $Y_1$ lowers $Y_2$ next period by 0.1.",
                "incorrectExplanation": "The row index is the equation (the variable being explained) and the column index the lagged regressor; reversing them gives the effect of $Y_2$ on $Y_1$, which is $a_{12} = 0.2$. The sign is negative, so the effect is a decrease."
            },
            "ro": {
                "title": "Interpretarea unui coeficient VAR(1)",
                "text": "În modelul VAR(1) cu $\\mathbf{A} = \\begin{pmatrix} 0{,}7 & 0{,}2 \\\\ -0{,}1 & 0{,}6 \\end{pmatrix}$, coeficientul $a_{21} = -0{,}1$ înseamnă că:",
                "options": [
                    "O creștere cu o unitate a lui $Y_2$ la $t-1$ scade $Y_1$ la $t$ cu 0,1",
                    "O creștere cu o unitate a lui $Y_1$ la $t-1$ scade $Y_2$ la $t$ cu 0,1, celelalte condiții rămînînd neschimbate",
                    "O creștere cu o unitate a lui $Y_1$ la $t-1$ crește $Y_2$ la $t$ cu 0,1",
                    "O creștere cu o unitate a lui $Y_2$ la $t-1$ crește $Y_2$ la $t$ cu 0,1"
                ],
                "correctExplanation": "Elementul $(i, j)$ al matricei $\\mathbf{A}$ este efectul lui $Y_{j,t-1}$ asupra lui $Y_{i,t}$. Așadar, $a_{21} = -0{,}1$ este efectul lui $Y_{1,t-1}$ asupra lui $Y_{2,t}$: o creștere cu o unitate a lui $Y_1$ scade $Y_2$ în perioada următoare cu 0,1.",
                "incorrectExplanation": "Indicele de linie desemnează ecuația (variabila explicată), iar indicele de coloană regresorul cu lag; inversarea lor dă efectul lui $Y_2$ asupra lui $Y_1$, adică $a_{12} = 0{,}2$. Semnul este negativ, deci efectul este o scădere."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Generalised impulse responses",
                "text": "What is the main property of the generalised impulse responses of Pesaran and Shin (1998)?",
                "options": [
                    "They require a structural model with $K(K-1)/2$ restrictions",
                    "They are computed from uncorrelated shocks, so the responses to different shocks add up",
                    "They do not depend on the ordering of the variables",
                    "They are always identical to the Cholesky responses of every variable"
                ],
                "correctExplanation": "A generalised response shocks one variable by one standard deviation and lets the other errors move as they typically do with it: $\\boldsymbol{\\Phi}_h\\boldsymbol{\\Sigma}\\mathbf{e}_k/\\sqrt{\\sigma_{kk}}$. No ordering is needed; for the variable ordered first it coincides with the Cholesky response.",
                "incorrectExplanation": "Generalised responses use the correlated reduced-form errors, so the responses to different shocks overlap and cannot be added up; they need no identifying restrictions; and they equal the Cholesky responses only for the variable placed first in the ordering."
            },
            "ro": {
                "title": "Răspunsuri la impuls generalizate",
                "text": "Care este principala proprietate a răspunsurilor la impuls generalizate ale lui Pesaran și Shin (1998)?",
                "options": [
                    "Cer un model structural cu $K(K-1)/2$ restricții",
                    "Sînt calculate din șocuri necorelate, deci răspunsurile la șocuri diferite se adună",
                    "Nu depind de ordinea variabilelor",
                    "Sînt întotdeauna identice cu răspunsurile Cholesky ale fiecărei variabile"
                ],
                "correctExplanation": "Un răspuns generalizat aplică unei variabile un șoc de o abatere standard și lasă celelalte erori să se miște așa cum se mișcă de obicei odată cu ea: $\\boldsymbol{\\Phi}_h\\boldsymbol{\\Sigma}\\mathbf{e}_k/\\sqrt{\\sigma_{kk}}$. Nu este nevoie de nicio ordine; pentru variabila așezată prima coincide cu răspunsul Cholesky.",
                "incorrectExplanation": "Răspunsurile generalizate folosesc erorile corelate ale formei reduse, deci răspunsurile la șocuri diferite se suprapun și nu se pot aduna; nu cer restricții de identificare; coincid cu răspunsurile Cholesky doar pentru variabila așezată prima."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Why the S&P 500 leads European markets",
                "text": "In daily data, yesterday's S&P 500 return Granger-causes today's BET and DAX returns, but not the reverse. The most plausible explanation is:",
                "options": [
                    "The VAR was estimated by OLS, which creates spurious lags",
                    "European investors react to news with a delay of several weeks",
                    "The S&P 500 is the structural cause of every movement of European prices",
                    "Wall Street closes after the European markets, so part of its daily return is news that Europe can price only the next day"
                ],
                "correctExplanation": "The S&P 500 closes at 22:00 Central European time, after Frankfurt and Bucharest. The American afternoon is reflected in the European prices of the next day. This is non-synchronous trading; with weekly returns the lead becomes much weaker.",
                "incorrectExplanation": "The effect lasts one day, not weeks; Granger causality is predictability, not structural causation; and OLS is the standard, consistent estimator of a VAR. The lead comes from the different closing times of the markets."
            },
            "ro": {
                "title": "Precedența S&P 500 față de piețele europene",
                "text": "În datele zilnice, randamentul S&P 500 de ieri cauzează în sens Granger randamentele BET și DAX de azi, dar nu invers. Explicația cea mai plauzibilă este:",
                "options": [
                    "VAR-ul a fost estimat prin OLS, care creează precedențe false",
                    "Investitorii europeni reacționează la știri cu o întîrziere de cîteva săptămîni",
                    "S&P 500 este cauza structurală a oricărei mișcări a prețurilor europene",
                    "Wall Street se închide după piețele europene, deci o parte din randamentul lui zilnic este o știre pe care Europa o poate include în prețuri abia a doua zi"
                ],
                "correctExplanation": "S&P 500 se închide la ora 22:00 (ora Europei Centrale), după Frankfurt și București. După-amiaza americană se reflectă în prețurile europene din ziua următoare. Este efectul tranzacționării nesincrone; cu randamente săptămînale precedența devine mult mai slabă.",
                "incorrectExplanation": "Efectul durează o zi, nu săptămîni; cauzalitatea Granger înseamnă predictibilitate, nu cauzalitate structurală; iar OLS este estimatorul standard și consistent al unui VAR. Precedența provine din orele diferite de închidere ale piețelor."
            }
        }
    ]
};
