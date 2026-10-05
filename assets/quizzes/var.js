// ============================================================
// Chapter 6 quiz bank: VAR models and Granger causality (EN + RO)
// 40 questions ported from the 2025/2026 site; 20 drawn per attempt.
// The bank grows to 24 questions when the chapter is rebuilt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['var'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Coefficient matrices in a VAR",
                "text": "In a VAR(p) model with $K$ variables, what is the dimension of each coefficient matrix $\\mathbf{A}_i$?",
                "options": [
                    "$K \\times 1$",
                    "$K \\times K$",
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
                    "$K \\times 1$",
                    "$K \\times K$",
                    "$K \\times p$",
                    "$Kp \\times Kp$"
                ],
                "correctExplanation": "Fiecare matrice $\\mathbf{A}_i$ transformă vectorul $\\mathbf{Y}_{t-i}$, de dimensiune $K$, într-o contribuție la vectorul $\\mathbf{Y}_t$, tot de dimensiune $K$; prin urmare, are dimensiunea $K \\times K$. Un VAR(p) are $p$ astfel de matrice, cîte una pentru fiecare lag, indiferent de $K$.",
                "incorrectExplanation": "$K \\times 1$ este dimensiunea vectorului de constante, iar $Kp \\times Kp$ este dimensiunea matricei companion; $K \\times p$ amestecă variabilele cu lag-urile. Fiecare matrice $\\mathbf{A}_i$ are dimensiunea $K \\times K$."
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
                "text": "Un VAR(p) cu $K$ variabile are o constantă în fiecare ecuație. Cîți coeficienți (constante și coeficienți ai lag-urilor, fără matricea de covarianță a erorilor) are în total?",
                "options": [
                    "$K^2 p$",
                    "$K(1 + Kp)$",
                    "$1 + Kp$",
                    "$K + p + K^2$"
                ],
                "correctExplanation": "Fiecare dintre cele $K$ ecuații are o constantă și $Kp$ coeficienți ai lag-urilor ($K$ variabile înmulțite cu $p$ lag-uri), deci totalul este $K(1 + Kp) = K + pK^2$.",
                "incorrectExplanation": "$K^2 p$ omite cele $K$ constante, $1 + Kp$ numără o singură ecuație, iar $K + p + K^2$ adună ordinul lag-ului în loc să înmulțească cu el. Totalul este $K(1 + Kp)$ și crește cu $K^2$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Curse of dimensionality",
                "text": "A VAR(4) model with $K = 5$ variables and a constant in each equation has how many coefficients in total?",
                "options": [
                    "20",
                    "100",
                    "105",
                    "125"
                ],
                "correctExplanation": "Each equation has $Kp + 1 = 5 \\times 4 + 1 = 21$ coefficients; with $K = 5$ equations the total is $5 \\times 21 = 105$.",
                "incorrectExplanation": "100 forgets the five constants, 20 counts only the lag coefficients of one equation and 125 is $5^3$. The total $K(Kp + 1) = 105$ shows how quickly a VAR becomes heavily parameterised."
            },
            "ro": {
                "title": "Blestemul dimensionalității",
                "text": "Un model VAR(4) cu $K = 5$ variabile și o constantă în fiecare ecuație are în total cîți coeficienți?",
                "options": [
                    "20",
                    "100",
                    "105",
                    "125"
                ],
                "correctExplanation": "Fiecare ecuație are $Kp + 1 = 5 \\times 4 + 1 = 21$ de coeficienți; cu $K = 5$ ecuații, totalul este $5 \\times 21 = 105$.",
                "incorrectExplanation": "100 omite cele cinci constante, 20 numără doar coeficienții lag-urilor dintr-o singură ecuație, iar 125 este $5^3$. Totalul $K(Kp + 1) = 105$ arată cît de repede crește numărul de parametri ai unui VAR."
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
            "correct": 2,
            "en": {
                "title": "Stability condition for a VAR(p)",
                "text": "For a general VAR(p), the stability condition requires that all roots of $\\det(\\mathbf{I}_K - \\mathbf{A}_1 z - \\cdots - \\mathbf{A}_p z^p) = 0$ lie:",
                "options": [
                    "Inside the unit circle",
                    "On the unit circle",
                    "Outside the unit circle",
                    "At the origin"
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
                    "În afara cercului unitate",
                    "În origine"
                ],
                "correctExplanation": "În forma cu polinom de lag-uri, rădăcinile trebuie să fie în afara cercului unitate, $|z| > 1$. Condiția este echivalentă cu aceea ca toate valorile proprii ale matricei companion să fie în interiorul cercului unitate, deoarece rădăcinile sînt inversele acestor valori proprii.",
                "incorrectExplanation": "„În interiorul cercului unitate” este condiția pentru valorile proprii ale matricei companion, nu pentru rădăcinile polinomului de lag-uri; rădăcini pe cercul unitate înseamnă rădăcini unitare (nestaționaritate). Pentru polinomul $\\det(\\mathbf{I}_K - \\mathbf{A}_1 z - \\cdots - \\mathbf{A}_p z^p)$, toate rădăcinile trebuie să verifice $|z| > 1$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Modulus of a complex eigenvalue",
                "text": "For the matrix $\\mathbf{A} = \\begin{pmatrix} 0.7 & 0.2 \\\\ -0.1 & 0.6 \\end{pmatrix}$ the eigenvalues are $\\lambda = 0.65 \\pm 0.132i$. The modulus $|\\lambda|$ equals:",
                "options": [
                    "0.782",
                    "0.663",
                    "1.30",
                    "0.44"
                ],
                "correctExplanation": "$|\\lambda| = \\sqrt{0.65^2 + 0.132^2} = \\sqrt{0.4225 + 0.0174} = \\sqrt{0.4399} \\approx 0.663$. Since $0.663 < 1$, the VAR(1) is stable and its impulse responses decay with damped oscillations.",
                "incorrectExplanation": "1.30 is the trace of $\\mathbf{A}$ (the sum of the eigenvalues) and 0.44 is its determinant (the product of the eigenvalues, equal to $|\\lambda|^2$ here); 0.782 is the sum $0.65 + 0.132$. The modulus of $a \\pm bi$ is $\\sqrt{a^2 + b^2} \\approx 0.663$."
            },
            "ro": {
                "title": "Modulul unei valori proprii complexe",
                "text": "Pentru matricea $\\mathbf{A} = \\begin{pmatrix} 0{,}7 & 0{,}2 \\\\ -0{,}1 & 0{,}6 \\end{pmatrix}$ valorile proprii sînt $\\lambda = 0{,}65 \\pm 0{,}132i$. Modulul $|\\lambda|$ este egal cu:",
                "options": [
                    "0,782",
                    "0,663",
                    "1,30",
                    "0,44"
                ],
                "correctExplanation": "$|\\lambda| = \\sqrt{0{,}65^2 + 0{,}132^2} = \\sqrt{0{,}4225 + 0{,}0174} = \\sqrt{0{,}4399} \\approx 0{,}663$. Deoarece $0{,}663 < 1$, modelul VAR(1) este stabil, iar funcțiile de răspuns la impuls se sting prin oscilații amortizate.",
                "incorrectExplanation": "1,30 este urma matricei $\\mathbf{A}$ (suma valorilor proprii), iar 0,44 este determinantul ei (produsul valorilor proprii, aici egal cu $|\\lambda|^2$); 0,782 este suma $0{,}65 + 0{,}132$. Modulul lui $a \\pm bi$ este $\\sqrt{a^2 + b^2} \\approx 0{,}663$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Purpose of the companion form",
                "text": "The companion form of a VAR(p) is useful because it:",
                "options": [
                    "Reduces the number of parameters to estimate",
                    "Rewrites any VAR(p) as a VAR(1), which simplifies the stability analysis, forecasting and the computation of impulse responses",
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
                    "Reduce numărul de parametri care trebuie estimați",
                    "Rescrie orice VAR(p) ca VAR(1), ceea ce simplifică analiza stabilității, prognoza și calculul funcțiilor de răspuns la impuls",
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
                "title": "Long-horizon impulse responses",
                "text": "For a stable VAR(1), the impulse response matrices $\\boldsymbol{\\Phi}_h$ converge, as $h \\to \\infty$, to:",
                "options": [
                    "The identity matrix $\\mathbf{I}_K$",
                    "The matrix $\\mathbf{A}$",
                    "The zero matrix $\\mathbf{0}$",
                    "The covariance matrix $\\boldsymbol{\\Sigma}$"
                ],
                "correctExplanation": "Since $\\boldsymbol{\\Phi}_h = \\mathbf{A}^h$ and all eigenvalues of $\\mathbf{A}$ have modulus below 1, $\\mathbf{A}^h \\to \\mathbf{0}$: shocks have only transitory effects.",
                "incorrectExplanation": "A limit equal to $\\mathbf{I}_K$, $\\mathbf{A}$ or $\\boldsymbol{\\Sigma}$ would mean that a one-off shock has a permanent effect, which happens only with unit roots. In a stable VAR every response dies out."
            },
            "ro": {
                "title": "Răspunsurile la impuls pe orizonturi lungi",
                "text": "Pentru un VAR(1) stabil, matricele de răspuns la impuls $\\boldsymbol{\\Phi}_h$ converg, cînd $h \\to \\infty$, către:",
                "options": [
                    "Matricea identitate $\\mathbf{I}_K$",
                    "Matricea $\\mathbf{A}$",
                    "Matricea zero $\\mathbf{0}$",
                    "Matricea de covarianță $\\boldsymbol{\\Sigma}$"
                ],
                "correctExplanation": "Deoarece $\\boldsymbol{\\Phi}_h = \\mathbf{A}^h$, iar toate valorile proprii ale lui $\\mathbf{A}$ au modulul sub 1, $\\mathbf{A}^h \\to \\mathbf{0}$: șocurile au doar efecte tranzitorii.",
                "incorrectExplanation": "O limită egală cu $\\mathbf{I}_K$, $\\mathbf{A}$ sau $\\boldsymbol{\\Sigma}$ ar însemna că un șoc singular are efect permanent, ceea ce se întîmplă doar în prezența rădăcinilor unitare. Într-un VAR stabil, orice răspuns se stinge."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Cumulative impulse responses",
                "text": "The long-run multiplier (the cumulative impulse response as $H \\to \\infty$) of a stable VAR(p) is:",
                "options": [
                    "$\\boldsymbol{\\Psi}_\\infty = \\mathbf{A}_1 + \\mathbf{A}_2 + \\cdots + \\mathbf{A}_p$",
                    "$\\boldsymbol{\\Psi}_\\infty = (\\mathbf{I}_K - \\mathbf{A}_1 - \\mathbf{A}_2 - \\cdots - \\mathbf{A}_p)^{-1}$",
                    "$\\boldsymbol{\\Psi}_\\infty = \\mathbf{I}_K$",
                    "$\\boldsymbol{\\Psi}_\\infty = \\mathbf{0}$"
                ],
                "correctExplanation": "Summing all $\\boldsymbol{\\Phi}_h$ gives $\\sum_{h \\ge 0} \\boldsymbol{\\Phi}_h = (\\mathbf{I}_K - \\mathbf{A}_1 - \\cdots - \\mathbf{A}_p)^{-1}$, the inverse of the lag polynomial evaluated at $z = 1$. The inverse exists because stability excludes a root at $z = 1$.",
                "incorrectExplanation": "The sum of the coefficient matrices is only the first-round effect; $\\mathbf{0}$ is the limit of the individual responses $\\boldsymbol{\\Phi}_h$, not of their sum, and $\\mathbf{I}_K$ is only the impact response. The cumulative effect is $(\\mathbf{I}_K - \\mathbf{A}_1 - \\cdots - \\mathbf{A}_p)^{-1}$."
            },
            "ro": {
                "title": "Răspunsuri la impuls cumulate",
                "text": "Multiplicatorul pe termen lung (răspunsul la impuls cumulat cînd $H \\to \\infty$) al unui VAR(p) stabil este:",
                "options": [
                    "$\\boldsymbol{\\Psi}_\\infty = \\mathbf{A}_1 + \\mathbf{A}_2 + \\cdots + \\mathbf{A}_p$",
                    "$\\boldsymbol{\\Psi}_\\infty = (\\mathbf{I}_K - \\mathbf{A}_1 - \\mathbf{A}_2 - \\cdots - \\mathbf{A}_p)^{-1}$",
                    "$\\boldsymbol{\\Psi}_\\infty = \\mathbf{I}_K$",
                    "$\\boldsymbol{\\Psi}_\\infty = \\mathbf{0}$"
                ],
                "correctExplanation": "Însumînd toate matricele $\\boldsymbol{\\Phi}_h$ se obține $\\sum_{h \\ge 0} \\boldsymbol{\\Phi}_h = (\\mathbf{I}_K - \\mathbf{A}_1 - \\cdots - \\mathbf{A}_p)^{-1}$, adică inversa polinomului de lag-uri evaluat în $z = 1$. Inversa există deoarece stabilitatea exclude o rădăcină în $z = 1$.",
                "incorrectExplanation": "Suma matricelor de coeficienți este doar efectul din prima rundă; $\\mathbf{0}$ este limita răspunsurilor individuale $\\boldsymbol{\\Phi}_h$, nu a sumei lor, iar $\\mathbf{I}_K$ este doar răspunsul din momentul impactului. Efectul cumulat este $(\\mathbf{I}_K - \\mathbf{A}_1 - \\cdots - \\mathbf{A}_p)^{-1}$."
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
            "correct": 1,
            "en": {
                "title": "Variable ordering in Cholesky identification",
                "text": "Why does the ordering of the variables matter when orthogonalised impulse responses are computed with the Cholesky decomposition?",
                "options": [
                    "Because OLS estimates depend on the order of the variables",
                    "Because the lower triangular matrix P forces the first variable not to respond contemporaneously to the shocks of the variables ordered after it",
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
                    "Deoarece estimațiile OLS depind de ordinea variabilelor",
                    "Deoarece matricea inferior triunghiulară P impune ca prima variabilă să nu răspundă contemporan la șocurile variabilelor așezate după ea",
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
                "title": "Bootstrap confidence intervals for impulse responses",
                "text": "The residual bootstrap for confidence intervals of impulse responses involves:",
                "options": [
                    "Estimating the model only once and using analytical formulas",
                    "Resampling the residuals with replacement, rebuilding the series, re-estimating the VAR and recomputing the impulse responses many times",
                    "Increasing the sample size by adding artificial data points to the original sample",
                    "Removing outliers from the data set and re-estimating the model"
                ],
                "correctExplanation": "Each bootstrap replication generates a new sample from the estimated VAR with resampled residuals, re-estimates the VAR and computes the impulse responses; the percentiles of the replicated responses give the confidence bands.",
                "incorrectExplanation": "Analytical (delta-method) intervals use a single estimation and are a different approach; the bootstrap does not enlarge the original sample and has nothing to do with removing outliers. Its essence is repeated resampling and re-estimation."
            },
            "ro": {
                "title": "Intervale de încredere bootstrap pentru răspunsurile la impuls",
                "text": "Metoda bootstrap pe reziduuri pentru intervalele de încredere ale răspunsurilor la impuls presupune:",
                "options": [
                    "Estimarea modelului o singură dată și folosirea unor formule analitice",
                    "Reeșantionarea reziduurilor cu întoarcere, reconstruirea seriilor, reestimarea VAR și recalcularea răspunsurilor la impuls de multe ori",
                    "Mărirea eșantionului prin adăugarea unor observații artificiale la eșantionul inițial",
                    "Eliminarea valorilor extreme din date și reestimarea modelului"
                ],
                "correctExplanation": "Fiecare replicare bootstrap generează un eșantion nou din VAR-ul estimat, cu reziduuri reeșantionate, reestimează VAR-ul și calculează răspunsurile la impuls; percentilele răspunsurilor replicate dau benzile de încredere.",
                "incorrectExplanation": "Intervalele analitice (metoda delta) folosesc o singură estimare și reprezintă o altă abordare; bootstrap-ul nu mărește eșantionul inițial și nu are legătură cu eliminarea valorilor extreme. Esența lui este reeșantionarea și reestimarea repetată."
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
            "correct": 1,
            "en": {
                "title": "FEVD across horizons",
                "text": "In a stable VAR, as the horizon $h$ of the forecast error variance decomposition increases:",
                "options": [
                    "Own shocks always dominate",
                    "The shares converge to their long-run values",
                    "All shocks contribute equally",
                    "The FEVD becomes undefined"
                ],
                "correctExplanation": "In a stable VAR the $h$-step forecast error variance converges to the unconditional variance, so the shares settle at long-run values that show the ultimate importance of each shock.",
                "incorrectExplanation": "Nothing forces own shocks to dominate or the shares to become equal; the long-run shares are determined by the model. The FEVD stays well defined because the forecast error variance converges to a finite limit."
            },
            "ro": {
                "title": "FEVD pe orizonturi diferite",
                "text": "Într-un VAR stabil, pe măsură ce orizontul $h$ al descompunerii varianței erorii de prognoză crește:",
                "options": [
                    "Șocurile proprii domină întotdeauna",
                    "Ponderile converg către valorile lor pe termen lung",
                    "Toate șocurile contribuie în mod egal",
                    "FEVD devine nedefinită"
                ],
                "correctExplanation": "Într-un VAR stabil, varianța erorii de prognoză la orizontul $h$ converge către varianța necondiționată, astfel încît ponderile se stabilizează la valori pe termen lung care arată importanța finală a fiecărui șoc.",
                "incorrectExplanation": "Nimic nu impune ca șocurile proprii să domine sau ca ponderile să devină egale; ponderile pe termen lung sînt determinate de model. FEVD rămîne bine definită deoarece varianța erorii de prognoză converge către o limită finită."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Properties of the FEVD",
                "text": "Which of the following is a property of the forecast error variance decomposition (FEVD) computed with the Cholesky decomposition?",
                "options": [
                    "The shares of all shocks for one variable can add up to more than 100%",
                    "At horizon $h = 1$, 100% of the forecast error variance of the first variable in the ordering is due to its own shock",
                    "The FEVD does not depend on the ordering of the variables",
                    "FEVD shares can be negative"
                ],
                "correctExplanation": "Because $\\mathbf{P}$ is lower triangular, the one-step forecast error of the first variable is driven only by its own orthogonalised shock, so its $h = 1$ share is 100%. Variables ordered later can already receive contributions from earlier shocks at $h = 1$.",
                "incorrectExplanation": "For each variable the shares are non-negative and add up to exactly 100%, and with Cholesky identification they generally change when the ordering changes. Note that only the first variable is guaranteed a 100% own share at $h = 1$; for the others the own shock need not even dominate."
            },
            "ro": {
                "title": "Proprietățile FEVD",
                "text": "Care dintre următoarele este o proprietate a descompunerii varianței erorii de prognoză (FEVD) calculate prin descompunerea Cholesky?",
                "options": [
                    "Ponderile tuturor șocurilor pentru o variabilă pot însuma mai mult de 100%",
                    "La orizontul $h = 1$, 100% din varianța erorii de prognoză a primei variabile din ordine se datorează propriului șoc",
                    "FEVD nu depinde de ordinea variabilelor",
                    "Ponderile FEVD pot fi negative"
                ],
                "correctExplanation": "Deoarece $\\mathbf{P}$ este inferior triunghiulară, eroarea de prognoză cu un pas a primei variabile este determinată doar de propriul șoc ortogonalizat, deci ponderea ei la $h = 1$ este 100%. Variabilele așezate mai tîrziu pot primi contribuții de la șocurile anterioare încă de la $h = 1$.",
                "incorrectExplanation": "Pentru fiecare variabilă, ponderile sînt nenegative și însumează exact 100%, iar în identificarea Cholesky ele se schimbă, în general, cînd se schimbă ordinea. Doar prima variabilă are garantată o pondere proprie de 100% la $h = 1$; pentru celelalte, propriul șoc nici măcar nu trebuie să domine."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Granger causality",
                "text": "What does the statement \"$X$ Granger-causes $Y$\" mean?",
                "options": [
                    "$X$ is the structural economic cause of $Y$",
                    "Past values of $X$ help forecast $Y$ beyond the information contained in the past of $Y$",
                    "$X$ and $Y$ are contemporaneously correlated",
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
                    "Valorile trecute ale lui $X$ ajută la prognoza lui $Y$, dincolo de informația din trecutul lui $Y$",
                    "$X$ și $Y$ sînt corelate contemporan",
                    "$X$ și $Y$ au aceeași tendință pe termen lung"
                ],
                "correctExplanation": "Cauzalitatea Granger privește conținutul predictiv suplimentar: valorile întîrziate ale lui $X$ reduc eroarea de prognoză a lui $Y$ după ce sînt folosite deja valorile întîrziate ale lui $Y$. Ea poate fi falsă (variabile omise, efecte de anticipare), deci nu dovedește o cauzalitate economică.",
                "incorrectExplanation": "Predictibilitatea nu înseamnă cauzalitate structurală; corelația contemporană ține de cauzalitatea instantanee, care nu implică lag-uri; o tendință comună pe termen lung înseamnă cointegrare. Cauzalitatea Granger privește lag-urile lui $X$ din ecuația lui $Y$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Granger causality test in a VAR",
                "text": "In a bivariate VAR(p), $Y_2$ does not Granger-cause $Y_1$ if:",
                "options": [
                    "All coefficients in the equation of $Y_2$ are zero",
                    "$a_{12}^{(1)} = a_{12}^{(2)} = \\cdots = a_{12}^{(p)} = 0$",
                    "The contemporaneous error covariance $\\sigma_{12} = 0$",
                    "The eigenvalues of the matrix $\\mathbf{A}$ are zero"
                ],
                "correctExplanation": "$a_{12}^{(i)}$ is the coefficient of $Y_{2,t-i}$ in the equation of $Y_1$. Non-causality is the joint null $H_0\\colon a_{12}^{(1)} = \\cdots = a_{12}^{(p)} = 0$, tested with a Wald or $F$ test.",
                "incorrectExplanation": "The equation of $Y_2$ is where one tests whether $Y_1$ Granger-causes $Y_2$; $\\sigma_{12} = 0$ concerns instantaneous causality; eigenvalues concern stability. Only the lags of $Y_2$ in the equation of $Y_1$ matter here."
            },
            "ro": {
                "title": "Testul Granger într-un VAR",
                "text": "Într-un VAR(p) bivariat, $Y_2$ nu cauzează Granger pe $Y_1$ dacă:",
                "options": [
                    "Toți coeficienții din ecuația lui $Y_2$ sînt zero",
                    "$a_{12}^{(1)} = a_{12}^{(2)} = \\cdots = a_{12}^{(p)} = 0$",
                    "Covarianța contemporană a erorilor $\\sigma_{12} = 0$",
                    "Valorile proprii ale matricei $\\mathbf{A}$ sînt nule"
                ],
                "correctExplanation": "$a_{12}^{(i)}$ este coeficientul lui $Y_{2,t-i}$ din ecuația lui $Y_1$. Lipsa cauzalității este ipoteza nulă comună $H_0\\colon a_{12}^{(1)} = \\cdots = a_{12}^{(p)} = 0$, testată cu un test Wald sau $F$.",
                "incorrectExplanation": "În ecuația lui $Y_2$ se testează dacă $Y_1$ cauzează Granger pe $Y_2$; $\\sigma_{12} = 0$ privește cauzalitatea instantanee; valorile proprii privesc stabilitatea. Aici contează doar lag-urile lui $Y_2$ din ecuația lui $Y_1$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Granger causality test: numerical example",
                "text": "A bivariate VAR(2) is estimated on $T = 100$ observations. In the equation of $Y_1$, the unrestricted residual sum of squares is $RSS_U = 45.2$ and the restricted one (without the lags of $Y_2$) is $RSS_R = 52.8$. The $F$ statistic of the Granger causality test is approximately:",
                "options": [
                    "3.09",
                    "5.42",
                    "7.98",
                    "12.35"
                ],
                "correctExplanation": "$F = \\dfrac{(RSS_R - RSS_U)/p}{RSS_U/(T - 2p - 1)} = \\dfrac{(52.8 - 45.2)/2}{45.2/95} = \\dfrac{3.8}{0.4758} \\approx 7.98$, with $(2, 95)$ degrees of freedom; at the 5% level $H_0$ is rejected.",
                "incorrectExplanation": "The other values come from wrong degrees of freedom or from dividing by the wrong sum of squares. The numerator has $p = 2$ restrictions and the denominator $T - 2p - 1 = 95$ degrees of freedom (five coefficients in the unrestricted equation)."
            },
            "ro": {
                "title": "Testul de cauzalitate Granger: exemplu numeric",
                "text": "Un VAR(2) bivariat este estimat pe $T = 100$ de observații. În ecuația lui $Y_1$, suma pătratelor reziduurilor modelului nerestricționat este $RSS_U = 45{,}2$, iar a celui restricționat (fără lag-urile lui $Y_2$) este $RSS_R = 52{,}8$. Statistica $F$ a testului de cauzalitate Granger este aproximativ:",
                "options": [
                    "3,09",
                    "5,42",
                    "7,98",
                    "12,35"
                ],
                "correctExplanation": "$F = \\dfrac{(RSS_R - RSS_U)/p}{RSS_U/(T - 2p - 1)} = \\dfrac{(52{,}8 - 45{,}2)/2}{45{,}2/95} = \\dfrac{3{,}8}{0{,}4758} \\approx 7{,}98$, cu $(2, 95)$ grade de libertate; la pragul de 5%, $H_0$ se respinge.",
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
                "correctExplanation": "Dacă $Z$ îl influențează pe $X$ înaintea lui $Y$ și nu este inclusă în sistem, valorile întîrziate ale lui $X$ țin locul lui $Z$ și par să prognozeze $Y$, deși $X$ nu are niciun efect asupra lui $Y$.",
                "incorrectExplanation": "Adăugarea unor variabile relevante reduce, nu creează, problema variabilelor omise; estimarea OLS ecuație cu ecuație este estimatorul standard al unui VAR; testele $F$ și Wald sînt asimptotic echivalente și testează aceleași restricții."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Bidirectional causality",
                "text": "If both \"$X$ Granger-causes $Y$\" and \"$Y$ Granger-causes $X$\" hold, the relationship is called:",
                "options": [
                    "No causality",
                    "Unidirectional causality",
                    "Bidirectional (feedback) causality",
                    "Instantaneous causality"
                ],
                "correctExplanation": "Bidirectional or feedback causality means that each variable helps predict the other; it is common among financial and macroeconomic series.",
                "incorrectExplanation": "Unidirectional causality runs only one way, and instantaneous causality concerns same-period correlation of the errors, not lags. When causality runs both ways, it is called bidirectional or feedback causality."
            },
            "ro": {
                "title": "Cauzalitate bidirecțională",
                "text": "Dacă sînt adevărate atît „$X$ cauzează Granger pe $Y$”, cît și „$Y$ cauzează Granger pe $X$”, relația se numește:",
                "options": [
                    "Lipsă de cauzalitate",
                    "Cauzalitate unidirecțională",
                    "Cauzalitate bidirecțională (cu feedback)",
                    "Cauzalitate instantanee"
                ],
                "correctExplanation": "Cauzalitatea bidirecțională (cu feedback) înseamnă că fiecare variabilă ajută la prognoza celeilalte; este frecventă la seriile financiare și macroeconomice.",
                "incorrectExplanation": "Cauzalitatea unidirecțională acționează într-un singur sens, iar cauzalitatea instantanee privește corelația erorilor din aceeași perioadă, nu lag-urile. Cînd cauzalitatea acționează în ambele sensuri, ea se numește bidirecțională sau cu feedback."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Instantaneous causality",
                "text": "Instantaneous causality between two variables of a VAR is tested by checking whether:",
                "options": [
                    "Lagged $X$ helps predict $Y$",
                    "The reduced-form errors are correlated within the same period, $\\sigma_{12} \\neq 0$",
                    "$X$ and $Y$ share a common stochastic trend",
                    "The VAR is stable"
                ],
                "correctExplanation": "Instantaneous causality concerns the same-period correlation of the errors, $H_0\\colon \\sigma_{12} = \\operatorname{Cov}(\\varepsilon_{1t}, \\varepsilon_{2t}) = 0$. It is symmetric and has no direction.",
                "incorrectExplanation": "Lagged predictive content is Granger causality; a common stochastic trend is cointegration; stability is an eigenvalue condition. Instantaneous causality involves no lags at all."
            },
            "ro": {
                "title": "Cauzalitate instantanee",
                "text": "Cauzalitatea instantanee dintre două variabile ale unui VAR se testează verificînd dacă:",
                "options": [
                    "Valorile întîrziate ale lui $X$ ajută la prognoza lui $Y$",
                    "Erorile din forma redusă sînt corelate în aceeași perioadă, $\\sigma_{12} \\neq 0$",
                    "$X$ și $Y$ au o tendință stochastică comună",
                    "VAR-ul este stabil"
                ],
                "correctExplanation": "Cauzalitatea instantanee privește corelația erorilor din aceeași perioadă, $H_0\\colon \\sigma_{12} = \\operatorname{Cov}(\\varepsilon_{1t}, \\varepsilon_{2t}) = 0$. Ea este simetrică și nu are o direcție.",
                "incorrectExplanation": "Conținutul predictiv al lag-urilor ține de cauzalitatea Granger; o tendință stochastică comună înseamnă cointegrare; stabilitatea este o condiție asupra valorilor proprii. Cauzalitatea instantanee nu implică lag-uri."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Toda-Yamamoto procedure",
                "text": "What is the main purpose of the Toda-Yamamoto procedure for testing Granger causality?",
                "options": [
                    "To improve the power of the test in small samples",
                    "To avoid estimating a VAR altogether",
                    "To allow valid Granger causality tests when the data are nonstationary or cointegrated",
                    "To reduce the number of estimated parameters"
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
                    "Testarea validă a cauzalității Granger atunci cînd datele sînt nestaționare sau cointegrate",
                    "Reducerea numărului de parametri estimați"
                ],
                "correctExplanation": "Cînd variabilele sînt integrate sau cointegrate, testul Wald obișnuit într-un VAR în niveluri are o distribuție nestandard. Toda și Yamamoto (1995) estimează un VAR($p + d_{max}$) în niveluri, unde $d_{max}$ este ordinul maxim de integrare, și testează doar primele $p$ lag-uri, ceea ce restabilește distribuția asimptotică $\\chi^2$.",
                "incorrectExplanation": "Procedura estimează în continuare un VAR, adaugă parametri (cele $d_{max}$ lag-uri suplimentare) în loc să îi elimine și reduce ușor puterea testului. Scopul ei este validitatea testului Wald indiferent de rădăcinile unitare sau de cointegrare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Information criteria for the lag order",
                "text": "Comparing AIC and BIC for selecting the lag order of a VAR, which statement is correct?",
                "options": [
                    "AIC penalises complexity more heavily than BIC and selects smaller models",
                    "BIC penalises complexity more heavily than AIC and tends to select smaller models",
                    "AIC and BIC always select the same model",
                    "BIC cannot be used for VAR models"
                ],
                "correctExplanation": "The penalty per parameter is $2$ for AIC and $\\ln T$ for BIC; since $\\ln T > 2$ for $T \\ge 8$, BIC favours more parsimonious models (and is consistent), while AIC tends to choose longer lags.",
                "incorrectExplanation": "The ordering of the penalties is the other way round, the two criteria often disagree, and BIC is routinely reported for VAR lag selection together with AIC, HQ and FPE."
            },
            "ro": {
                "title": "Criterii informaționale pentru ordinul lag-ului",
                "text": "Comparînd AIC și BIC pentru alegerea ordinului lag-ului unui VAR, care afirmație este corectă?",
                "options": [
                    "AIC penalizează complexitatea mai sever decît BIC și alege modele mai mici",
                    "BIC penalizează complexitatea mai sever decît AIC și tinde să aleagă modele mai mici",
                    "AIC și BIC aleg întotdeauna același model",
                    "BIC nu se poate folosi pentru modele VAR"
                ],
                "correctExplanation": "Penalizarea pentru fiecare parametru este $2$ pentru AIC și $\\ln T$ pentru BIC; deoarece $\\ln T > 2$ pentru $T \\ge 8$, BIC favorizează modele mai parcimonioase (și este consistent), în timp ce AIC tinde să aleagă mai multe lag-uri.",
                "incorrectExplanation": "Ordinea penalizărilor este inversă, cele două criterii dau adesea rezultate diferite, iar BIC se raportează în mod obișnuit la alegerea lag-ului unui VAR, alături de AIC, HQ și FPE."
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
                "correctExplanation": "Cînd fiecare ecuație conține aceiași regresori (termenul liber și aceleași lag-uri ale tuturor variabilelor), GLS pe sistem (SUR) se reduce la OLS ecuație cu ecuație (Zellner). Dacă erorile au distribuția Normală, OLS coincide și cu estimatorul de verosimilitate maximă.",
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
                "correctExplanation": "Statistica portmanteau multivariată însumează matricele de autocorelație ale reziduurilor pînă la lag-ul $h$; sub $H_0$ (fără autocorelație reziduală) are aproximativ distribuția $\\chi^2$. Testul verifică dacă VAR-ul a captat dependența serială. Reziduurile pot rămîne corelate între ecuații în aceeași perioadă; această corelație este descrisă de $\\boldsymbol{\\Sigma}$.",
                "incorrectExplanation": "Normalitatea se verifică printr-un test Jarque-Bera (multivariat), heteroscedasticitatea condiționată printr-un test ARCH-LM, iar staționaritatea prin teste de rădăcină unitară înainte de estimare. Testul portmanteau privește autocorelația reziduurilor."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Rejected portmanteau test",
                "text": "The portmanteau test rejects the null hypothesis for the residuals of a VAR. The most appropriate action is to:",
                "options": [
                    "Apply a logarithmic transformation to the data",
                    "Increase the lag order $p$ or add variables to the system",
                    "Switch to univariate ARIMA models",
                    "Remove the constant term from the model"
                ],
                "correctExplanation": "Rejection means that residual autocorrelation remains, i.e. the model has not captured all the dynamics; more lags or omitted variables are the natural remedies.",
                "incorrectExplanation": "A log transformation addresses scale or variance, not leftover autocorrelation; univariate models discard the cross-dynamics; the constant has no effect on residual autocorrelation. The dynamics must be enriched."
            },
            "ro": {
                "title": "Respingerea testului portmanteau",
                "text": "Testul portmanteau respinge ipoteza nulă pentru reziduurile unui VAR. Cea mai potrivită măsură este:",
                "options": [
                    "Aplicarea unei transformări logaritmice asupra datelor",
                    "Creșterea ordinului $p$ al lag-ului sau adăugarea unor variabile în sistem",
                    "Trecerea la modele ARIMA univariate",
                    "Eliminarea termenului liber din model"
                ],
                "correctExplanation": "Respingerea arată că a rămas autocorelație în reziduuri, adică modelul nu a captat întreaga dinamică; remediile firești sînt mai multe lag-uri sau variabilele omise.",
                "incorrectExplanation": "Transformarea logaritmică privește scala sau varianța, nu autocorelația rămasă; modelele univariate renunță la dinamica încrucișată; termenul liber nu influențează autocorelația reziduurilor. Dinamica modelului trebuie îmbogățită."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Checks before estimating a VAR",
                "text": "Before estimating a VAR in levels, you should always check:",
                "options": [
                    "That all variables are I(2)",
                    "The order of integration (stationarity) of each variable",
                    "That the variables are perfectly correlated",
                    "That the sample size is exactly 100"
                ],
                "correctExplanation": "A standard VAR assumes stationary variables. Unit root tests (ADF, KPSS) come first; with I(1) variables one tests for cointegration and uses a VECM if it is present, or a VAR in differences if it is not.",
                "incorrectExplanation": "I(2) variables are a problem, not a requirement; perfect correlation would make the regressors collinear; there is no required sample size. The first step is to establish the order of integration."
            },
            "ro": {
                "title": "Verificări înaintea estimării unui VAR",
                "text": "Înainte de estimarea unui VAR în niveluri, trebuie verificat întotdeauna:",
                "options": [
                    "Că toate variabilele sînt I(2)",
                    "Ordinul de integrare (staționaritatea) fiecărei variabile",
                    "Că variabilele sînt perfect corelate",
                    "Că volumul eșantionului este exact 100"
                ],
                "correctExplanation": "Un VAR standard presupune variabile staționare. Primul pas îl constituie testele de rădăcină unitară (ADF, KPSS); pentru variabile I(1) se testează cointegrarea și se folosește un VECM dacă aceasta există, respectiv un VAR în diferențe dacă nu există.",
                "incorrectExplanation": "Variabilele I(2) reprezintă o problemă, nu o cerință; corelația perfectă ar face regresorii coliniari; nu există un volum de eșantion impus. Primul pas este stabilirea ordinului de integrare."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Cointegrated variables in a VAR",
                "text": "If the variables are I(1) and cointegrated, the appropriate model is:",
                "options": [
                    "A VAR in levels with unrestricted standard inference",
                    "A VAR in first differences",
                    "A vector error correction model (VECM)",
                    "Separate univariate ARIMA models"
                ],
                "correctExplanation": "The VECM combines the short-run dynamics in differences with the long-run equilibrium relationship through the error correction term.",
                "incorrectExplanation": "A VAR in differences discards the long-run relationship (it is misspecified under cointegration); standard inference in a levels VAR is unreliable for some hypotheses; univariate models ignore the joint dynamics. The VECM keeps both horizons."
            },
            "ro": {
                "title": "Variabile cointegrate într-un VAR",
                "text": "Dacă variabilele sînt I(1) și cointegrate, modelul potrivit este:",
                "options": [
                    "Un VAR în niveluri, cu inferență standard nerestricționată",
                    "Un VAR în diferențe de ordinul întîi",
                    "Un model vectorial cu corecția erorilor (VECM)",
                    "Modele ARIMA univariate separate"
                ],
                "correctExplanation": "VECM combină dinamica pe termen scurt, în diferențe, cu relația de echilibru pe termen lung, prin termenul de corecție a erorilor.",
                "incorrectExplanation": "Un VAR în diferențe pierde relația pe termen lung (este greșit specificat cînd există cointegrare); inferența standard într-un VAR în niveluri nu este de încredere pentru unele ipoteze; modelele univariate ignoră dinamica comună. VECM păstrează ambele orizonturi."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Structural VAR",
                "text": "The main difference between a structural VAR (SVAR) and a reduced-form VAR is that:",
                "options": [
                    "The SVAR uses more lags",
                    "The SVAR identifies structural shocks that have an economic interpretation",
                    "The SVAR requires more data",
                    "The SVAR cannot be used for forecasting"
                ],
                "correctExplanation": "An SVAR imposes identifying restrictions to recover orthogonal structural shocks with an economic meaning (for example, a monetary policy shock) from the correlated reduced-form errors.",
                "incorrectExplanation": "The lag length, the data requirements and the forecasts are the same as for the underlying reduced form. What the SVAR adds is identification: a mapping from reduced-form errors to economically meaningful shocks."
            },
            "ro": {
                "title": "VAR structural",
                "text": "Principala diferență dintre un VAR structural (SVAR) și un VAR în formă redusă este că:",
                "options": [
                    "SVAR folosește mai multe lag-uri",
                    "SVAR identifică șocuri structurale care au o interpretare economică",
                    "SVAR necesită mai multe date",
                    "SVAR nu poate fi folosit pentru prognoză"
                ],
                "correctExplanation": "Un SVAR impune restricții de identificare pentru a obține, din erorile corelate ale formei reduse, șocuri structurale ortogonale cu semnificație economică (de exemplu, un șoc de politică monetară).",
                "incorrectExplanation": "Numărul de lag-uri, necesarul de date și prognozele sînt aceleași ca pentru forma redusă de la bază. Ceea ce adaugă SVAR este identificarea: o legătură între erorile formei reduse și șocuri cu semnificație economică."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Identification schemes for an SVAR",
                "text": "Which of the following is NOT an identification scheme for structural VAR models?",
                "options": [
                    "Short-run (Cholesky) restrictions",
                    "Long-run (Blanchard-Quah) restrictions",
                    "Minimising the OLS residual sum of squares",
                    "Sign restrictions"
                ],
                "correctExplanation": "Minimising the residual sum of squares is how the reduced form is estimated; it says nothing about which structural shocks lie behind the errors. Identification needs extra restrictions: short-run zeros, long-run restrictions or sign restrictions.",
                "incorrectExplanation": "Cholesky (recursive short-run zeros), Blanchard-Quah (zero long-run effects) and sign restrictions (Uhlig) are all standard schemes. OLS only delivers the reduced form."
            },
            "ro": {
                "title": "Scheme de identificare pentru SVAR",
                "text": "Care dintre următoarele NU este o schemă de identificare pentru modelele VAR structurale?",
                "options": [
                    "Restricții pe termen scurt (Cholesky)",
                    "Restricții pe termen lung (Blanchard-Quah)",
                    "Minimizarea sumei pătratelor reziduurilor OLS",
                    "Restricții de semn"
                ],
                "correctExplanation": "Minimizarea sumei pătratelor reziduurilor este modul în care se estimează forma redusă; ea nu spune nimic despre șocurile structurale din spatele erorilor. Identificarea cere restricții suplimentare: zerouri pe termen scurt, restricții pe termen lung sau restricții de semn.",
                "incorrectExplanation": "Cholesky (zerouri recursive pe termen scurt), Blanchard-Quah (efecte nule pe termen lung) și restricțiile de semn (Uhlig) sînt toate scheme standard. OLS furnizează doar forma redusă."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Identification problem in an SVAR",
                "text": "In an SVAR with $K$ variables, how many additional restrictions are needed for exact identification, beyond the information in $\\boldsymbol{\\Sigma}$?",
                "options": [
                    "$K$",
                    "$K^2$",
                    "$K(K-1)/2$",
                    "$K(K+1)/2$"
                ],
                "correctExplanation": "The impact matrix has $K^2$ unknown elements, while the symmetric $\\boldsymbol{\\Sigma}$ provides only $K(K+1)/2$ distinct moments; the gap is $K^2 - K(K+1)/2 = K(K-1)/2$ restrictions (for example, the zeros above the diagonal in Cholesky).",
                "incorrectExplanation": "$K(K+1)/2$ is the number of moments supplied by $\\boldsymbol{\\Sigma}$ and $K^2$ the number of unknowns; $K$ is too few for $K > 3$. The difference $K(K-1)/2$ is the number of extra restrictions."
            },
            "ro": {
                "title": "Problema identificării într-un SVAR",
                "text": "Într-un SVAR cu $K$ variabile, cîte restricții suplimentare sînt necesare pentru identificarea exactă, dincolo de informația din $\\boldsymbol{\\Sigma}$?",
                "options": [
                    "$K$",
                    "$K^2$",
                    "$K(K-1)/2$",
                    "$K(K+1)/2$"
                ],
                "correctExplanation": "Matricea de impact are $K^2$ elemente necunoscute, iar matricea simetrică $\\boldsymbol{\\Sigma}$ furnizează doar $K(K+1)/2$ momente distincte; diferența este de $K^2 - K(K+1)/2 = K(K-1)/2$ restricții (de exemplu, zerourile de deasupra diagonalei în Cholesky).",
                "incorrectExplanation": "$K(K+1)/2$ este numărul de momente furnizate de $\\boldsymbol{\\Sigma}$, iar $K^2$ numărul de necunoscute; $K$ este prea puțin pentru $K > 3$. Diferența $K(K-1)/2$ reprezintă numărul de restricții suplimentare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Autocovariance matrices",
                "text": "For a weakly stationary multivariate time series, the autocovariance matrices $\\boldsymbol{\\Gamma}(h) = \\operatorname{Cov}(\\mathbf{Y}_{t+h}, \\mathbf{Y}_t)$ satisfy:",
                "options": [
                    "$\\boldsymbol{\\Gamma}(-h) = \\boldsymbol{\\Gamma}(h)$",
                    "$\\boldsymbol{\\Gamma}(-h) = \\boldsymbol{\\Gamma}(h)^\\prime$ (transpose)",
                    "$\\boldsymbol{\\Gamma}(-h) = -\\boldsymbol{\\Gamma}(h)$",
                    "$\\boldsymbol{\\Gamma}(-h) = \\boldsymbol{\\Gamma}(h)^{-1}$"
                ],
                "correctExplanation": "Element by element, $\\gamma_{ij}(h) = \\gamma_{ji}(-h)$: the covariance between $Y_i$ today and $Y_j$ $h$ periods earlier is not the same as with $Y_j$ $h$ periods later. Hence $\\boldsymbol{\\Gamma}(-h) = \\boldsymbol{\\Gamma}(h)^\\prime$, and in general $\\boldsymbol{\\Gamma}(-h) \\neq \\boldsymbol{\\Gamma}(h)$.",
                "incorrectExplanation": "The univariate symmetry $\\gamma(-h) = \\gamma(h)$ does not carry over to cross-covariances, because leads and lags of different variables differ (one variable may lead the other). The sign flip and the inverse have no basis."
            },
            "ro": {
                "title": "Matricele de autocovarianță",
                "text": "Pentru o serie de timp multivariată slab staționară, matricele de autocovarianță $\\boldsymbol{\\Gamma}(h) = \\operatorname{Cov}(\\mathbf{Y}_{t+h}, \\mathbf{Y}_t)$ verifică:",
                "options": [
                    "$\\boldsymbol{\\Gamma}(-h) = \\boldsymbol{\\Gamma}(h)$",
                    "$\\boldsymbol{\\Gamma}(-h) = \\boldsymbol{\\Gamma}(h)^\\prime$ (transpusa)",
                    "$\\boldsymbol{\\Gamma}(-h) = -\\boldsymbol{\\Gamma}(h)$",
                    "$\\boldsymbol{\\Gamma}(-h) = \\boldsymbol{\\Gamma}(h)^{-1}$"
                ],
                "correctExplanation": "Element cu element, $\\gamma_{ij}(h) = \\gamma_{ji}(-h)$: covarianța dintre $Y_i$ de azi și $Y_j$ de acum $h$ perioade diferă de covarianța cu $Y_j$ de peste $h$ perioade. Prin urmare, $\\boldsymbol{\\Gamma}(-h) = \\boldsymbol{\\Gamma}(h)^\\prime$, iar în general $\\boldsymbol{\\Gamma}(-h) \\neq \\boldsymbol{\\Gamma}(h)$.",
                "incorrectExplanation": "Simetria univariată $\\gamma(-h) = \\gamma(h)$ nu se păstrează pentru covarianțele încrucișate, deoarece avansurile și întîrzierile unor variabile diferite nu coincid (o variabilă o poate devansa pe cealaltă). Schimbarea semnului și inversa nu au nicio justificare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Lyapunov equation",
                "text": "The variance-covariance matrix $\\boldsymbol{\\Gamma}(0)$ of a stationary VAR(1) $\\mathbf{Y}_t = \\mathbf{c} + \\mathbf{A}\\mathbf{Y}_{t-1} + \\boldsymbol{\\varepsilon}_t$ satisfies:",
                "options": [
                    "$\\boldsymbol{\\Gamma}(0) = \\mathbf{A} + \\boldsymbol{\\Sigma}$",
                    "$\\boldsymbol{\\Gamma}(0) = \\mathbf{A}\\boldsymbol{\\Gamma}(0)\\mathbf{A}^\\prime + \\boldsymbol{\\Sigma}$",
                    "$\\boldsymbol{\\Gamma}(0) = \\mathbf{A}\\boldsymbol{\\Sigma}\\mathbf{A}^\\prime$",
                    "$\\boldsymbol{\\Gamma}(0) = (\\mathbf{I} - \\mathbf{A})^{-1}\\boldsymbol{\\Sigma}$"
                ],
                "correctExplanation": "Taking variances on both sides, with $\\boldsymbol{\\varepsilon}_t$ uncorrelated with $\\mathbf{Y}_{t-1}$: $\\operatorname{Var}(\\mathbf{Y}_t) = \\mathbf{A}\\operatorname{Var}(\\mathbf{Y}_{t-1})\\mathbf{A}^\\prime + \\boldsymbol{\\Sigma}$; stationarity gives the discrete Lyapunov equation $\\boldsymbol{\\Gamma}(0) = \\mathbf{A}\\boldsymbol{\\Gamma}(0)\\mathbf{A}^\\prime + \\boldsymbol{\\Sigma}$, solved by $\\operatorname{vec}\\boldsymbol{\\Gamma}(0) = (\\mathbf{I} - \\mathbf{A} \\otimes \\mathbf{A})^{-1}\\operatorname{vec}\\boldsymbol{\\Sigma}$.",
                "incorrectExplanation": "Variances transform as $\\mathbf{A}(\\cdot)\\mathbf{A}^\\prime$, not additively in $\\mathbf{A}$; $\\mathbf{A}\\boldsymbol{\\Sigma}\\mathbf{A}^\\prime$ omits the current shock and the recursion; $(\\mathbf{I} - \\mathbf{A})^{-1}$ belongs to the mean, not to the variance."
            },
            "ro": {
                "title": "Ecuația Lyapunov",
                "text": "Matricea de varianță-covarianță $\\boldsymbol{\\Gamma}(0)$ a unui VAR(1) staționar $\\mathbf{Y}_t = \\mathbf{c} + \\mathbf{A}\\mathbf{Y}_{t-1} + \\boldsymbol{\\varepsilon}_t$ verifică:",
                "options": [
                    "$\\boldsymbol{\\Gamma}(0) = \\mathbf{A} + \\boldsymbol{\\Sigma}$",
                    "$\\boldsymbol{\\Gamma}(0) = \\mathbf{A}\\boldsymbol{\\Gamma}(0)\\mathbf{A}^\\prime + \\boldsymbol{\\Sigma}$",
                    "$\\boldsymbol{\\Gamma}(0) = \\mathbf{A}\\boldsymbol{\\Sigma}\\mathbf{A}^\\prime$",
                    "$\\boldsymbol{\\Gamma}(0) = (\\mathbf{I} - \\mathbf{A})^{-1}\\boldsymbol{\\Sigma}$"
                ],
                "correctExplanation": "Aplicînd varianța ambilor membri, cu $\\boldsymbol{\\varepsilon}_t$ necorelat cu $\\mathbf{Y}_{t-1}$: $\\operatorname{Var}(\\mathbf{Y}_t) = \\mathbf{A}\\operatorname{Var}(\\mathbf{Y}_{t-1})\\mathbf{A}^\\prime + \\boldsymbol{\\Sigma}$; din staționaritate rezultă ecuația Lyapunov discretă $\\boldsymbol{\\Gamma}(0) = \\mathbf{A}\\boldsymbol{\\Gamma}(0)\\mathbf{A}^\\prime + \\boldsymbol{\\Sigma}$, cu soluția $\\operatorname{vec}\\boldsymbol{\\Gamma}(0) = (\\mathbf{I} - \\mathbf{A} \\otimes \\mathbf{A})^{-1}\\operatorname{vec}\\boldsymbol{\\Sigma}$.",
                "incorrectExplanation": "Varianțele se transformă după regula $\\mathbf{A}(\\cdot)\\mathbf{A}^\\prime$, nu aditiv în $\\mathbf{A}$; $\\mathbf{A}\\boldsymbol{\\Sigma}\\mathbf{A}^\\prime$ omite șocul curent și recursivitatea; $(\\mathbf{I} - \\mathbf{A})^{-1}$ ține de medie, nu de varianță."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Long-horizon VAR forecasts",
                "text": "For a stable VAR(1) $\\mathbf{Y}_t = \\mathbf{c} + \\mathbf{A}\\mathbf{Y}_{t-1} + \\boldsymbol{\\varepsilon}_t$, forecasts at long horizons converge to:",
                "options": [
                    "Zero",
                    "The last observation $\\mathbf{Y}_T$",
                    "The unconditional mean $\\boldsymbol{\\mu} = (\\mathbf{I} - \\mathbf{A})^{-1}\\mathbf{c}$",
                    "The constant vector $\\mathbf{c}$"
                ],
                "correctExplanation": "The unconditional mean solves $\\boldsymbol{\\mu} = \\mathbf{c} + \\mathbf{A}\\boldsymbol{\\mu}$, so $\\boldsymbol{\\mu} = (\\mathbf{I} - \\mathbf{A})^{-1}\\mathbf{c}$. Since $\\mathbf{Y}_{T+h|T} - \\boldsymbol{\\mu} = \\mathbf{A}^h(\\mathbf{Y}_T - \\boldsymbol{\\mu}) \\to \\mathbf{0}$, the information in the starting point fades and forecasts revert to $\\boldsymbol{\\mu}$.",
                "incorrectExplanation": "Zero is the limit only when $\\mathbf{c} = \\mathbf{0}$; the last observation is the long-run forecast of a random walk, not of a stable VAR; $\\mathbf{c}$ equals the mean only if $\\mathbf{A} = \\mathbf{0}$."
            },
            "ro": {
                "title": "Prognoze VAR pe orizonturi lungi",
                "text": "Pentru un VAR(1) stabil $\\mathbf{Y}_t = \\mathbf{c} + \\mathbf{A}\\mathbf{Y}_{t-1} + \\boldsymbol{\\varepsilon}_t$, prognozele pe orizonturi lungi converg către:",
                "options": [
                    "Zero",
                    "Ultima observație $\\mathbf{Y}_T$",
                    "Media necondiționată $\\boldsymbol{\\mu} = (\\mathbf{I} - \\mathbf{A})^{-1}\\mathbf{c}$",
                    "Vectorul de constante $\\mathbf{c}$"
                ],
                "correctExplanation": "Media necondiționată verifică $\\boldsymbol{\\mu} = \\mathbf{c} + \\mathbf{A}\\boldsymbol{\\mu}$, deci $\\boldsymbol{\\mu} = (\\mathbf{I} - \\mathbf{A})^{-1}\\mathbf{c}$. Deoarece $\\mathbf{Y}_{T+h|T} - \\boldsymbol{\\mu} = \\mathbf{A}^h(\\mathbf{Y}_T - \\boldsymbol{\\mu}) \\to \\mathbf{0}$, informația din punctul de plecare se estompează, iar prognozele revin la $\\boldsymbol{\\mu}$.",
                "incorrectExplanation": "Zero este limita doar cînd $\\mathbf{c} = \\mathbf{0}$; ultima observație este prognoza pe termen lung a unui mers aleator, nu a unui VAR stabil; $\\mathbf{c}$ este egal cu media doar dacă $\\mathbf{A} = \\mathbf{0}$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Forecast error MSE",
                "text": "For a stable VAR(1), as the forecast horizon $h$ increases, the mean squared error matrix of the forecast converges to:",
                "options": [
                    "The zero matrix",
                    "The innovation covariance matrix $\\boldsymbol{\\Sigma}$",
                    "The unconditional variance $\\boldsymbol{\\Gamma}(0)$",
                    "Infinity"
                ],
                "correctExplanation": "$\\operatorname{MSE}(h) = \\sum_{i=0}^{h-1} \\mathbf{A}^i \\boldsymbol{\\Sigma} \\mathbf{A}^{i\\prime}$, which increases with $h$ and converges to $\\boldsymbol{\\Gamma}(0)$: at long horizons the forecast is the mean and its error variance is the variance of the process.",
                "incorrectExplanation": "$\\boldsymbol{\\Sigma}$ is the one-step MSE (the starting point, not the limit); the MSE never shrinks to zero, and it stays bounded because the VAR is stable. The limit is $\\boldsymbol{\\Gamma}(0)$."
            },
            "ro": {
                "title": "Eroarea medie pătratică a prognozei",
                "text": "Pentru un VAR(1) stabil, pe măsură ce orizontul de prognoză $h$ crește, matricea erorii medii pătratice (MSE) a prognozei converge către:",
                "options": [
                    "Matricea zero",
                    "Matricea de covarianță a inovațiilor $\\boldsymbol{\\Sigma}$",
                    "Varianța necondiționată $\\boldsymbol{\\Gamma}(0)$",
                    "Infinit"
                ],
                "correctExplanation": "$\\operatorname{MSE}(h) = \\sum_{i=0}^{h-1} \\mathbf{A}^i \\boldsymbol{\\Sigma} \\mathbf{A}^{i\\prime}$, care crește cu $h$ și converge către $\\boldsymbol{\\Gamma}(0)$: pe orizonturi lungi, prognoza este media, iar varianța erorii ei este varianța procesului.",
                "incorrectExplanation": "$\\boldsymbol{\\Sigma}$ este MSE pentru un pas (punctul de plecare, nu limita); MSE nu scade niciodată la zero și rămîne mărginită deoarece VAR-ul este stabil. Limita este $\\boldsymbol{\\Gamma}(0)$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Likelihood ratio test in a VAR",
                "text": "Under the null hypothesis, the likelihood ratio test of $r$ linear restrictions on the coefficients of a stationary VAR has asymptotically:",
                "options": [
                    "A Normal distribution",
                    "An F distribution",
                    "A $\\chi^2$ distribution with $r$ degrees of freedom",
                    "A Student's t distribution"
                ],
                "correctExplanation": "$LR = T(\\ln|\\hat{\\boldsymbol{\\Sigma}}_R| - \\ln|\\hat{\\boldsymbol{\\Sigma}}_U|) \\xrightarrow{d} \\chi^2(r)$, where $r$ is the number of restrictions (for example, the number of coefficients set to zero when the lag order is reduced).",
                "incorrectExplanation": "Normal and $t$ distributions apply to single coefficients, not to a joint test of several restrictions; an $F$ version is sometimes used as a small-sample approximation, but the asymptotic distribution of the LR statistic is $\\chi^2(r)$."
            },
            "ro": {
                "title": "Testul raportului de verosimilitate într-un VAR",
                "text": "Sub ipoteza nulă, testul raportului de verosimilitate pentru $r$ restricții liniare asupra coeficienților unui VAR staționar are asimptotic:",
                "options": [
                    "Distribuția Normală",
                    "O distribuție F",
                    "O distribuție $\\chi^2$ cu $r$ grade de libertate",
                    "O distribuție Student t"
                ],
                "correctExplanation": "$LR = T(\\ln|\\hat{\\boldsymbol{\\Sigma}}_R| - \\ln|\\hat{\\boldsymbol{\\Sigma}}_U|) \\xrightarrow{d} \\chi^2(r)$, unde $r$ este numărul de restricții (de exemplu, numărul de coeficienți anulați la reducerea ordinului lag-ului).",
                "incorrectExplanation": "Distribuția Normală și distribuția $t$ se folosesc pentru un singur coeficient, nu pentru un test comun al mai multor restricții; o variantă $F$ se folosește uneori ca aproximare pentru eșantioane mici, dar distribuția asimptotică a statisticii LR este $\\chi^2(r)$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Diebold-Mariano test",
                "text": "The Diebold-Mariano test is used to:",
                "options": [
                    "Test for Granger causality between two variables",
                    "Test for the presence of unit roots",
                    "Test whether one forecasting model is significantly more accurate than another",
                    "Test the normality of VAR residuals"
                ],
                "correctExplanation": "The test takes the loss differential $d_t = L(e_{1t}) - L(e_{2t})$ between two competing forecasts and tests $H_0\\colon E[d_t] = 0$ with a HAC (autocorrelation-robust) standard error.",
                "incorrectExplanation": "Granger causality is tested with Wald or F tests on lag coefficients, unit roots with ADF or KPSS, and residual normality with Jarque-Bera. The Diebold-Mariano test compares predictive accuracy."
            },
            "ro": {
                "title": "Testul Diebold-Mariano",
                "text": "Testul Diebold-Mariano se folosește pentru:",
                "options": [
                    "Testarea cauzalității Granger dintre două variabile",
                    "Testarea prezenței rădăcinilor unitare",
                    "Testarea faptului că un model de prognoză este semnificativ mai precis decît altul",
                    "Testarea normalității reziduurilor unui VAR"
                ],
                "correctExplanation": "Testul pornește de la diferența de pierdere $d_t = L(e_{1t}) - L(e_{2t})$ dintre două prognoze concurente și testează $H_0\\colon E[d_t] = 0$ cu o eroare standard HAC (robustă la autocorelație).",
                "incorrectExplanation": "Cauzalitatea Granger se testează cu teste Wald sau F pe coeficienții lag-urilor, rădăcinile unitare cu ADF sau KPSS, iar normalitatea reziduurilor cu Jarque-Bera. Testul Diebold-Mariano compară precizia prognozelor."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Interpreting a VAR(1) coefficient",
                "text": "In the VAR(1) model with $\\mathbf{A} = \\begin{pmatrix} 0.7 & 0.2 \\\\ -0.1 & 0.6 \\end{pmatrix}$, the coefficient $a_{21} = -0.1$ means that:",
                "options": [
                    "A one-unit increase in $Y_2$ at $t-1$ lowers $Y_1$ at $t$ by 0.1",
                    "A one-unit increase in $Y_1$ at $t-1$ raises $Y_2$ at $t$ by 0.1",
                    "A one-unit increase in $Y_1$ at $t-1$ lowers $Y_2$ at $t$ by 0.1, other things equal",
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
                    "O creștere cu o unitate a lui $Y_1$ la $t-1$ crește $Y_2$ la $t$ cu 0,1",
                    "O creștere cu o unitate a lui $Y_1$ la $t-1$ scade $Y_2$ la $t$ cu 0,1, celelalte condiții rămînînd neschimbate",
                    "O creștere cu o unitate a lui $Y_2$ la $t-1$ crește $Y_2$ la $t$ cu 0,1"
                ],
                "correctExplanation": "Elementul $(i, j)$ al matricei $\\mathbf{A}$ este efectul lui $Y_{j,t-1}$ asupra lui $Y_{i,t}$. Așadar, $a_{21} = -0{,}1$ este efectul lui $Y_{1,t-1}$ asupra lui $Y_{2,t}$: o creștere cu o unitate a lui $Y_1$ scade $Y_2$ în perioada următoare cu 0,1.",
                "incorrectExplanation": "Indicele de linie desemnează ecuația (variabila explicată), iar indicele de coloană regresorul întîrziat; inversarea lor dă efectul lui $Y_2$ asupra lui $Y_1$, adică $a_{12} = 0{,}2$. Semnul este negativ, deci efectul este o scădere."
            }
        }
    ]
};
