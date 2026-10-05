// ============================================================
// Chapter 3 quiz bank: Unit roots and ARIMA models (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['arima'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "Deterministic and stochastic trends",
                "text": "What is the main difference between a trend-stationary and a difference-stationary series?",
                "options": [
                    "In a trend-stationary series shocks fade; in a difference-stationary series they are permanent",
                    "A trend-stationary series has no trend at all",
                    "A difference-stationary series has a constant variance",
                    "The two have different mean functions"
                ],
                "correctExplanation": "Both can have the mean $\\alpha + \\beta t$. In $y_t = \\alpha + \\beta t + u_t$ a shock dies out with $u_t$; in $\\Delta y_t = \\beta + u_t$ every shock stays in the level forever, and the variance grows with $t$.",
                "incorrectExplanation": "Both series can have the same linear mean. What differs is the memory: shocks fade around a fixed line in the trend-stationary case and are permanent in the difference-stationary case, whose variance grows with $t$."
            },
            "ro": {
                "title": "Trend determinist și trend stochastic",
                "text": "Care este principala diferență dintre o serie staționară în jurul trendului și una staționară în diferențe?",
                "options": [
                    "În seria staționară în jurul trendului șocurile se sting; în cea staționară în diferențe sînt permanente",
                    "Seria staționară în jurul trendului nu are deloc trend",
                    "Seria staționară în diferențe are varianța constantă",
                    "Cele două au funcții de medie diferite"
                ],
                "correctExplanation": "Ambele pot avea media $\\alpha + \\beta t$. În $y_t = \\alpha + \\beta t + u_t$ un șoc se stinge odată cu $u_t$; în $\\Delta y_t = \\beta + u_t$ orice șoc rămîne pentru totdeauna în nivel, iar varianța crește cu $t$.",
                "incorrectExplanation": "Ambele serii pot avea aceeași medie liniară. Diferă memoria: șocurile se sting în jurul unei drepte fixe în primul caz și sînt permanente în al doilea, unde varianța crește cu $t$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Integrated processes",
                "text": "What does $y_t \\sim I(2)$ mean?",
                "options": [
                    "The series is stationary around a quadratic trend",
                    "The series must be differenced twice to become stationary",
                    "The series has two structural breaks",
                    "The series is stationary after one difference"
                ],
                "correctExplanation": "$I(d)$ means that $\\Delta^d y_t$ is stationary and $\\Delta^{d-1}y_t$ is not; for $d = 2$, $\\Delta^2 y_t = y_t - 2y_{t-1} + y_{t-2}$ is stationary.",
                "incorrectExplanation": "The order of integration counts the differences needed for stationarity, not breaks or the degree of a deterministic trend. A series that is stationary after one difference is $I(1)$."
            },
            "ro": {
                "title": "Procese integrate",
                "text": "Ce înseamnă $y_t \\sim I(2)$?",
                "options": [
                    "Seria este staționară în jurul unui trend pătratic",
                    "Seria trebuie diferențiată de două ori ca să devină staționară",
                    "Seria are două rupturi structurale",
                    "Seria este staționară după o diferențiere"
                ],
                "correctExplanation": "$I(d)$ înseamnă că $\\Delta^d y_t$ este staționar, iar $\\Delta^{d-1}y_t$ nu este; pentru $d = 2$, $\\Delta^2 y_t = y_t - 2y_{t-1} + y_{t-2}$ este staționar.",
                "incorrectExplanation": "Ordinul de integrare numără diferențierile necesare pentru staționaritate, nu rupturile sau gradul unui trend determinist. O serie staționară după o diferențiere este $I(1)$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The unit root",
                "text": "For $y_t = 1.5y_{t-1} - 0.5y_{t-2} + \\varepsilon_t$, what are the roots of $\\phi(z) = 1 - 1.5z + 0.5z^2$?",
                "options": [
                    "$z = 1.5$ and $z = 0.5$, so $y_t$ is stationary",
                    "$z = -1$ and $z = 2$, so $y_t$ is stationary",
                    "$z = 1$ and $z = 2$, so $y_t \\sim I(1)$",
                    "Both roots equal 1, so $y_t \\sim I(2)$"
                ],
                "correctExplanation": "$1 - 1.5z + 0.5z^2 = (1 - z)(1 - 0.5z)$: one root on the unit circle and one outside it, so $\\Delta y_t = 0.5\\Delta y_{t-1} + \\varepsilon_t$ is a stationary AR(1).",
                "incorrectExplanation": "The roots are those of the polynomial, not its coefficients. Factoring gives $(1 - z)(1 - 0.5z)$: one unit root, so one difference is needed."
            },
            "ro": {
                "title": "Rădăcina unitară",
                "text": "Pentru $y_t = 1{,}5y_{t-1} - 0{,}5y_{t-2} + \\varepsilon_t$, care sînt rădăcinile lui $\\phi(z) = 1 - 1{,}5z + 0{,}5z^2$?",
                "options": [
                    "$z = 1{,}5$ și $z = 0{,}5$, deci $y_t$ este staționar",
                    "$z = -1$ și $z = 2$, deci $y_t$ este staționar",
                    "$z = 1$ și $z = 2$, deci $y_t \\sim I(1)$",
                    "Ambele rădăcini sînt egale cu 1, deci $y_t \\sim I(2)$"
                ],
                "correctExplanation": "$1 - 1{,}5z + 0{,}5z^2 = (1 - z)(1 - 0{,}5z)$: o rădăcină pe cercul unitate și una în afara lui, deci $\\Delta y_t = 0{,}5\\Delta y_{t-1} + \\varepsilon_t$ este un AR(1) staționar.",
                "incorrectExplanation": "Rădăcinile sînt ale polinomului, nu coeficienții lui. Factorizarea dă $(1 - z)(1 - 0{,}5z)$: o rădăcină unitară, deci este nevoie de o diferențiere."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Spurious regression",
                "text": "Two independent random walks are regressed on each other by OLS. What do you typically observe?",
                "options": [
                    "A $t$-statistic close to 0 and $R^2$ close to 0",
                    "A rejection rate of the slope test close to 5%",
                    "A Durbin-Watson statistic close to 2",
                    "A large $|t|$, a high $R^2$ and a Durbin-Watson statistic close to 0"
                ],
                "correctExplanation": "This is the spurious regression of Granger and Newbold (1974): the residuals are themselves $I(1)$, the usual standard error is far too small, $R^2$ stays high and DW tends to 0. Rule of thumb: be suspicious when $R^2 > \\mathrm{DW}$.",
                "incorrectExplanation": "Those are the results of a regression between stationary, unrelated series. With independent random walks the $t$-test rejects far more often than 5%, and the residuals are strongly autocorrelated (DW near 0)."
            },
            "ro": {
                "title": "Regresia falsă",
                "text": "Două mersuri aleatoare independente sînt regresate unul pe celălalt prin OLS. Ce observați de obicei?",
                "options": [
                    "O statistică $t$ apropiată de 0 și $R^2$ apropiat de 0",
                    "O rată de respingere a testului pantei apropiată de 5%",
                    "O statistică Durbin-Watson apropiată de 2",
                    "Un $|t|$ mare, un $R^2$ ridicat și o statistică Durbin-Watson apropiată de 0"
                ],
                "correctExplanation": "Aceasta este regresia falsă a lui Granger și Newbold (1974): reziduurile sînt ele însele $I(1)$, eroarea standard obișnuită este mult prea mică, $R^2$ rămîne ridicat, iar DW tinde la 0. Regula practică: suspectați o regresie falsă cînd $R^2 > \\mathrm{DW}$.",
                "incorrectExplanation": "Acestea sînt rezultatele unei regresii între serii staționare fără legătură. Cu mersuri aleatoare independente, testul $t$ respinge mult mai des decît în 5% din cazuri, iar reziduurile sînt puternic autocorelate (DW aproape de 0)."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Spurious regression and sample size",
                "text": "In a spurious regression of one random walk on another, what happens as the sample size $T$ grows?",
                "options": [
                    "The problem disappears because OLS is consistent",
                    "The rejection rate converges to 5%",
                    "The $t$-statistic grows like $\\sqrt{T}$ and the rejection rate tends to 100%",
                    "The estimated slope converges to 0"
                ],
                "correctExplanation": "Phillips (1986): the slope converges to a random variable, not to 0, and $t$ diverges at rate $\\sqrt{T}$. In the simulation of the chapter the rejection rate rises from about 68% at $T = 50$ to about 92% at $T = 1000$.",
                "incorrectExplanation": "More data do not cure a spurious regression: the slope does not converge to 0 and the rejection rate increases with $T$. The remedy is to difference the series or to test for cointegration."
            },
            "ro": {
                "title": "Regresia falsă și volumul eșantionului",
                "text": "Într-o regresie falsă a unui mers aleator pe altul, ce se întîmplă cînd volumul eșantionului $T$ crește?",
                "options": [
                    "Problema dispare, deoarece estimatorul OLS este consistent",
                    "Rata de respingere converge la 5%",
                    "Statistica $t$ crește ca $\\sqrt{T}$, iar rata de respingere tinde la 100%",
                    "Panta estimată converge la 0"
                ],
                "correctExplanation": "Phillips (1986): panta converge la o variabilă aleatoare, nu la 0, iar $t$ crește ca $\\sqrt{T}$. În simularea din capitol, rata de respingere urcă de la circa 68% pentru $T = 50$ la circa 92% pentru $T = 1000$.",
                "incorrectExplanation": "Mai multe date nu vindecă o regresie falsă: panta nu converge la 0, iar rata de respingere crește cu $T$. Remediul este diferențierea seriilor sau testarea cointegrării."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "ADF hypotheses",
                "text": "What are the hypotheses of the augmented Dickey-Fuller (ADF) test on $\\Delta y_t = c + \\gamma y_{t-1} + \\sum_j \\delta_j \\Delta y_{t-j} + \\varepsilon_t$?",
                "options": [
                    "$H_0$: $\\gamma = 0$ (unit root); $H_1$: $\\gamma < 0$ (stationary)",
                    "$H_0$: $\\gamma < 0$ (stationary); $H_1$: $\\gamma = 0$ (unit root)",
                    "$H_0$: $\\gamma = 0$; $H_1$: $\\gamma \\neq 0$, a two-sided test",
                    "$H_0$: all $\\delta_j = 0$; $H_1$: some $\\delta_j \\neq 0$"
                ],
                "correctExplanation": "With $\\gamma = \\phi - 1$, the null is a unit root and the alternative stationarity; the test is one-sided and rejects for very negative values of $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$.",
                "incorrectExplanation": "The ADF null is the unit root; stationarity as the null is the KPSS test. The alternative is one-sided ($\\gamma < 0$), and the lagged differences only clean the errors; they are not tested."
            },
            "ro": {
                "title": "Ipotezele testului ADF",
                "text": "Care sînt ipotezele testului Dickey-Fuller augmentat (ADF) pe $\\Delta y_t = c + \\gamma y_{t-1} + \\sum_j \\delta_j \\Delta y_{t-j} + \\varepsilon_t$?",
                "options": [
                    "$H_0$: $\\gamma = 0$ (rădăcină unitară); $H_1$: $\\gamma < 0$ (staționar)",
                    "$H_0$: $\\gamma < 0$ (staționar); $H_1$: $\\gamma = 0$ (rădăcină unitară)",
                    "$H_0$: $\\gamma = 0$; $H_1$: $\\gamma \\neq 0$, un test bilateral",
                    "$H_0$: toți $\\delta_j = 0$; $H_1$: unii $\\delta_j \\neq 0$"
                ],
                "correctExplanation": "Cu $\\gamma = \\phi - 1$, ipoteza nulă este rădăcina unitară, iar alternativa staționaritatea; testul este unilateral și respinge pentru valori foarte negative ale lui $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$.",
                "incorrectExplanation": "Ipoteza nulă ADF este rădăcina unitară; staționaritatea ca ipoteză nulă aparține testului KPSS. Alternativa este unilaterală ($\\gamma < 0$), iar diferențele decalate doar curăță erorile; ele nu sînt testate."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Reading an ADF result",
                "text": "An ADF test with a constant gives $\\tau = -2.1$; the 5% critical value is $-2.86$. What do you conclude?",
                "options": [
                    "Reject the unit root, since $-2.1 < -1.645$",
                    "Reject the unit root, since $|-2.1| > 1.96$",
                    "The series is proved to be $I(1)$",
                    "Do not reject the unit root at 5%"
                ],
                "correctExplanation": "The test rejects only if $\\tau$ is below the Dickey-Fuller critical value: $-2.1 > -2.86$, so the unit root is not rejected. Not rejecting is not a proof: the test has low power near $\\phi = 1$.",
                "incorrectExplanation": "Normal critical values ($-1.645$, $\\pm 1.96$) do not apply to $\\tau$ under a unit root. And a non-rejection does not prove $H_0$: it only means the data are compatible with a unit root."
            },
            "ro": {
                "title": "Interpretarea unui rezultat ADF",
                "text": "Un test ADF cu constantă dă $\\tau = -2{,}1$; valoarea critică de 5% este $-2{,}86$. Ce concluzie trageți?",
                "options": [
                    "Respingem rădăcina unitară, deoarece $-2{,}1 < -1{,}645$",
                    "Respingem rădăcina unitară, deoarece $|-2{,}1| > 1{,}96$",
                    "S-a demonstrat că seria este $I(1)$",
                    "Nu respingem rădăcina unitară la 5%"
                ],
                "correctExplanation": "Testul respinge doar dacă $\\tau$ este sub valoarea critică Dickey-Fuller: $-2{,}1 > -2{,}86$, deci rădăcina unitară nu este respinsă. Nerespingerea nu este o demonstrație: testul are putere mică în apropierea lui $\\phi = 1$.",
                "incorrectExplanation": "Valorile critice ale distribuției Normale ($-1{,}645$, $\\pm 1{,}96$) nu se aplică lui $\\tau$ cînd există o rădăcină unitară. Iar nerespingerea nu demonstrează $H_0$: arată doar că datele sînt compatibile cu o rădăcină unitară."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The Dickey-Fuller distribution",
                "text": "Why does the Dickey-Fuller test use its own critical values instead of Student or Normal ones?",
                "options": [
                    "Because the errors are never Normal in economic data",
                    "Under a unit root the regressor $y_{t-1}$ is non-stationary and $\\tau$ has a non-standard, left-shifted distribution",
                    "Because the sample is usually small",
                    "Because the test is two-sided"
                ],
                "correctExplanation": "Under $H_0$ the regressor is a random walk; $\\hat\\phi$ converges at rate $T$ and $\\tau$ converges to a functional of Brownian motion. Its 5% quantiles are about $-1.94$, $-2.86$ and $-3.41$ for the three specifications.",
                "incorrectExplanation": "The problem remains with Normal errors and in large samples: it comes from the non-stationary regressor. The test is one-sided, to the left."
            },
            "ro": {
                "title": "Distribuția Dickey-Fuller",
                "text": "De ce folosește testul Dickey-Fuller valori critice proprii, nu pe cele ale legii Student sau ale distribuției Normale?",
                "options": [
                    "Pentru că erorile nu sînt niciodată normale în datele economice",
                    "Cînd există o rădăcină unitară, regresorul $y_{t-1}$ este nestaționar, iar $\\tau$ are o distribuție nestandard, deplasată la stînga",
                    "Pentru că eșantionul este de obicei mic",
                    "Pentru că testul este bilateral"
                ],
                "correctExplanation": "În ipoteza $H_0$, regresorul este un mers aleator; $\\hat\\phi$ converge cu viteza $T$, iar $\\tau$ converge la o funcțională a mișcării browniene. Cuantilele ei de 5% sînt circa $-1{,}94$, $-2{,}86$ și $-3{,}41$ pentru cele trei specificații.",
                "incorrectExplanation": "Problema rămîne și cu erori normale și în eșantioane mari: ea provine din regresorul nestaționar. Testul este unilateral, la stînga."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Deterministic terms",
                "text": "Which ADF specification is appropriate for the log of Romanian real GDP?",
                "options": [
                    "No constant",
                    "Constant and linear trend",
                    "Constant only",
                    "Constant and quadratic trend"
                ],
                "correctExplanation": "Log GDP trends upwards, so the alternative must be ``stationary around a trend''. Without the trend, a trend-stationary series looks like a unit root and the test almost never rejects.",
                "incorrectExplanation": "A specification without a trend cannot describe a trending series under the alternative; a quadratic trend is not plausible for log GDP and costs power."
            },
            "ro": {
                "title": "Termenii determiniști",
                "text": "Ce specificație ADF este potrivită pentru logaritmul PIB-ului real al României?",
                "options": [
                    "Fără constantă",
                    "Constantă și trend liniar",
                    "Doar constantă",
                    "Constantă și trend pătratic"
                ],
                "correctExplanation": "Logaritmul PIB crește în timp, deci alternativa trebuie să fie „staționar în jurul unui trend”. Fără trend, o serie staționară în jurul trendului pare să aibă rădăcină unitară, iar testul aproape niciodată nu respinge.",
                "incorrectExplanation": "O specificație fără trend nu poate descrie o serie cu trend în ipoteza alternativă; un trend pătratic nu este plauzibil pentru logaritmul PIB și costă putere."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Lags in the ADF test",
                "text": "Why are lagged differences $\\Delta y_{t-j}$ added to the Dickey-Fuller regression?",
                "options": [
                    "To increase the $R^2$ of the regression",
                    "To test for seasonality",
                    "To make the series stationary before testing",
                    "To remove autocorrelation from the errors, so that the test has the right size"
                ],
                "correctExplanation": "If $\\Delta y_t$ is autocorrelated, the DF errors are not white noise and the size is wrong. Said and Dickey (1984): with enough lags the test remains valid even for ARMA errors. The number of lags is chosen by AIC or BIC up to $12(T/100)^{1/4}$.",
                "incorrectExplanation": "The lags do not difference the series again and are not a seasonality test; a higher $R^2$ is not the goal. They make the errors close to white noise so that the Dickey-Fuller critical values apply."
            },
            "ro": {
                "title": "Decalajele din testul ADF",
                "text": "De ce se adaugă diferențe decalate $\\Delta y_{t-j}$ în regresia Dickey-Fuller?",
                "options": [
                    "Pentru a mări $R^2$ al regresiei",
                    "Pentru a testa sezonalitatea",
                    "Pentru a face seria staționară înainte de testare",
                    "Pentru a elimina autocorelația din erori, astfel încît testul să aibă mărimea corectă"
                ],
                "correctExplanation": "Dacă $\\Delta y_t$ este autocorelat, erorile DF nu sînt zgomot alb, iar mărimea testului este greșită. Said și Dickey (1984): cu suficiente decalaje, testul rămîne valid chiar și pentru erori ARMA. Numărul de decalaje se alege după AIC sau BIC, pînă la $12(T/100)^{1/4}$.",
                "incorrectExplanation": "Decalajele nu diferențiază din nou seria și nu testează sezonalitatea; un $R^2$ mai mare nu este scopul. Ele fac erorile apropiate de zgomotul alb, astfel încît valorile critice Dickey-Fuller să fie valabile."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Power of unit-root tests",
                "text": "With $T = 100$ observations of an AR(1) with $\\phi = 0.95$, how often does the Dickey-Fuller test (constant, 5%) reject the unit root?",
                "options": [
                    "Rarely: in roughly one sample out of eight",
                    "Almost always, since $\\phi < 1$",
                    "In exactly 5% of the samples",
                    "In about half of the samples"
                ],
                "correctExplanation": "The simulation of the chapter gives about 12% at $T = 100$, 47% at $T = 250$ and 97% at $T = 500$. Near $\\phi = 1$ the test has low power; the time span matters, not the frequency.",
                "incorrectExplanation": "A stationary $\\phi$ close to 1 is hard to tell from a unit root in short samples. The 5% is the size (the rejection rate when $\\phi = 1$), not the power."
            },
            "ro": {
                "title": "Puterea testelor de rădăcină unitară",
                "text": "Cu $T = 100$ de observații dintr-un AR(1) cu $\\phi = 0{,}95$, cît de des respinge testul Dickey-Fuller (constantă, 5%) rădăcina unitară?",
                "options": [
                    "Rar: în aproximativ un eșantion din opt",
                    "Aproape întotdeauna, deoarece $\\phi < 1$",
                    "În exact 5% din eșantioane",
                    "În aproximativ jumătate din eșantioane"
                ],
                "correctExplanation": "Simularea din capitol dă circa 12% pentru $T = 100$, 47% pentru $T = 250$ și 97% pentru $T = 500$. În apropierea lui $\\phi = 1$ testul are putere mică; contează lungimea perioadei, nu frecvența.",
                "incorrectExplanation": "Un $\\phi$ staționar apropiat de 1 este greu de deosebit de o rădăcină unitară în eșantioane scurte. Valoarea de 5% este mărimea testului (rata de respingere cînd $\\phi = 1$), nu puterea lui."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The Phillips-Perron test",
                "text": "How does the Phillips-Perron test differ from the ADF test?",
                "options": [
                    "It has stationarity as the null hypothesis",
                    "It uses Normal critical values",
                    "It keeps the regression without lagged differences and corrects $\\tau$ with a long-run variance estimate",
                    "It allows for a structural break at an unknown date"
                ],
                "correctExplanation": "Phillips and Perron (1988) correct the Dickey-Fuller $t$-ratio non-parametrically (Newey-West, Bartlett weights); the null and the critical values are those of ADF. It is robust to heteroskedasticity, but has size problems with a negative MA component.",
                "incorrectExplanation": "Stationarity as the null is KPSS; breaks are handled by Zivot-Andrews. Phillips-Perron uses the same Dickey-Fuller critical values as ADF."
            },
            "ro": {
                "title": "Testul Phillips-Perron",
                "text": "Prin ce diferă testul Phillips-Perron de testul ADF?",
                "options": [
                    "Are staționaritatea ca ipoteză nulă",
                    "Folosește valorile critice ale distribuției Normale",
                    "Păstrează regresia fără diferențe decalate și corectează $\\tau$ printr-o estimare a varianței pe termen lung",
                    "Permite o ruptură structurală la o dată necunoscută"
                ],
                "correctExplanation": "Phillips și Perron (1988) corectează neparametric raportul $t$ Dickey-Fuller (Newey-West, ponderi Bartlett); ipoteza nulă și valorile critice sînt cele ale testului ADF. Testul este robust la heteroscedasticitate, dar are probleme de mărime cînd există o componentă MA negativă.",
                "incorrectExplanation": "Staționaritatea ca ipoteză nulă aparține testului KPSS; rupturile sînt tratate de Zivot-Andrews. Phillips-Perron folosește aceleași valori critice Dickey-Fuller ca ADF."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The KPSS test",
                "text": "What is the null hypothesis of the KPSS test?",
                "options": [
                    "The series has a unit root",
                    "The series is white noise",
                    "The series has a structural break",
                    "The series is stationary around a level or a linear trend"
                ],
                "correctExplanation": "KPSS writes $y_t$ as a trend, a random walk and a stationary error, and tests that the random walk has zero variance. Large values of $\\eta = \\sum S_t^2/(T^2\\hat\\lambda^2)$ reject stationarity (5%: 0.463 for a level, 0.146 for a trend).",
                "incorrectExplanation": "The unit root is the null of ADF and PP, not of KPSS. KPSS allows any stationary dynamics, so it is not a white-noise test, and it does not model breaks."
            },
            "ro": {
                "title": "Testul KPSS",
                "text": "Care este ipoteza nulă a testului KPSS?",
                "options": [
                    "Seria are o rădăcină unitară",
                    "Seria este zgomot alb",
                    "Seria are o ruptură structurală",
                    "Seria este staționară în jurul unui nivel sau al unui trend liniar"
                ],
                "correctExplanation": "KPSS scrie $y_t$ ca sumă dintre un trend, un mers aleator și o eroare staționară și testează dacă mersul aleator are varianța zero. Valorile mari ale lui $\\eta = \\sum S_t^2/(T^2\\hat\\lambda^2)$ resping staționaritatea (5%: 0,463 pentru nivel, 0,146 pentru trend).",
                "incorrectExplanation": "Rădăcina unitară este ipoteza nulă a testelor ADF și PP, nu a testului KPSS. KPSS permite orice dinamică staționară, deci nu este un test de zgomot alb, și nu modelează rupturi."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "ADF and KPSS together (1)",
                "text": "ADF does not reject the unit root and KPSS rejects stationarity. What is the conclusion?",
                "options": [
                    "Both tests point to stationarity",
                    "The results conflict, so there must be a break",
                    "Both tests point to a unit root: the series is $I(1)$",
                    "The tests are inconclusive"
                ],
                "correctExplanation": "The two tests have opposite nulls; here both agree on a unit root. This is the pattern of log prices, the exchange rate and log GDP in the chapter.",
                "incorrectExplanation": "Stationarity would need ADF to reject and KPSS not to reject. A conflict means both reject; inconclusive means neither rejects."
            },
            "ro": {
                "title": "ADF și KPSS împreună (1)",
                "text": "ADF nu respinge rădăcina unitară, iar KPSS respinge staționaritatea. Care este concluzia?",
                "options": [
                    "Ambele teste indică staționaritatea",
                    "Rezultatele sînt în conflict, deci trebuie să existe o ruptură",
                    "Ambele teste indică o rădăcină unitară: seria este $I(1)$",
                    "Testele sînt neconcludente"
                ],
                "correctExplanation": "Cele două teste au ipoteze nule opuse; aici ambele indică o rădăcină unitară. Acesta este tiparul logaritmilor prețurilor, al cursului de schimb și al logaritmului PIB din capitol.",
                "incorrectExplanation": "Staționaritatea ar cere ca ADF să respingă și KPSS să nu respingă. Un conflict înseamnă că ambele resping; neconcludent înseamnă că niciunul nu respinge."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ADF and KPSS together (2)",
                "text": "For Romanian 12-month inflation, neither ADF nor KPSS rejects its null. How should this be read?",
                "options": [
                    "The series is proved to be stationary",
                    "Inconclusive: the sample cannot tell a very persistent stationary series from a unit root",
                    "The series is proved to be $I(1)$",
                    "One of the two tests is computed incorrectly"
                ],
                "correctExplanation": "When neither test rejects, both hypotheses are compatible with the data, typically because the series is very persistent or the sample short. Then $d$ is chosen by the purpose: $d = 1$ gives the more cautious, wider intervals.",
                "incorrectExplanation": "Non-rejections never prove a null. Two non-rejections with opposite nulls are a sign of low power, not of an error in the computation."
            },
            "ro": {
                "title": "ADF și KPSS împreună (2)",
                "text": "Pentru inflația anuală din România, nici ADF, nici KPSS nu își resping ipoteza nulă. Cum se interpretează acest rezultat?",
                "options": [
                    "S-a demonstrat că seria este staționară",
                    "Neconcludent: eșantionul nu poate deosebi o serie staționară foarte persistentă de o rădăcină unitară",
                    "S-a demonstrat că seria este $I(1)$",
                    "Unul dintre cele două teste este calculat greșit"
                ],
                "correctExplanation": "Cînd niciun test nu respinge, ambele ipoteze sînt compatibile cu datele, de obicei pentru că seria este foarte persistentă sau eșantionul scurt. Atunci $d$ se alege după scop: $d = 1$ dă intervale mai largi, deci mai prudente.",
                "incorrectExplanation": "Nerespingerile nu demonstrează niciodată o ipoteză nulă. Două nerespingeri cu ipoteze nule opuse arată o putere mică, nu o eroare de calcul."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Structural breaks",
                "text": "According to Perron (1989), what does a one-time shift in the mean of a stationary series do to the ADF test?",
                "options": [
                    "It makes the test reject the unit root less often",
                    "It makes the test reject the unit root more often",
                    "It has no effect if the sample is large",
                    "It changes the critical values of the test"
                ],
                "correctExplanation": "The DF regression explains the shift with $\\hat\\phi$ close to 1, so the series looks like a unit root and power falls. In the chapter's simulation, a level shift lowers the rejection rate from 100% to about 72%.",
                "incorrectExplanation": "The break biases the test towards non-rejection, not rejection, and the bias does not vanish with more data. The critical values of ADF are unchanged; tests that model the break (Zivot-Andrews) have their own."
            },
            "ro": {
                "title": "Rupturi structurale",
                "text": "Potrivit lui Perron (1989), ce efect are asupra testului ADF o schimbare unică a mediei unei serii staționare?",
                "options": [
                    "Face ca testul să respingă mai rar rădăcina unitară",
                    "Face ca testul să respingă mai des rădăcina unitară",
                    "Nu are niciun efect dacă eșantionul este mare",
                    "Schimbă valorile critice ale testului"
                ],
                "correctExplanation": "Regresia DF explică schimbarea printr-un $\\hat\\phi$ apropiat de 1, deci seria pare să aibă rădăcină unitară, iar puterea scade. În simularea din capitol, o schimbare de nivel reduce rata de respingere de la 100% la circa 72%.",
                "incorrectExplanation": "Ruptura deplasează testul spre nerespingere, nu spre respingere, iar efectul nu dispare cu mai multe date. Valorile critice ADF rămîn aceleași; testele care modelează ruptura (Zivot-Andrews) au valori critice proprii."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The Zivot-Andrews test",
                "text": "How does the Zivot-Andrews test choose the break date?",
                "options": [
                    "It runs the ADF regression with a break dummy at every candidate date and takes the date with the most negative $t$-statistic",
                    "The date is fixed in advance from known events",
                    "It takes the date of the largest observation",
                    "It uses the middle of the sample"
                ],
                "correctExplanation": "Zivot and Andrews (1992) answered the critique of Perron (1989), whose dates were chosen after looking at the data. The search over dates makes the critical values more negative (about $-4.81$ at 5% for a break in the level).",
                "incorrectExplanation": "Fixing the date in advance is Perron's approach. The test searches all dates in the central part of the sample and keeps the minimum $t$-ratio of $\\gamma$."
            },
            "ro": {
                "title": "Testul Zivot-Andrews",
                "text": "Cum alege testul Zivot-Andrews data rupturii?",
                "options": [
                    "Estimează regresia ADF cu o variabilă dummy pentru ruptură la fiecare dată posibilă și alege data cu statistica $t$ cea mai negativă",
                    "Data este fixată dinainte pe baza evenimentelor cunoscute",
                    "Alege data celei mai mari observații",
                    "Folosește mijlocul eșantionului"
                ],
                "correctExplanation": "Zivot și Andrews (1992) au răspuns criticii aduse lui Perron (1989), ale cărui date fuseseră alese după examinarea datelor. Căutarea pe toate datele face valorile critice mai negative (circa $-4{,}81$ la 5% pentru o ruptură în nivel).",
                "incorrectExplanation": "Fixarea dinainte a datei este abordarea lui Perron. Testul caută toate datele din partea centrală a eșantionului și păstrează raportul $t$ minim al lui $\\gamma$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "ARIMA(0,1,1) and exponential smoothing",
                "text": "Which forecasting method gives the same point forecasts as ARIMA(0,1,1) with $\\theta = -0.6$?",
                "options": [
                    "Simple exponential smoothing with $\\alpha = 0.6$",
                    "The random walk with drift",
                    "Holt's linear method",
                    "Simple exponential smoothing with $\\alpha = 0.4$"
                ],
                "correctExplanation": "$\\hat y_{T+1} = y_T + \\theta\\hat\\varepsilon_T = (1 + \\theta)y_T - \\theta\\hat y_T$: SES with $\\alpha = 1 + \\theta = 0.4$ (Chapter 0). Holt's method corresponds to ARIMA(0,2,2).",
                "incorrectExplanation": "The smoothing parameter is $1 + \\theta$, not $-\\theta$; a random walk would have $\\theta = 0$, and Holt's linear method corresponds to an ARIMA(0,2,2)."
            },
            "ro": {
                "title": "ARIMA(0,1,1) și netezirea exponențială",
                "text": "Ce metodă de prognoză dă aceleași prognoze punctuale ca ARIMA(0,1,1) cu $\\theta = -0{,}6$?",
                "options": [
                    "Netezirea exponențială simplă cu $\\alpha = 0{,}6$",
                    "Mersul aleator cu derivă",
                    "Metoda liniară Holt",
                    "Netezirea exponențială simplă cu $\\alpha = 0{,}4$"
                ],
                "correctExplanation": "$\\hat y_{T+1} = y_T + \\theta\\hat\\varepsilon_T = (1 + \\theta)y_T - \\theta\\hat y_T$: SES cu $\\alpha = 1 + \\theta = 0{,}4$ (Capitolul 0). Metoda Holt corespunde unui ARIMA(0,2,2).",
                "incorrectExplanation": "Parametrul de netezire este $1 + \\theta$, nu $-\\theta$; un mers aleator ar avea $\\theta = 0$, iar metoda liniară Holt corespunde unui ARIMA(0,2,2)."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The constant in ARIMA",
                "text": "In an ARIMA$(p,1,q)$ model with a non-zero constant, what do the long-run forecasts look like?",
                "options": [
                    "A flat line at the sample mean",
                    "A quadratic curve",
                    "A straight line with slope equal to the mean change (the drift)",
                    "A line that returns to a fixed trend"
                ],
                "correctExplanation": "With $d = 1$ the constant is the drift: the forecasts of $\\Delta y_t$ converge to the mean change, so the level forecasts grow linearly from the last value. With $d = 0$ the forecasts go to the mean; with $d = 2$ a constant gives a quadratic trend.",
                "incorrectExplanation": "A flat line corresponds to $d = 1$ without a constant (or $d = 0$, towards the mean); a quadratic curve to $d = 2$ with a constant. The line starts from the last observation, not from a fixed trend."
            },
            "ro": {
                "title": "Constanta în ARIMA",
                "text": "Într-un model ARIMA$(p,1,q)$ cu constantă nenulă, cum arată prognozele pe termen lung?",
                "options": [
                    "O linie orizontală la media eșantionului",
                    "O curbă pătratică",
                    "O dreaptă cu panta egală cu variația medie (deriva)",
                    "O dreaptă care revine la un trend fix"
                ],
                "correctExplanation": "Cu $d = 1$, constanta este deriva: prognozele lui $\\Delta y_t$ converg la variația medie, deci prognozele nivelului cresc liniar de la ultima valoare. Cu $d = 0$, prognozele tind spre medie; cu $d = 2$, o constantă dă un trend pătratic.",
                "incorrectExplanation": "O linie orizontală corespunde lui $d = 1$ fără constantă (sau lui $d = 0$, spre medie); o curbă pătratică lui $d = 2$ cu constantă. Dreapta pornește de la ultima observație, nu de la un trend fix."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "An ARIMA forecast by hand",
                "text": "ARIMA(1,1,0) without a constant, $\\phi = 0.6$; the data end with $y_T = 108$ and $\\Delta y_T = 5$. What is $\\hat y_{T+1}$?",
                "options": [
                    "108",
                    "111",
                    "113",
                    "110.4"
                ],
                "correctExplanation": "Forecast the change, then add it: $\\widehat{\\Delta y}_{T+1} = 0.6 \\cdot 5 = 3$, so $\\hat y_{T+1} = 108 + 3 = 111$; next, $\\hat y_{T+2} = 111 + 1.8 = 112.8$.",
                "incorrectExplanation": "108 ignores the momentum, 113 adds the whole last change, and 110.4 multiplies the wrong quantity. The model is $\\Delta y_t = 0.6\\,\\Delta y_{t-1} + \\varepsilon_t$."
            },
            "ro": {
                "title": "O prognoză ARIMA calculată de mînă",
                "text": "ARIMA(1,1,0) fără constantă, $\\phi = 0{,}6$; datele se termină cu $y_T = 108$ și $\\Delta y_T = 5$. Cît este $\\hat y_{T+1}$?",
                "options": [
                    "108",
                    "111",
                    "113",
                    "110,4"
                ],
                "correctExplanation": "Prognozăm variația, apoi o adunăm: $\\widehat{\\Delta y}_{T+1} = 0{,}6 \\cdot 5 = 3$, deci $\\hat y_{T+1} = 108 + 3 = 111$; apoi $\\hat y_{T+2} = 111 + 1{,}8 = 112{,}8$.",
                "incorrectExplanation": "108 ignoră inerția, 113 adună întreaga ultimă variație, iar 110,4 înmulțește mărimea greșită. Modelul este $\\Delta y_t = 0{,}6\\,\\Delta y_{t-1} + \\varepsilon_t$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Forecast intervals of a random walk",
                "text": "For a random walk, how does the width of the 95% forecast interval change with the horizon $h$?",
                "options": [
                    "It stays constant",
                    "It grows like $h$",
                    "It grows like $\\sqrt{h}$",
                    "It converges to a finite limit"
                ],
                "correctExplanation": "The forecast error is the sum of $h$ future shocks, with variance $h\\sigma^2$, so the half-width is $1.96\\sigma\\sqrt{h}$: four times the horizon, twice the width. For $d = 0$ the width converges to a limit.",
                "incorrectExplanation": "A constant or bounded width belongs to stationary models; growth like $h$ (and faster) appears with $d = 2$, not with a random walk."
            },
            "ro": {
                "title": "Intervalele de prognoză ale unui mers aleator",
                "text": "Pentru un mers aleator, cum se schimbă lățimea intervalului de prognoză de 95% cu orizontul $h$?",
                "options": [
                    "Rămîne constantă",
                    "Crește ca $h$",
                    "Crește ca $\\sqrt{h}$",
                    "Converge la o limită finită"
                ],
                "correctExplanation": "Eroarea de prognoză este suma a $h$ șocuri viitoare, cu varianța $h\\sigma^2$, deci semilățimea este $1{,}96\\sigma\\sqrt{h}$: un orizont de patru ori mai mare, o lățime de două ori mai mare. Pentru $d = 0$, lățimea converge la o limită.",
                "incorrectExplanation": "O lățime constantă sau mărginită aparține modelelor staționare; o creștere ca $h$ (și mai rapidă) apare pentru $d = 2$, nu pentru un mers aleator."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Over-differencing",
                "text": "After differencing a series once more, $\\hat\\rho(1) = -0.48$, the variance has doubled and an MA(1) fit gives $\\hat\\theta = -0.99$. What happened?",
                "options": [
                    "The series needs a further difference",
                    "The series was over-differenced",
                    "The series has a seasonal unit root",
                    "The model is correctly specified"
                ],
                "correctExplanation": "Differencing an $I(0)$ series creates an MA unit root ($\\theta = -1$), $\\rho(1)$ near $-0.5$ and a larger variance. In the chapter, the second difference of Romanian GDP shows all three symptoms.",
                "incorrectExplanation": "A further difference would make things worse. These symptoms are those of a non-invertible MA part created by an extra difference, not of seasonality or of a correct model."
            },
            "ro": {
                "title": "Supradiferențierea",
                "text": "După încă o diferențiere a unei serii, $\\hat\\rho(1) = -0{,}48$, varianța s-a dublat, iar un MA(1) estimat dă $\\hat\\theta = -0{,}99$. Ce s-a întîmplat?",
                "options": [
                    "Seria are nevoie de încă o diferențiere",
                    "Seria a fost supradiferențiată",
                    "Seria are o rădăcină unitară sezonieră",
                    "Modelul este specificat corect"
                ],
                "correctExplanation": "Diferențierea unei serii $I(0)$ creează o rădăcină unitară MA ($\\theta = -1$), un $\\rho(1)$ apropiat de $-0{,}5$ și o varianță mai mare. În capitol, a doua diferență a PIB-ului României arată toate cele trei simptome.",
                "incorrectExplanation": "O nouă diferențiere ar înrăutăți situația. Aceste simptome sînt cele ale unei părți MA neinversabile, create de o diferențiere în plus, nu ale sezonalității sau ale unui model corect."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Automatic ARIMA",
                "text": "An automatic search reports AICc = 451 for ARIMA(1,1,0) and AICc = 456 for ARIMA(2,0,0) with a mean. Can you choose ARIMA(1,1,0) because its AICc is lower?",
                "options": [
                    "Yes: the lower AICc always wins",
                    "Yes, but only if both models converged",
                    "No: AICc can only compare models with the same $p$",
                    "No: AICc values are comparable only between models with the same $d$"
                ],
                "correctExplanation": "With $d = 1$ the likelihood is computed for $\\Delta y_t$, a different series than $y_t$. Hyndman and Khandakar (2008) choose $d$ first (by KPSS tests), then compare $p$ and $q$ by AICc.",
                "incorrectExplanation": "AICc does compare different orders $p$ and $q$, and convergence is necessary but not sufficient; the problem is that the two likelihoods refer to different data."
            },
            "ro": {
                "title": "Selecția automată ARIMA",
                "text": "O căutare automată raportează AICc = 451 pentru ARIMA(1,1,0) și AICc = 456 pentru ARIMA(2,0,0) cu medie. Puteți alege ARIMA(1,1,0) pentru că are un AICc mai mic?",
                "options": [
                    "Da: cîștigă întotdeauna valoarea AICc mai mică",
                    "Da, dar numai dacă ambele estimări au convers",
                    "Nu: AICc poate compara doar modele cu același $p$",
                    "Nu: valorile AICc sînt comparabile doar între modele cu același $d$"
                ],
                "correctExplanation": "Cu $d = 1$, verosimilitatea se calculează pentru $\\Delta y_t$, o altă serie decît $y_t$. Hyndman și Khandakar (2008) aleg întîi $d$ (prin teste KPSS), apoi compară $p$ și $q$ după AICc.",
                "incorrectExplanation": "AICc compară ordine diferite $p$ și $q$, iar convergența este necesară, dar nu suficientă; problema este că cele două verosimilități se referă la date diferite."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Exchange-rate forecasts",
                "text": "What did Meese and Rogoff (1983) find about exchange-rate forecasts?",
                "options": [
                    "Out of sample, no model beat the random walk in root mean squared error",
                    "Monetary models clearly beat the random walk",
                    "ARIMA models were always the most accurate",
                    "The random walk was the worst model at short horizons"
                ],
                "correctExplanation": "Their comparison of structural and time series models for dollar exchange rates, at horizons of 1 to 12 months, made the random walk the benchmark for every exchange-rate forecast. For EUR/RON since 2024, an ARIMA(1,1,0) gains only a little, and mostly on a few days with jumps.",
                "incorrectExplanation": "The finding was the opposite: neither the monetary models nor the time series models outperformed the random walk out of sample."
            },
            "ro": {
                "title": "Prognoza cursurilor de schimb",
                "text": "Ce au constatat Meese și Rogoff (1983) despre prognozele cursurilor de schimb?",
                "options": [
                    "În afara eșantionului, niciun model nu a depășit mersul aleator ca rădăcină a erorii pătratice medii",
                    "Modelele monetare au depășit clar mersul aleator",
                    "Modelele ARIMA au fost întotdeauna cele mai precise",
                    "Mersul aleator a fost cel mai slab model pe orizonturi scurte"
                ],
                "correctExplanation": "Comparația lor între modele structurale și modele de serii de timp pentru cursurile dolarului, pe orizonturi de 1--12 luni, a făcut din mersul aleator reperul oricărei prognoze a cursului de schimb. Pentru EUR/RON din 2024, un ARIMA(1,1,0) cîștigă foarte puțin, mai ales în cîteva zile cu salturi.",
                "incorrectExplanation": "Rezultatul a fost opus: nici modelele monetare, nici modelele de serii de timp nu au depășit mersul aleator în afara eșantionului."
            }
        }
    ]
};
