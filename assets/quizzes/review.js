// ============================================================
// Chapter 15 quiz bank: Review and exam preparation (EN + RO)
// 24 questions from Chapters 0-10, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order
// (spread 6/6/6/6). incorrectExplanation does not name a letter:
// the engine prepends "The correct answer is X) ..." after shuffling.
// ============================================================
window.TSA_DATA.quizzes['review'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "Box–Jenkins workflow",
                "text": "What is the correct order of the Box–Jenkins workflow?",
                "options": [
                    "Plot and transform, test for unit roots, identify with ACF and PACF, estimate, check the residuals, forecast",
                    "Estimate several models, keep the best in-sample fit, then plot the data",
                    "Forecast first, then check the residuals of the forecast errors, then identify",
                    "Identify with ACF and PACF on the raw levels, estimate, and skip the residual checks if BIC is low"
                ],
                "correctExplanation": "The method starts from the data: plot, transform and decide the differences with ADF and KPSS; then identify with the correlogram of the stationary series, estimate, check the residuals and only then forecast.",
                "incorrectExplanation": "Estimating or forecasting before looking at the data reverses the logic, and the correlogram of non-stationary levels decays slowly whatever the model; a low BIC never replaces the residual checks."
            },
            "ro": {
                "title": "Etapele metodei Box–Jenkins",
                "text": "Care este ordinea corectă a etapelor metodei Box–Jenkins?",
                "options": [
                    "Grafic și transformare, teste de rădăcină unitară, identificare cu ACF și PACF, estimare, verificarea reziduurilor, prognoză",
                    "Estimăm mai multe modele, îl păstrăm pe cel mai bine potrivit în eșantion, apoi facem graficul datelor",
                    "Întîi prognoza, apoi verificarea reziduurilor din erorile de prognoză, apoi identificarea",
                    "Identificare cu ACF și PACF pe nivelurile brute, estimare și renunțarea la verificarea reziduurilor dacă BIC este mic"
                ],
                "correctExplanation": "Metoda pornește de la date: grafic, transformare și decizia asupra diferențierilor cu ADF și KPSS; apoi identificarea din corelograma seriei staționare, estimarea, verificarea reziduurilor și abia apoi prognoza.",
                "incorrectExplanation": "Estimarea sau prognoza înaintea examinării datelor inversează logica, iar corelograma nivelurilor nestaționare scade lent oricare ar fi modelul; un BIC mic nu înlocuiește niciodată verificarea reziduurilor."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ADF and KPSS together",
                "text": "ADF does not reject its null hypothesis and KPSS rejects its null hypothesis. What do you conclude?",
                "options": [
                    "Both point to stationarity: model the levels",
                    "Both point to a unit root: difference the series",
                    "The tests contradict each other: the result is inconclusive",
                    "The series is white noise"
                ],
                "correctExplanation": "ADF has a unit root under the null and KPSS has stationarity under the null; not rejecting the first and rejecting the second both indicate a unit root.",
                "incorrectExplanation": "The two tests have opposite null hypotheses. Here they agree: one keeps the unit root and the other rejects stationarity. They would be inconclusive if neither rejected or both rejected."
            },
            "ro": {
                "title": "ADF și KPSS împreună",
                "text": "ADF nu respinge ipoteza nulă, iar KPSS respinge ipoteza nulă. Ce concluzie trageți?",
                "options": [
                    "Ambele indică staționaritatea: modelăm nivelurile",
                    "Ambele indică o rădăcină unitară: diferențiem seria",
                    "Testele se contrazic: rezultatul este neconcludent",
                    "Seria este zgomot alb"
                ],
                "correctExplanation": "La ADF ipoteza nulă este rădăcina unitară, la KPSS staționaritatea; nerespingerea primei și respingerea celei de-a doua indică amîndouă o rădăcină unitară.",
                "incorrectExplanation": "Cele două teste au ipoteze nule opuse. Aici ele concordă: unul păstrează rădăcina unitară, celălalt respinge staționaritatea. Rezultatul ar fi neconcludent dacă niciunul nu ar respinge sau dacă ambele ar respinge."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Ljung–Box on residuals",
                "text": "The residuals of an ARMA(1,1) model give a Ljung–Box statistic Q(10). Which distribution do you compare it with?",
                "options": [
                    "χ² with 10 degrees of freedom",
                    "χ² with 2 degrees of freedom",
                    "χ² with 8 degrees of freedom",
                    "Student-t with 10 degrees of freedom"
                ],
                "correctExplanation": "On the residuals of a fitted ARMA(p,q) the statistic loses one degree of freedom for each estimated ARMA parameter: m − p − q = 10 − 2 = 8.",
                "incorrectExplanation": "Using m degrees of freedom makes the test too lenient on residuals; the statistic is a sum of squared autocorrelations and follows a χ², not a Student distribution."
            },
            "ro": {
                "title": "Ljung–Box pe reziduuri",
                "text": "Reziduurile unui model ARMA(1,1) dau statistica Ljung–Box Q(10). Cu ce distribuție o comparați?",
                "options": [
                    "χ² cu 10 grade de libertate",
                    "χ² cu 2 grade de libertate",
                    "χ² cu 8 grade de libertate",
                    "Student-t cu 10 grade de libertate"
                ],
                "correctExplanation": "Pe reziduurile unui ARMA(p,q) estimat, statistica pierde cîte un grad de libertate pentru fiecare parametru ARMA estimat: m − p − q = 10 − 2 = 8.",
                "incorrectExplanation": "Cu m grade de libertate testul este prea îngăduitor pe reziduuri; statistica este o sumă de pătrate de autocorelații și urmează o distribuție χ², nu Student."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Reading ACF and PACF",
                "text": "The ACF decays geometrically and the PACF cuts off after lag 2. Which model is suggested?",
                "options": [
                    "MA(2)",
                    "ARMA(2,2)",
                    "A random walk",
                    "AR(2)"
                ],
                "correctExplanation": "For an AR(p) the PACF is zero after lag p while the ACF decays; here p = 2.",
                "incorrectExplanation": "An MA(q) has the opposite pattern (the ACF cuts off); an ARMA has both decaying; a random walk has an ACF that stays close to 1 for many lags."
            },
            "ro": {
                "title": "Citirea ACF și PACF",
                "text": "ACF scade geometric, iar PACF se anulează după lagul 2. Ce model este sugerat?",
                "options": [
                    "MA(2)",
                    "ARMA(2,2)",
                    "Un mers aleator",
                    "AR(2)"
                ],
                "correctExplanation": "La un AR(p), PACF este zero după lagul p, iar ACF descrește; aici p = 2.",
                "incorrectExplanation": "Un MA(q) are tiparul opus (ACF se anulează); un ARMA are ambele funcții descrescătoare; un mers aleator are o ACF care rămîne aproape de 1 pe multe laguri."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "MA(1) autocorrelation",
                "text": "For an MA(1) process $X_t = \\varepsilon_t + 0.5\\,\\varepsilon_{t-1}$, what is $\\rho(1)$?",
                "options": [
                    "0.4",
                    "0.5",
                    "0.25",
                    "0.8"
                ],
                "correctExplanation": "ρ(1) = θ/(1 + θ²) = 0.5/1.25 = 0.4; all higher autocorrelations are zero.",
                "incorrectExplanation": "The autocorrelation is not θ itself: the variance of the MA(1) is σ²(1 + θ²), so θ is divided by 1 + θ²."
            },
            "ro": {
                "title": "Autocorelația unui MA(1)",
                "text": "Pentru procesul MA(1) $X_t = \\varepsilon_t + 0{,}5\\,\\varepsilon_{t-1}$, cît este $\\rho(1)$?",
                "options": [
                    "0,4",
                    "0,5",
                    "0,25",
                    "0,8"
                ],
                "correctExplanation": "ρ(1) = θ/(1 + θ²) = 0,5/1,25 = 0,4; toate autocorelațiile de ordin mai mare sînt zero.",
                "incorrectExplanation": "Autocorelația nu este chiar θ: varianța unui MA(1) este σ²(1 + θ²), deci θ se împarte la 1 + θ²."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Over-differencing",
                "text": "Which sign suggests that a series has been differenced once too often?",
                "options": [
                    "A slowly decaying ACF",
                    "A first autocorrelation near −0.5 and an estimated MA coefficient near −1",
                    "A large positive first autocorrelation",
                    "A significant seasonal spike at lag 12"
                ],
                "correctExplanation": "Differencing a stationary series creates a non-invertible MA component: the first autocorrelation moves towards −0.5, the variance rises and the MA coefficient approaches −1.",
                "incorrectExplanation": "A slowly decaying ACF signals too few differences, not too many; a seasonal spike calls for seasonal terms."
            },
            "ro": {
                "title": "Supradiferențierea",
                "text": "Ce semn arată că o serie a fost diferențiată o dată în plus?",
                "options": [
                    "O ACF care scade lent",
                    "O primă autocorelație în jur de −0,5 și un coeficient MA estimat aproape de −1",
                    "O primă autocorelație mare și pozitivă",
                    "O valoare sezonieră semnificativă la lagul 12"
                ],
                "correctExplanation": "Diferențierea unei serii staționare creează o componentă MA neinvertibilă: prima autocorelație se apropie de −0,5, varianța crește, iar coeficientul MA se apropie de −1.",
                "incorrectExplanation": "O ACF care scade lent arată prea puține diferențieri, nu prea multe; o valoare sezonieră mare cere termeni sezonieri."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "SARIMA forecast by hand",
                "text": "For $z_t = \\Delta\\Delta_{12}\\ln P_t$ an AR(1) × SAR(1) model is $(1 - \\phi L)(1 - \\Phi L^{12})z_t = \\varepsilon_t$. What is $\\hat z_{T+1}$?",
                "options": [
                    "$\\phi z_T + \\Phi z_{T-12}$",
                    "$\\phi z_{T+1} + \\Phi z_{T-11}$",
                    "$\\phi z_T + \\Phi z_{T-11} - \\phi\\Phi z_{T-12}$",
                    "$(\\phi + \\Phi) z_T$"
                ],
                "correctExplanation": "Multiplying the two polynomials gives $z_t = \\phi z_{t-1} + \\Phi z_{t-12} - \\phi\\Phi z_{t-13} + \\varepsilon_t$; at $t = T + 1$ the lags are $T$, $T - 11$ and $T - 12$.",
                "incorrectExplanation": "The cross term $\\phi\\Phi$ of the multiplicative model is easy to forget, and the seasonal lag of $T + 1$ is $T - 11$, not $T - 12$."
            },
            "ro": {
                "title": "Prognoză SARIMA de mînă",
                "text": "Pentru $z_t = \\Delta\\Delta_{12}\\ln P_t$, un model AR(1) × SAR(1) este $(1 - \\phi L)(1 - \\Phi L^{12})z_t = \\varepsilon_t$. Cît este $\\hat z_{T+1}$?",
                "options": [
                    "$\\phi z_T + \\Phi z_{T-12}$",
                    "$\\phi z_{T+1} + \\Phi z_{T-11}$",
                    "$\\phi z_T + \\Phi z_{T-11} - \\phi\\Phi z_{T-12}$",
                    "$(\\phi + \\Phi) z_T$"
                ],
                "correctExplanation": "Înmulțirea celor două polinoame dă $z_t = \\phi z_{t-1} + \\Phi z_{t-12} - \\phi\\Phi z_{t-13} + \\varepsilon_t$; la $t = T + 1$, lagurile sînt $T$, $T - 11$ și $T - 12$.",
                "incorrectExplanation": "Termenul încrucișat $\\phi\\Phi$ al modelului multiplicativ se uită ușor, iar lagul sezonier al lui $T + 1$ este $T - 11$, nu $T - 12$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Reading EViews output",
                "text": "An EViews output has the dependent variable DLOG(IPC,1,12). What is modelled?",
                "options": [
                    "$\\ln \\mathrm{IPC}_t$ in levels",
                    "$\\Delta\\ln \\mathrm{IPC}_t$ only",
                    "$\\Delta_{12}\\mathrm{IPC}_t$ without logarithms",
                    "$\\Delta\\Delta_{12}\\ln \\mathrm{IPC}_t$: the monthly change of annual inflation"
                ],
                "correctExplanation": "DLOG(X,d,s) takes the logarithm and applies d regular differences and one seasonal difference of period s: here one regular and one seasonal difference.",
                "incorrectExplanation": "Without the seasonal difference the series would be monthly inflation, and without the regular difference it would be annual inflation; DLOG always works on the logarithm."
            },
            "ro": {
                "title": "Citirea rezultatelor EViews",
                "text": "Un rezultat EViews are variabila dependentă DLOG(IPC,1,12). Ce se modelează?",
                "options": [
                    "$\\ln \\mathrm{IPC}_t$ în niveluri",
                    "doar $\\Delta\\ln \\mathrm{IPC}_t$",
                    "$\\Delta_{12}\\mathrm{IPC}_t$ fără logaritm",
                    "$\\Delta\\Delta_{12}\\ln \\mathrm{IPC}_t$: variația lunară a inflației anuale"
                ],
                "correctExplanation": "DLOG(X,d,s) ia logaritmul și aplică d diferențe obișnuite și o diferență sezonieră de perioadă s: aici o diferență obișnuită și una sezonieră.",
                "incorrectExplanation": "Fără diferența sezonieră seria ar fi inflația lunară, iar fără diferența obișnuită ar fi inflația anuală; DLOG lucrează întotdeauna pe logaritm."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Spurious regression",
                "text": "You regress one random walk on another, independent random walk. What do you typically find?",
                "options": [
                    "A large t statistic, a high $R^2$ and a Durbin–Watson statistic close to 0",
                    "A t statistic close to 0 and $R^2$ close to 0",
                    "White-noise residuals",
                    "A slope that converges to zero as T grows"
                ],
                "correctExplanation": "Granger and Newbold (1974): with I(1) series the t statistic grows with the sample, $R^2$ is high and the residuals are themselves a random walk (DW near 0).",
                "incorrectExplanation": "Independent random walks look related in levels; more data make the problem worse, not better. The remedy is to test the order of integration and cointegration."
            },
            "ro": {
                "title": "Regresia falsă",
                "text": "Regresați un mers aleator pe un alt mers aleator, independent. Ce obțineți de obicei?",
                "options": [
                    "O statistică t mare, un $R^2$ mare și o statistică Durbin–Watson apropiată de 0",
                    "O statistică t apropiată de 0 și un $R^2$ apropiat de 0",
                    "Reziduuri de tip zgomot alb",
                    "O pantă care tinde la zero cînd T crește"
                ],
                "correctExplanation": "Granger și Newbold (1974): pentru serii I(1), statistica t crește cu eșantionul, $R^2$ este mare, iar reziduurile sînt ele însele un mers aleator (DW aproape de 0).",
                "incorrectExplanation": "Mersurile aleatoare independente par legate în niveluri; mai multe date agravează problema. Remediul este testarea ordinului de integrare și a cointegrării."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Engle–Granger critical values",
                "text": "In the Engle–Granger test, the ADF statistic of the OLS residuals is compared with:",
                "options": [
                    "The usual Dickey–Fuller critical values",
                    "MacKinnon critical values, more negative than the Dickey–Fuller ones",
                    "Standard Normal critical values (−1.96)",
                    "Student-t critical values with T − 2 degrees of freedom"
                ],
                "correctExplanation": "OLS chooses the coefficients that make the residuals look as stationary as possible, so the critical values must be more negative and depend on the number of variables.",
                "incorrectExplanation": "Dickey–Fuller values would reject too often; Normal and Student values are wrong for any unit-root statistic."
            },
            "ro": {
                "title": "Valorile critice Engle–Granger",
                "text": "În testul Engle–Granger, statistica ADF a reziduurilor OLS se compară cu:",
                "options": [
                    "Valorile critice Dickey–Fuller obișnuite",
                    "Valorile critice MacKinnon, mai negative decît cele Dickey–Fuller",
                    "Valorile critice ale distribuției Normale (−1,96)",
                    "Valorile critice Student-t cu T − 2 grade de libertate"
                ],
                "correctExplanation": "OLS alege coeficienții care fac reziduurile să pară cît mai staționare, deci valorile critice trebuie să fie mai negative și depind de numărul de variabile.",
                "incorrectExplanation": "Valorile Dickey–Fuller ar respinge prea des; valorile Normale și Student sînt greșite pentru orice statistică de rădăcină unitară."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Adjustment coefficients in a VECM",
                "text": "In a VECM, the adjustment coefficient α of one variable is small and not significant. What does this mean?",
                "options": [
                    "The series are not cointegrated",
                    "The variable is stationary",
                    "That variable does not adjust to the equilibrium error: it is weakly exogenous",
                    "The long-run coefficient β is zero"
                ],
                "correctExplanation": "α measures how each variable reacts to last period's equilibrium error; a zero row of α means the variable drives the common trend and the others do the adjusting.",
                "incorrectExplanation": "Cointegration is decided by the rank of Π = αβ′, not by one coefficient; at least one other variable must adjust when the series are cointegrated."
            },
            "ro": {
                "title": "Coeficienții de ajustare dintr-un VECM",
                "text": "Într-un VECM, coeficientul de ajustare α al unei variabile este mic și nesemnificativ. Ce înseamnă acest lucru?",
                "options": [
                    "Seriile nu sînt cointegrate",
                    "Variabila este staționară",
                    "Variabila nu se ajustează la eroarea de echilibru: este slab exogenă",
                    "Coeficientul pe termen lung β este zero"
                ],
                "correctExplanation": "α măsoară reacția fiecărei variabile la eroarea de echilibru din perioada anterioară; un rînd nul al lui α înseamnă că variabila conduce trendul comun, iar celelalte se ajustează.",
                "incorrectExplanation": "Cointegrarea se decide prin rangul lui Π = αβ′, nu printr-un singur coeficient; cînd seriile sînt cointegrate, cel puțin o altă variabilă trebuie să se ajusteze."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Johansen trace test",
                "text": "With three variables, the trace test rejects r = 0, does not reject r ≤ 1. What is the cointegration rank?",
                "options": [
                    "r = 0",
                    "r = 2",
                    "r = 3: all variables are stationary",
                    "r = 1, with two common stochastic trends"
                ],
                "correctExplanation": "The test is sequential: stop at the first null that is not rejected; r = 1 cointegrating vector and n − r = 2 common trends.",
                "incorrectExplanation": "Rejecting r = 0 already excludes the rank 0; r = 3 would mean that every variable is stationary, which contradicts their being I(1)."
            },
            "ro": {
                "title": "Testul urmei Johansen",
                "text": "Cu trei variabile, testul urmei respinge r = 0 și nu respinge r ≤ 1. Care este rangul de cointegrare?",
                "options": [
                    "r = 0",
                    "r = 2",
                    "r = 3: toate variabilele sînt staționare",
                    "r = 1, cu două trenduri stochastice comune"
                ],
                "correctExplanation": "Testul este secvențial: ne oprim la prima ipoteză nulă care nu se respinge; un vector de cointegrare și n − r = 2 trenduri comune.",
                "incorrectExplanation": "Respingerea lui r = 0 exclude deja rangul 0; r = 3 ar însemna că fiecare variabilă este staționară, ceea ce contrazice faptul că sînt I(1)."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Meaning of Granger causality",
                "text": "Inflation Granger-causes the interbank rate (p = 0.01). What does this establish?",
                "options": [
                    "Past inflation improves the forecast of the interbank rate, given the past of the rate",
                    "Inflation causes the central bank to raise rates",
                    "The interbank rate does not affect inflation",
                    "The two series are cointegrated"
                ],
                "correctExplanation": "Granger causality is incremental predictability within the chosen information set; it says nothing about structural causation or about the reverse direction.",
                "incorrectExplanation": "A causal policy claim needs identification (Chapter 6, structural VAR); the reverse direction has its own test; cointegration is a different property of the levels."
            },
            "ro": {
                "title": "Semnificația cauzalității Granger",
                "text": "Inflația cauzează Granger dobînda interbancară (p = 0,01). Ce stabilește acest rezultat?",
                "options": [
                    "Inflația trecută îmbunătățește prognoza dobînzii, dată fiind istoria dobînzii",
                    "Inflația determină banca centrală să crească dobînzile",
                    "Dobînda interbancară nu influențează inflația",
                    "Cele două serii sînt cointegrate"
                ],
                "correctExplanation": "Cauzalitatea Granger înseamnă predictibilitate suplimentară în setul de informații ales; nu spune nimic despre cauzalitatea structurală sau despre direcția inversă.",
                "incorrectExplanation": "O afirmație cauzală despre politica monetară cere identificare (Capitolul 6, VAR structural); direcția inversă are propriul test; cointegrarea este o altă proprietate, a nivelurilor."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Granger F from two RSS",
                "text": "Restricted RSS = 110, unrestricted RSS = 100, p = 2 lags tested, 100 residual degrees of freedom. What is F?",
                "options": [
                    "10",
                    "5",
                    "0.05",
                    "1.1"
                ],
                "correctExplanation": "F = [(110 − 100)/2] / [100/100] = 5/1 = 5, compared with F(2, 100), whose 5% critical value is about 3.09.",
                "incorrectExplanation": "The numerator is divided by the number of restrictions and the denominator by its degrees of freedom; the ratio of the two RSS is not the F statistic."
            },
            "ro": {
                "title": "Testul Granger F din două RSS",
                "text": "RSS restricționat = 110, RSS nerestricționat = 100, p = 2 laguri testate, 100 de grade de libertate ale reziduurilor. Cît este F?",
                "options": [
                    "10",
                    "5",
                    "0,05",
                    "1,1"
                ],
                "correctExplanation": "F = [(110 − 100)/2] / [100/100] = 5/1 = 5, comparat cu F(2, 100), a cărui valoare critică la 5% este circa 3,09.",
                "incorrectExplanation": "Numărătorul se împarte la numărul de restricții, iar numitorul la gradele lui de libertate; raportul celor două RSS nu este statistica F."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "GARCH half-life",
                "text": "A GARCH(1,1) has α + β = 0.98. After how many days does a volatility shock halve?",
                "options": [
                    "50 days",
                    "About 2 days",
                    "About 34 days",
                    "Never"
                ],
                "correctExplanation": "The half-life is ln 0.5 / ln(α + β) = −0.693 / −0.0202 ≈ 34 days.",
                "incorrectExplanation": "1/(1 − α − β) = 50 is not a half-life; the shock does halve because α + β < 1; it would never fade only for α + β = 1 (IGARCH)."
            },
            "ro": {
                "title": "Timpul de înjumătățire GARCH",
                "text": "Un GARCH(1,1) are α + β = 0,98. După cîte zile se înjumătățește un șoc al volatilității?",
                "options": [
                    "După 50 de zile",
                    "După circa 2 zile",
                    "După circa 34 de zile",
                    "Niciodată"
                ],
                "correctExplanation": "Timpul de înjumătățire este ln 0,5 / ln(α + β) = −0,693 / −0,0202 ≈ 34 de zile.",
                "incorrectExplanation": "1/(1 − α − β) = 50 nu este un timp de înjumătățire; șocul se înjumătățește, pentru că α + β < 1; nu s-ar stinge niciodată doar pentru α + β = 1 (IGARCH)."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "GARCH long-run variance",
                "text": "A GARCH(1,1) for daily returns in % has ω = 0.02, α = 0.08, β = 0.90. What is the long-run daily variance?",
                "options": [
                    "0.02",
                    "0.2",
                    "It does not exist",
                    "1"
                ],
                "correctExplanation": "σ̄² = ω/(1 − α − β) = 0.02/0.02 = 1, a daily volatility of 1%, about 16% a year with 252 days.",
                "incorrectExplanation": "ω alone is not the long-run variance; the long-run variance exists because α + β = 0.98 < 1."
            },
            "ro": {
                "title": "Varianța pe termen lung GARCH",
                "text": "Un GARCH(1,1) pentru randamente zilnice în % are ω = 0,02, α = 0,08, β = 0,90. Cît este varianța zilnică pe termen lung?",
                "options": [
                    "0,02",
                    "0,2",
                    "Nu există",
                    "1"
                ],
                "correctExplanation": "σ̄² = ω/(1 − α − β) = 0,02/0,02 = 1, o volatilitate zilnică de 1%, circa 16% pe an, cu 252 de zile.",
                "incorrectExplanation": "ω singur nu este varianța pe termen lung; varianța pe termen lung există, pentru că α + β = 0,98 < 1."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "VaR convention",
                "text": "How does the course write the loss that is exceeded on 1% of days?",
                "options": [
                    "VaR 1%, a positive number: $\\mathrm{VaR}_{0.01} = -q_{0.01}$",
                    "VaR 99%, a negative number",
                    "VaR 99%, the 99% quantile of returns",
                    "VaR 1%, the 1% quantile of returns, a negative number"
                ],
                "correctExplanation": "The level is the tail probability and VaR is reported as a positive loss: minus the 1% quantile of the return distribution.",
                "incorrectExplanation": "Writing VaR 99% or reporting the raw (negative) quantile mixes two conventions; the course uses VaR 1% with a positive sign."
            },
            "ro": {
                "title": "Convenția VaR",
                "text": "Cum scriem în curs pierderea depășită în 1% din zile?",
                "options": [
                    "VaR 1%, un număr pozitiv: $\\mathrm{VaR}_{0{,}01} = -q_{0{,}01}$",
                    "VaR 99%, un număr negativ",
                    "VaR 99%, cuantila de 99% a randamentelor",
                    "VaR 1%, cuantila de 1% a randamentelor, un număr negativ"
                ],
                "correctExplanation": "Nivelul este probabilitatea cozii, iar VaR se raportează ca pierdere pozitivă: minus cuantila de 1% a distribuției randamentelor.",
                "incorrectExplanation": "Scrierea „VaR 99%” sau raportarea cuantilei brute (negative) amestecă două convenții; în curs folosim VaR 1%, cu semn pozitiv."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "MASE below one",
                "text": "A SARIMA forecast has MASE = 0.66 on the test sample. What does it mean?",
                "options": [
                    "It is 66% accurate",
                    "Its mean absolute error is 66% of that of the in-sample seasonal naive method",
                    "It beats the seasonal naive method significantly",
                    "Its errors are 0.66 percentage points"
                ],
                "correctExplanation": "MASE scales the MAE by the in-sample MAE of the seasonal naive method; below 1 the model is better on average, whatever the units of the series.",
                "incorrectExplanation": "MASE is not a share of correct forecasts and not a test: significance needs a Diebold–Mariano test; it has no units."
            },
            "ro": {
                "title": "MASE sub unu",
                "text": "O prognoză SARIMA are MASE = 0,66 pe eșantionul de test. Ce înseamnă?",
                "options": [
                    "Are o acuratețe de 66%",
                    "Eroarea ei absolută medie este 66% din cea a metodei naive sezoniere în eșantion",
                    "Bate semnificativ metoda naivă sezonieră",
                    "Erorile ei sînt de 0,66 puncte procentuale"
                ],
                "correctExplanation": "MASE împarte MAE la MAE din eșantion al metodei naive sezoniere; sub 1, modelul este mai bun în medie, oricare ar fi unitatea de măsură a seriei.",
                "incorrectExplanation": "MASE nu este o proporție de prognoze corecte și nici un test: semnificația cere testul Diebold–Mariano; MASE nu are unitate de măsură."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Diebold–Mariano test",
                "text": "Two forecasts on 24 test months: Diebold–Mariano (HLN) statistic −1.40, p = 0.17. What do you conclude?",
                "options": [
                    "The first forecast is significantly better",
                    "The second forecast is significantly better",
                    "Equal accuracy is not rejected: the gain may be due to chance",
                    "The forecasts are identical"
                ],
                "correctExplanation": "The test compares the mean loss difference with its standard error; p = 0.17 > 0.05, so 24 months do not show a real difference.",
                "incorrectExplanation": "A negative statistic favours the first forecast, but not significantly; not rejecting equal accuracy does not mean the forecasts are identical."
            },
            "ro": {
                "title": "Testul Diebold–Mariano",
                "text": "Două prognoze pe 24 de luni de test: statistica Diebold–Mariano (HLN) este −1,40, p = 0,17. Ce concluzie trageți?",
                "options": [
                    "Prima prognoză este semnificativ mai bună",
                    "A doua prognoză este semnificativ mai bună",
                    "Acuratețea egală nu se respinge: cîștigul poate fi întîmplător",
                    "Prognozele sînt identice"
                ],
                "correctExplanation": "Testul compară media diferențelor de pierdere cu eroarea ei standard; p = 0,17 > 0,05, deci 24 de luni nu arată o diferență reală.",
                "incorrectExplanation": "O statistică negativă favorizează prima prognoză, dar nu semnificativ; nerespingerea acurateței egale nu înseamnă că prognozele sînt identice."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Cross-validation for time series",
                "text": "How should a forecasting model of a time series be validated?",
                "options": [
                    "Random k-fold cross-validation",
                    "Leave-one-out on randomly chosen days",
                    "On the training sample, with the highest $R^2$",
                    "Rolling origins: train on the past, test on the following block, move forward"
                ],
                "correctExplanation": "Walk-forward (rolling-origin) validation respects the order of time, so no information from the future enters the estimation.",
                "incorrectExplanation": "Random folds put future observations into the training set (leakage) and produce spectacular but false scores; in-sample $R^2$ says nothing about forecasting."
            },
            "ro": {
                "title": "Validarea încrucișată pentru serii de timp",
                "text": "Cum se validează un model de prognoză pentru o serie de timp?",
                "options": [
                    "Validare încrucișată aleatoare în k grupuri",
                    "Leave-one-out pe zile alese aleator",
                    "Pe eșantionul de antrenare, după cel mai mare $R^2$",
                    "Cu origini mobile: antrenare pe trecut, test pe blocul următor, apoi mai departe"
                ],
                "correctExplanation": "Validarea walk-forward (cu origini mobile) respectă ordinea timpului, deci nicio informație din viitor nu intră în estimare.",
                "incorrectExplanation": "Grupurile aleatoare pun observații din viitor în setul de antrenare (leakage) și dau scoruri spectaculoase, dar false; $R^2$ în eșantion nu spune nimic despre prognoză."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "ARFIMA with d = 0.3",
                "text": "An ARFIMA(0,d,0) has d = 0.3. Which statement is correct?",
                "options": [
                    "It is stationary with long memory; H = 0.8",
                    "It is non-stationary; H = 0.3",
                    "It has short memory like an AR(1); H = 0.5",
                    "It is not mean-reverting"
                ],
                "correctExplanation": "For 0 < d < 1/2 the process is stationary, its ACF decays hyperbolically, and H = d + 1/2 = 0.8; it is mean-reverting since d < 1.",
                "incorrectExplanation": "Non-stationarity starts at d = 1/2 and the loss of mean reversion at d = 1; an AR(1) has an exponentially decaying ACF."
            },
            "ro": {
                "title": "ARFIMA cu d = 0,3",
                "text": "Un ARFIMA(0,d,0) are d = 0,3. Care afirmație este corectă?",
                "options": [
                    "Este staționar, cu memorie lungă; H = 0,8",
                    "Este nestaționar; H = 0,3",
                    "Are memorie scurtă, ca un AR(1); H = 0,5",
                    "Nu revine la medie"
                ],
                "correctExplanation": "Pentru 0 < d < 1/2 procesul este staționar, ACF scade hiperbolic, iar H = d + 1/2 = 0,8; revine la medie, pentru că d < 1.",
                "incorrectExplanation": "Nestaționaritatea începe la d = 1/2, iar pierderea revenirii la medie la d = 1; un AR(1) are o ACF care scade exponențial."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Kalman filter and smoothing",
                "text": "In the local level model, what is the steady-state Kalman gain $\\bar K$?",
                "options": [
                    "The variance of the measurement noise",
                    "The weight α of simple exponential smoothing",
                    "The probability of a regime change",
                    "The AR coefficient of the level"
                ],
                "correctExplanation": "In the steady state the update is $a_{t+1} = a_t + \\bar K(y_t - a_t)$, which is the SES recursion with α = $\\bar K$ (Muth, 1960).",
                "incorrectExplanation": "The gain is a weight between 0 and 1, not a variance or a probability; the level in this model is a random walk, with no AR coefficient."
            },
            "ro": {
                "title": "Filtrul Kalman și netezirea",
                "text": "În modelul local level, ce este cîștigul Kalman de echilibru $\\bar K$?",
                "options": [
                    "Varianța zgomotului de măsurare",
                    "Ponderea α a netezirii exponențiale simple",
                    "Probabilitatea unei schimbări de regim",
                    "Coeficientul AR al nivelului"
                ],
                "correctExplanation": "În starea de echilibru actualizarea este $a_{t+1} = a_t + \\bar K(y_t - a_t)$, adică recurența SES cu α = $\\bar K$ (Muth, 1960).",
                "incorrectExplanation": "Cîștigul este o pondere între 0 și 1, nu o varianță și nici o probabilitate; nivelul din acest model este un mers aleator, fără coeficient AR."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Markov switching durations",
                "text": "In a two-regime Markov switching model, $p_{11} = 0.9$. What is the expected duration of regime 1?",
                "options": [
                    "9 periods",
                    "0.9 periods",
                    "10 periods",
                    "1.1 periods"
                ],
                "correctExplanation": "The duration is geometric with mean 1/(1 − p₁₁) = 1/0.1 = 10 periods.",
                "incorrectExplanation": "p₁₁/(1 − p₁₁) = 9 counts only the extra periods after the first; 0.9 is a probability, not a duration."
            },
            "ro": {
                "title": "Durata regimurilor Markov switching",
                "text": "Într-un model Markov switching cu două regimuri, $p_{11} = 0{,}9$. Cît durează în medie regimul 1?",
                "options": [
                    "9 perioade",
                    "0,9 perioade",
                    "10 perioade",
                    "1,1 perioade"
                ],
                "correctExplanation": "Durata este geometrică, cu media 1/(1 − p₁₁) = 1/0,1 = 10 perioade.",
                "incorrectExplanation": "p₁₁/(1 − p₁₁) = 9 numără doar perioadele de după prima; 0,9 este o probabilitate, nu o durată."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "AIC and BIC",
                "text": "Compared with AIC, BIC typically selects:",
                "options": [
                    "Larger models",
                    "Exactly the same models",
                    "Models with a better in-sample fit",
                    "Smaller, more parsimonious models"
                ],
                "correctExplanation": "BIC penalises each parameter with ln T instead of 2, so for T > 7 it prefers fewer parameters; it is consistent, while AIC targets forecasting accuracy.",
                "incorrectExplanation": "The heavier penalty of BIC works against larger models and against in-sample fit; the two criteria often disagree."
            },
            "ro": {
                "title": "AIC și BIC",
                "text": "În comparație cu AIC, BIC alege de obicei:",
                "options": [
                    "Modele mai mari",
                    "Exact aceleași modele",
                    "Modele cu o potrivire mai bună în eșantion",
                    "Modele mai mici, mai parcimonioase"
                ],
                "correctExplanation": "BIC penalizează fiecare parametru cu ln T în loc de 2, deci pentru T > 7 preferă mai puțini parametri; este consistent, în timp ce AIC țintește acuratețea prognozei.",
                "incorrectExplanation": "Penalizarea mai mare a BIC defavorizează modelele mari și potrivirea în eșantion; cele două criterii nu concordă adesea."
            }
        }
    ]
};
