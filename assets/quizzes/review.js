// ============================================================
// Chapter 15 quiz bank: Review and exam preparation (EN + RO)
// 19 questions ported from the 2025/2026 site; 19 drawn per attempt.
// The bank grows to 24 questions when the chapter is rebuilt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['review'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Analysis workflow",
                "text": "What is the correct order of the time series analysis workflow?",
                "options": [
                    "Model estimation, data exploration, diagnostics, forecasting",
                    "Data exploration, stationarity testing, model selection and estimation, diagnostics, forecasting",
                    "Forecasting, data exploration, model estimation, diagnostics",
                    "Diagnostics, model selection, data exploration, forecasting"
                ],
                "correctExplanation": "A systematic workflow runs: explore and plot the data, transform and test for stationarity, select and estimate candidate models, check the residual diagnostics, validate out of sample and only then forecast.",
                "incorrectExplanation": "Estimating or diagnosing a model before looking at the data, or forecasting before checking the model, reverses the logic: diagnostics can only be run on an estimated model, and the model choice depends on what the exploration and stationarity tests reveal. The workflow starts with the data."
            },
            "ro": {
                "title": "Etapele analizei",
                "text": "Care este ordinea corectă a etapelor în analiza unei serii de timp?",
                "options": [
                    "Estimarea modelului, explorarea datelor, diagnosticarea, prognoza",
                    "Explorarea datelor, testarea staționarității, alegerea și estimarea modelului, diagnosticarea, prognoza",
                    "Prognoza, explorarea datelor, estimarea modelului, diagnosticarea",
                    "Diagnosticarea, alegerea modelului, explorarea datelor, prognoza"
                ],
                "correctExplanation": "O analiză sistematică urmează pașii: explorarea și reprezentarea grafică a datelor, transformarea și testarea staționarității, alegerea și estimarea modelelor candidate, verificarea reziduurilor, validarea în afara eșantionului și abia apoi prognoza.",
                "incorrectExplanation": "Estimarea sau diagnosticarea unui model înainte de examinarea datelor, ori prognoza înainte de verificarea modelului, inversează logica: diagnosticarea se poate face doar pe un model estimat, iar alegerea modelului depinde de ce arată explorarea datelor și testele de staționaritate. Analiza începe cu datele."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "RMSE much larger than MAE",
                "text": "If RMSE is much larger than MAE, this suggests that:",
                "options": [
                    "The model is overfitting",
                    "There are a few large errors",
                    "The model is underfitting",
                    "The data are stationary"
                ],
                "correctExplanation": "Because RMSE squares the errors, it is pulled up by a few large ones, while MAE weighs all errors linearly. Always $\\text{RMSE} \\geq \\text{MAE}$, and a large ratio $\\text{RMSE}/\\text{MAE}$ points to a heavy-tailed error distribution with some very large misses.",
                "incorrectExplanation": "The RMSE/MAE ratio describes the shape of the error distribution, not whether the model over- or underfits, and it says nothing about stationarity. A large gap means a few errors are much larger than the rest."
            },
            "ro": {
                "title": "RMSE mult mai mare decît MAE",
                "text": "Dacă RMSE este mult mai mare decît MAE, aceasta sugerează că:",
                "options": [
                    "Modelul este supraajustat (overfitting)",
                    "Există cîteva erori mari",
                    "Modelul este subajustat (underfitting)",
                    "Datele sînt staționare"
                ],
                "correctExplanation": "Deoarece RMSE ridică erorile la pătrat, cîteva erori mari o cresc puternic, în timp ce MAE ponderează toate erorile liniar. Întotdeauna $\\text{RMSE} \\geq \\text{MAE}$, iar un raport $\\text{RMSE}/\\text{MAE}$ mare indică o distribuție a erorilor cu cozi groase, cu cîteva erori foarte mari.",
                "incorrectExplanation": "Raportul RMSE/MAE descrie forma distribuției erorilor, nu supraajustarea sau subajustarea modelului, și nu spune nimic despre staționaritate. O diferență mare înseamnă că cîteva erori sînt mult mai mari decît celelalte."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Limitation of MAPE",
                "text": "MAPE is problematic when:",
                "options": [
                    "The data have a trend",
                    "The actual values are close to or equal to zero",
                    "The forecast horizon is long",
                    "Several models are being compared"
                ],
                "correctExplanation": "$\\text{MAPE} = \\frac{100}{n}\\sum_t\\left|\\frac{y_t - \\hat{y}_t}{y_t}\\right|$. Dividing by small $y_t$ produces extreme values, and $y_t = 0$ makes it undefined.",
                "incorrectExplanation": "A trend, a long horizon or the comparison of several models are not problems for MAPE as such; comparing models is in fact what it is used for. Its weakness is the division by actual values close to zero."
            },
            "ro": {
                "title": "Limita indicatorului MAPE",
                "text": "MAPE este problematic atunci cînd:",
                "options": [
                    "Datele au trend",
                    "Valorile efective sînt apropiate de zero sau egale cu zero",
                    "Orizontul de prognoză este lung",
                    "Se compară mai multe modele"
                ],
                "correctExplanation": "$\\text{MAPE} = \\frac{100}{n}\\sum_t\\left|\\frac{y_t - \\hat{y}_t}{y_t}\\right|$. Împărțirea la valori $y_t$ mici produce valori extreme, iar pentru $y_t = 0$ indicatorul nu este definit.",
                "incorrectExplanation": "Trendul, orizontul lung sau compararea mai multor modele nu sînt probleme pentru MAPE ca atare; compararea modelelor este chiar scopul lui. Slăbiciunea lui este împărțirea la valori efective apropiate de zero."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Volatility clustering",
                "text": "S&P 500 returns show periods of high volatility followed by further high volatility. Which model should you consider?",
                "options": [
                    "An ARIMA model of higher order",
                    "Exponential smoothing",
                    "A GARCH model for the conditional variance",
                    "Seasonal differencing"
                ],
                "correctExplanation": "Volatility clustering is a stylised fact of financial returns. GARCH(1,1) models it through $\\sigma_t^2 = \\omega + \\alpha \\varepsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$, so large shocks raise future conditional variance.",
                "incorrectExplanation": "ARIMA models and exponential smoothing describe the conditional mean, not the conditional variance, and seasonal differencing removes seasonal patterns. Time-varying volatility calls for a GARCH-type model."
            },
            "ro": {
                "title": "Volatility clustering",
                "text": "Randamentele S&P 500 prezintă perioade de volatilitate ridicată urmate de alte perioade de volatilitate ridicată. Ce model trebuie luat în considerare?",
                "options": [
                    "Un model ARIMA de ordin mai mare",
                    "Netezirea exponențială",
                    "Un model GARCH pentru varianța condiționată",
                    "Diferențierea sezonieră"
                ],
                "correctExplanation": "Volatility clustering este un fapt stilizat al randamentelor financiare. GARCH(1,1) îl modelează prin $\\sigma_t^2 = \\omega + \\alpha \\varepsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$, astfel încît șocurile mari cresc varianța condiționată viitoare.",
                "incorrectExplanation": "Modelele ARIMA și netezirea exponențială descriu media condiționată, nu varianța condiționată, iar diferențierea sezonieră elimină tiparele sezoniere. Volatilitatea variabilă în timp cere un model de tip GARCH."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Combining ADF and KPSS",
                "text": "Both tests include a constant and a linear trend. If the ADF test does not reject its null and the KPSS test rejects its null, the series is most likely:",
                "options": [
                    "Stationary",
                    "Non-stationary (unit root)",
                    "Trend-stationary",
                    "White noise"
                ],
                "correctExplanation": "ADF: $H_0$ unit root; KPSS: $H_0$ stationarity (here around a trend). Failing to reject the unit root and rejecting stationarity point in the same direction: the series has a unit root.",
                "incorrectExplanation": "Stationarity, trend stationarity or white noise would make KPSS (with trend) unlikely to reject and ADF likely to reject. The two results agree on a unit root."
            },
            "ro": {
                "title": "Combinarea testelor ADF și KPSS",
                "text": "Ambele teste includ o constantă și un trend liniar. Dacă testul ADF nu își respinge ipoteza nulă, iar testul KPSS și-o respinge, seria este cel mai probabil:",
                "options": [
                    "Staționară",
                    "Nestaționară (cu rădăcină unitară)",
                    "Staționară în jurul unui trend",
                    "Zgomot alb"
                ],
                "correctExplanation": "ADF: $H_0$ rădăcină unitară; KPSS: $H_0$ staționaritate (aici în jurul unui trend). Nerespingerea rădăcinii unitare și respingerea staționarității indică același lucru: seria are rădăcină unitară.",
                "incorrectExplanation": "Dacă seria ar fi staționară, staționară în jurul unui trend sau zgomot alb, KPSS (cu trend) ar respinge greu, iar ADF ar respinge probabil. Cele două rezultate concordă asupra existenței unei rădăcini unitare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Multiplicative seasonality",
                "text": "The airline passengers series shows a seasonal amplitude that increases over time. Which decomposition is appropriate?",
                "options": [
                    "Additive: $Y_t = T_t + S_t + R_t$",
                    "Multiplicative: $Y_t = T_t \\times S_t \\times R_t$",
                    "Both work equally well",
                    "Neither; differencing should be used instead"
                ],
                "correctExplanation": "When the seasonal amplitude grows in proportion to the level, the multiplicative form is appropriate. Taking logarithms turns it into an additive decomposition.",
                "incorrectExplanation": "The additive form assumes seasonal swings of constant size, so it does not fit growing swings, and the two forms are therefore not equivalent here. Differencing is a modelling tool, not a decomposition. Proportional seasonal effects call for the multiplicative form."
            },
            "ro": {
                "title": "Sezonalitate multiplicativă",
                "text": "Seria numărului de pasageri ai companiilor aeriene are o amplitudine sezonieră care crește în timp. Ce descompunere este potrivită?",
                "options": [
                    "Aditivă: $Y_t = T_t + S_t + R_t$",
                    "Multiplicativă: $Y_t = T_t \\times S_t \\times R_t$",
                    "Ambele funcționează la fel de bine",
                    "Niciuna; trebuie folosită diferențierea"
                ],
                "correctExplanation": "Cînd amplitudinea sezonieră crește proporțional cu nivelul, forma multiplicativă este cea potrivită. Prin logaritmare, ea devine o descompunere aditivă.",
                "incorrectExplanation": "Forma aditivă presupune oscilații sezoniere de mărime constantă, deci nu se potrivește unor oscilații crescătoare, iar cele două forme nu sînt echivalente aici. Diferențierea este un instrument de modelare, nu o descompunere. Efectele sezoniere proporționale cer forma multiplicativă."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "SARIMA for monthly data",
                "text": "For monthly data with yearly seasonality, the seasonal period $m$ in SARIMA is:",
                "options": [
                    "4",
                    "7",
                    "12",
                    "52"
                ],
                "correctExplanation": "A yearly pattern in monthly data repeats every 12 observations, so $m = 12$. For comparison: quarterly data $m = 4$, weekly data $m = 52$, daily data with a weekly pattern $m = 7$.",
                "incorrectExplanation": "The values 4, 7 and 52 correspond to quarterly data, daily data with a weekly pattern and weekly data with a yearly pattern. Monthly data with yearly seasonality have $m = 12$."
            },
            "ro": {
                "title": "SARIMA pentru date lunare",
                "text": "Pentru date lunare cu sezonalitate anuală, perioada sezonieră $m$ din SARIMA este:",
                "options": [
                    "4",
                    "7",
                    "12",
                    "52"
                ],
                "correctExplanation": "Un tipar anual în date lunare se repetă la fiecare 12 observații, deci $m = 12$. Pentru comparație: date trimestriale $m = 4$, date săptămînale $m = 52$, date zilnice cu tipar săptămînal $m = 7$.",
                "incorrectExplanation": "Valorile 4, 7 și 52 corespund datelor trimestriale, datelor zilnice cu tipar săptămînal și datelor săptămînale cu tipar anual. Datele lunare cu sezonalitate anuală au $m = 12$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Comparing models across scales",
                "text": "When comparing SARIMA and Prophet forecasts across series measured in different units, which metric is scale-independent?",
                "options": [
                    "RMSE",
                    "MAE",
                    "MAPE",
                    "MSE"
                ],
                "correctExplanation": "MAPE expresses errors as percentages of the actual values, so it can be compared across series and data sets with different scales (provided the actual values are not close to zero).",
                "incorrectExplanation": "RMSE, MAE and MSE are expressed in the units of the series (or their square), so they change when the data are rescaled. MAPE is the unit-free choice among these four."
            },
            "ro": {
                "title": "Compararea modelelor pe scale diferite",
                "text": "La compararea prognozelor SARIMA și Prophet pe serii măsurate în unități diferite, ce indicator este independent de scală?",
                "options": [
                    "RMSE",
                    "MAE",
                    "MAPE",
                    "MSE"
                ],
                "correctExplanation": "MAPE exprimă erorile ca procente din valorile efective, deci poate fi comparată între serii și seturi de date cu scale diferite (cu condiția ca valorile efective să nu fie apropiate de zero).",
                "incorrectExplanation": "RMSE, MAE și MSE se exprimă în unitățile seriei (sau în pătratul lor), deci se modifică dacă datele sînt rescalate. Dintre cei patru indicatori, MAPE este cel fără unitate de măsură."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Structural breaks in Prophet",
                "text": "US retail sales experienced a structural break during COVID-19. Prophet handles such a break through:",
                "options": [
                    "Automatic differencing",
                    "Changepoints in the trend",
                    "Seasonal adjustment",
                    "GARCH modelling"
                ],
                "correctExplanation": "Prophet's piecewise trend lets the growth rate change at a set of changepoints, with a sparsity prior on the size of the changes, so it can adapt to breaks such as the COVID-19 shock.",
                "incorrectExplanation": "Prophet does not difference the series, its seasonal terms model recurring patterns rather than breaks, and GARCH models volatility. Breaks in the trend are captured by changepoints."
            },
            "ro": {
                "title": "Rupturi structurale în Prophet",
                "text": "Vînzările cu amănuntul din SUA au suferit o ruptură structurală în timpul pandemiei de COVID-19. Prophet tratează o astfel de ruptură prin:",
                "options": [
                    "Diferențiere automată",
                    "Puncte de schimbare (changepoints) ale trendului",
                    "Ajustare sezonieră",
                    "Modelare GARCH"
                ],
                "correctExplanation": "Trendul pe porțiuni din Prophet permite modificarea ratei de creștere într-un set de puncte de schimbare (changepoints), cu o distribuție a priori care favorizează puține modificări, așa că se poate adapta unor rupturi precum șocul COVID-19.",
                "incorrectExplanation": "Prophet nu diferențiază seria, termenii lui sezonieri modelează tipare recurente, nu rupturi, iar GARCH modelează volatilitatea. Rupturile de trend sînt captate prin changepoints."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Ljung-Box test",
                "text": "After fitting an ARIMA model, the Ljung-Box test on the residuals checks for:",
                "options": [
                    "Normality",
                    "Remaining autocorrelation",
                    "Heteroscedasticity",
                    "Stationarity"
                ],
                "correctExplanation": "The Ljung-Box statistic $Q = n(n+2)\\sum_{k=1}^{h}\\hat{\\rho}_k^2/(n-k)$ tests whether the first $h$ residual autocorrelations are jointly zero. Rejection means the model has left linear structure in the residuals.",
                "incorrectExplanation": "Normality is checked with tests such as Jarque-Bera, heteroscedasticity with ARCH-LM (or Ljung-Box on squared residuals), and stationarity with ADF or KPSS. Ljung-Box on the residuals targets remaining autocorrelation."
            },
            "ro": {
                "title": "Testul Ljung-Box",
                "text": "După estimarea unui model ARIMA, testul Ljung-Box aplicat reziduurilor verifică existența:",
                "options": [
                    "Normalității",
                    "Autocorelației reziduale",
                    "Heteroscedasticității",
                    "Staționarității"
                ],
                "correctExplanation": "Statistica Ljung-Box $Q = n(n+2)\\sum_{k=1}^{h}\\hat{\\rho}_k^2/(n-k)$ testează dacă primele $h$ autocorelații ale reziduurilor sînt simultan nule. Respingerea arată că modelul a lăsat structură liniară în reziduuri.",
                "incorrectExplanation": "Normalitatea se verifică prin teste precum Jarque-Bera, heteroscedasticitatea prin ARCH-LM (sau Ljung-Box pe pătratele reziduurilor), iar staționaritatea prin ADF sau KPSS. Testul Ljung-Box pe reziduuri vizează autocorelația rămasă."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Interpreting the ACF and PACF",
                "text": "The ACF decays geometrically and the PACF cuts off after lag 2. This suggests:",
                "options": [
                    "MA(2)",
                    "AR(2)",
                    "ARMA(2,2)",
                    "A random walk"
                ],
                "correctExplanation": "A PACF cut-off gives the AR order and an ACF cut-off gives the MA order. A decaying ACF together with a PACF that cuts off at lag 2 identifies AR(2).",
                "incorrectExplanation": "MA(2) would show the cut-off in the ACF, ARMA(2,2) would show decay in both functions, and a random walk would have an ACF that stays close to 1 over many lags. The pattern described is that of AR(2)."
            },
            "ro": {
                "title": "Interpretarea ACF și PACF",
                "text": "ACF scade geometric, iar PACF se anulează după lag-ul 2. Ce model sugerează acest tipar?",
                "options": [
                    "MA(2)",
                    "AR(2)",
                    "ARMA(2,2)",
                    "Un mers aleator"
                ],
                "correctExplanation": "Anularea PACF dă ordinul AR, iar anularea ACF dă ordinul MA. O ACF descrescătoare împreună cu o PACF care se anulează după lag-ul 2 identifică un AR(2).",
                "incorrectExplanation": "Un MA(2) ar avea anularea în ACF, un ARMA(2,2) ar avea scădere treptată în ambele funcții, iar un mers aleator ar avea o ACF apropiată de 1 pe multe lag-uri. Tiparul descris este cel al unui AR(2)."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Why use log returns",
                "text": "For the S&P 500 we model log returns $r_t = \\ln(P_t/P_{t-1})$ rather than prices because:",
                "options": [
                    "Prices are always stationary",
                    "Returns are approximately stationary, while prices are not",
                    "Log returns are easier to compute",
                    "Returns have stronger autocorrelation"
                ],
                "correctExplanation": "Log prices behave like a random walk (non-stationary). Their first difference, the log return, is approximately stationary and therefore suitable for ARMA and GARCH modelling.",
                "incorrectExplanation": "Prices are not stationary, ease of computation is not the reason, and returns in fact have much weaker autocorrelation than prices. The motivation is stationarity."
            },
            "ro": {
                "title": "De ce folosim randamente logaritmice",
                "text": "Pentru S&P 500 modelăm randamentele logaritmice $r_t = \\ln(P_t/P_{t-1})$ în locul prețurilor deoarece:",
                "options": [
                    "Prețurile sînt întotdeauna staționare",
                    "Randamentele sînt aproximativ staționare, iar prețurile nu",
                    "Randamentele logaritmice se calculează mai ușor",
                    "Randamentele au autocorelație mai puternică"
                ],
                "correctExplanation": "Logaritmul prețului se comportă ca un mers aleator (nestaționar). Prima lui diferență, randamentul logaritmic, este aproximativ staționară și deci potrivită pentru modelarea ARMA și GARCH.",
                "incorrectExplanation": "Prețurile nu sînt staționare, ușurința calculului nu este motivul, iar randamentele au de fapt autocorelații mult mai slabe decît prețurile. Motivul este staționaritatea."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "AIC and BIC",
                "text": "Compared with AIC, BIC typically selects:",
                "options": [
                    "More complex models",
                    "Simpler (more parsimonious) models",
                    "Exactly the same models as AIC",
                    "Models with a better in-sample fit"
                ],
                "correctExplanation": "$\\text{BIC} = -2\\ln L + k\\ln n$ and $\\text{AIC} = -2\\ln L + 2k$. For $n \\geq 8$, $\\ln n > 2$, so BIC penalises each extra parameter more heavily and favours smaller models.",
                "incorrectExplanation": "The heavier penalty pushes BIC towards fewer parameters, not more, so the two criteria often disagree, and the larger models preferred by AIC typically have the better in-sample fit. BIC selects more parsimonious models."
            },
            "ro": {
                "title": "AIC și BIC",
                "text": "Comparativ cu AIC, BIC selectează de regulă:",
                "options": [
                    "Modele mai complexe",
                    "Modele mai simple (mai parcimonioase)",
                    "Exact aceleași modele ca AIC",
                    "Modele cu o ajustare mai bună în eșantion"
                ],
                "correctExplanation": "$\\text{BIC} = -2\\ln L + k\\ln n$, iar $\\text{AIC} = -2\\ln L + 2k$. Pentru $n \\geq 8$, $\\ln n > 2$, deci BIC penalizează mai sever fiecare parametru suplimentar și favorizează modelele mai mici.",
                "incorrectExplanation": "Penalizarea mai severă împinge BIC spre mai puțini parametri, nu spre mai mulți, deci cele două criterii diferă adesea, iar modelele mai mari preferate de AIC au de regulă o ajustare mai bună în eșantion. BIC selectează modele mai parcimonioase."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Cross-validation for time series",
                "text": "For time series cross-validation we use:",
                "options": [
                    "Random k-fold cross-validation",
                    "Leave-one-out cross-validation",
                    "Rolling-origin (expanding-window) cross-validation",
                    "Stratified cross-validation"
                ],
                "correctExplanation": "Rolling-origin validation estimates the model on data up to time $t$, forecasts the following observations, then moves the origin forward. Training data always precede the evaluation data.",
                "incorrectExplanation": "Random k-fold, leave-one-out and stratified schemes mix past and future observations, so the model is trained on data that come after the points it is evaluated on. Time series validation must keep the chronological order."
            },
            "ro": {
                "title": "Validarea încrucișată pentru serii de timp",
                "text": "Pentru validarea încrucișată a seriilor de timp folosim:",
                "options": [
                    "Validarea încrucișată k-fold aleatoare",
                    "Validarea încrucișată leave-one-out",
                    "Validarea cu origine mobilă (fereastră extinsă)",
                    "Validarea încrucișată stratificată"
                ],
                "correctExplanation": "Validarea cu origine mobilă (rolling origin) estimează modelul pe datele pînă la momentul $t$, prognozează observațiile următoare, apoi mută originea înainte. Datele de antrenare preced întotdeauna datele de evaluare.",
                "incorrectExplanation": "Schemele k-fold aleatoare, leave-one-out și stratificate amestecă observațiile trecute cu cele viitoare, deci modelul este antrenat pe date ulterioare punctelor pe care este evaluat. Validarea pentru serii de timp trebuie să păstreze ordinea cronologică."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Multiple seasonality",
                "text": "Hourly data with daily, weekly and yearly patterns are best handled by:",
                "options": [
                    "SARIMA with $m = 24$",
                    "Simple exponential smoothing",
                    "TBATS or Prophet",
                    "ARIMA with differencing"
                ],
                "correctExplanation": "Standard SARIMA has a single seasonal period. TBATS (with trigonometric seasonal terms) and Prophet (with Fourier terms for each period) can include several seasonal cycles at once.",
                "incorrectExplanation": "SARIMA with $m = 24$ captures only the daily cycle, simple exponential smoothing has no seasonal component at all, and ordinary differencing does not model seasonality. Several seasonal periods require TBATS, Prophet or similar models."
            },
            "ro": {
                "title": "Sezonalitate multiplă",
                "text": "Datele orare cu tipare zilnice, săptămînale și anuale sînt modelate cel mai bine cu:",
                "options": [
                    "SARIMA cu $m = 24$",
                    "Netezirea exponențială simplă",
                    "TBATS sau Prophet",
                    "ARIMA cu diferențiere"
                ],
                "correctExplanation": "SARIMA standard are o singură perioadă sezonieră. TBATS (cu termeni sezonieri trigonometrici) și Prophet (cu termeni Fourier pentru fiecare perioadă) pot include simultan mai multe cicluri sezoniere.",
                "incorrectExplanation": "SARIMA cu $m = 24$ captează doar ciclul zilnic, netezirea exponențială simplă nu are deloc componentă sezonieră, iar diferențierea obișnuită nu modelează sezonalitatea. Mai multe perioade sezoniere cer TBATS, Prophet sau modele similare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Persistence in GARCH",
                "text": "In GARCH(1,1), high volatility persistence means that $\\alpha + \\beta$ is:",
                "options": [
                    "Close to 0",
                    "Close to 1",
                    "Greater than 1",
                    "Negative"
                ],
                "correctExplanation": "$\\alpha + \\beta$ measures persistence: the effect of a shock on future conditional variance decays like $(\\alpha + \\beta)^h$. Values close to 1 mean slow decay; $\\alpha + \\beta < 1$ is required for covariance stationarity.",
                "incorrectExplanation": "A value close to 0 means shocks die out almost immediately, a value above 1 implies an explosive, non-stationary variance, and negative values are ruled out by $\\alpha, \\beta \\geq 0$. High persistence means $\\alpha + \\beta$ just below 1."
            },
            "ro": {
                "title": "Persistența în GARCH",
                "text": "În GARCH(1,1), o persistență ridicată a volatilității înseamnă că $\\alpha + \\beta$ este:",
                "options": [
                    "Apropiat de 0",
                    "Apropiat de 1",
                    "Mai mare decît 1",
                    "Negativ"
                ],
                "correctExplanation": "$\\alpha + \\beta$ măsoară persistența: efectul unui șoc asupra varianței condiționate viitoare scade ca $(\\alpha + \\beta)^h$. Valorile apropiate de 1 înseamnă o scădere lentă; condiția $\\alpha + \\beta < 1$ este necesară pentru staționaritatea în covarianță.",
                "incorrectExplanation": "O valoare apropiată de 0 înseamnă că șocurile se sting aproape imediat, o valoare peste 1 implică o varianță explozivă, nestaționară, iar valorile negative sînt excluse de $\\alpha, \\beta \\geq 0$. Persistența ridicată înseamnă $\\alpha + \\beta$ puțin sub 1."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Forecast uncertainty",
                "text": "As the forecast horizon increases, prediction intervals typically:",
                "options": [
                    "Narrow",
                    "Stay constant",
                    "Widen",
                    "Oscillate"
                ],
                "correctExplanation": "Each additional step adds the variance of new shocks, so the forecast error variance grows with the horizon. For a stationary model it converges to the unconditional variance; for a unit-root model it grows without bound.",
                "incorrectExplanation": "Intervals cannot narrow as uncertainty accumulates, they stay constant only in trivial cases, and they do not oscillate. Prediction intervals widen with the horizon."
            },
            "ro": {
                "title": "Incertitudinea prognozei",
                "text": "Pe măsură ce orizontul de prognoză crește, intervalele de prognoză de regulă:",
                "options": [
                    "Se îngustează",
                    "Rămîn constante",
                    "Se lărgesc",
                    "Oscilează"
                ],
                "correctExplanation": "Fiecare pas suplimentar adaugă varianța unor șocuri noi, deci varianța erorii de prognoză crește odată cu orizontul. Pentru un model staționar ea converge către varianța necondiționată; pentru un model cu rădăcină unitară crește nemărginit.",
                "incorrectExplanation": "Intervalele nu se pot îngusta cînd incertitudinea se acumulează, rămîn constante doar în cazuri banale și nu oscilează. Intervalele de prognoză se lărgesc odată cu orizontul."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Directional accuracy below 50%",
                "text": "On a long test sample in which up and down days are about equally frequent, a model has 45% directional accuracy for stock returns. The model is:",
                "options": [
                    "Better than a coin flip",
                    "Worse than a coin flip",
                    "Optimal for trading",
                    "Statistically significantly better than chance"
                ],
                "correctExplanation": "With balanced up and down moves, random guessing gets about 50% of directions right. A 45% hit rate is below that benchmark; financial returns are notoriously hard to predict.",
                "incorrectExplanation": "A hit rate under 50% cannot be better than a coin flip, let alone significantly better than chance or optimal for trading. On a balanced sample, 45% is worse than random guessing."
            },
            "ro": {
                "title": "Acuratețe direcțională sub 50%",
                "text": "Pe un eșantion de test lung, în care zilele de creștere și de scădere sînt aproximativ la fel de frecvente, un model are o acuratețe direcțională de 45% pentru randamentele acțiunilor. Modelul este:",
                "options": [
                    "Mai bun decît aruncarea unei monede",
                    "Mai slab decît aruncarea unei monede",
                    "Optim pentru tranzacționare",
                    "Semnificativ statistic mai bun decît hazardul"
                ],
                "correctExplanation": "Cînd creșterile și scăderile sînt echilibrate, ghicirea la întîmplare nimerește aproximativ 50% dintre direcții. O rată de 45% este sub acest reper; randamentele financiare sînt notoriu greu de prognozat.",
                "incorrectExplanation": "O rată de reușită sub 50% nu poate fi mai bună decît aruncarea unei monede, cu atît mai puțin semnificativ mai bună decît hazardul sau optimă pentru tranzacționare. Pe un eșantion echilibrat, 45% este mai slab decît ghicirea la întîmplare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Principle of parsimony",
                "text": "The principle of parsimony suggests choosing:",
                "options": [
                    "The model with the best in-sample fit",
                    "The simplest model that fits the data adequately",
                    "The most complex model available",
                    "The model with the most parameters"
                ],
                "correctExplanation": "Occam's razor: among models that describe the data adequately, prefer the simplest. Extra parameters improve in-sample fit but usually hurt out-of-sample forecasts through overfitting.",
                "incorrectExplanation": "The best in-sample fit is usually obtained by the most complex model, which is exactly what parsimony warns against. Choose the simplest adequate model."
            },
            "ro": {
                "title": "Principiul parcimoniei",
                "text": "Principiul parcimoniei recomandă alegerea:",
                "options": [
                    "Modelului cu cea mai bună ajustare în eșantion",
                    "Celui mai simplu model care descrie adecvat datele",
                    "Celui mai complex model disponibil",
                    "Modelului cu cei mai mulți parametri"
                ],
                "correctExplanation": "Briciul lui Occam: dintre modelele care descriu adecvat datele, îl preferăm pe cel mai simplu. Parametrii suplimentari îmbunătățesc ajustarea în eșantion, dar de regulă înrăutățesc prognozele în afara eșantionului, prin supraajustare.",
                "incorrectExplanation": "Cea mai bună ajustare în eșantion este obținută de regulă de modelul cel mai complex, tocmai ceea ce principiul parcimoniei ne cere să evităm. Alegem cel mai simplu model adecvat."
            }
        }
    ]
};
