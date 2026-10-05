// ============================================================
// Chapter 0 quiz bank: Introduction: components and exponential smoothing (EN + RO)
// 20 questions ported from the 2025/2026 site; 20 drawn per attempt.
// The bank grows to 24 questions when the chapter is rebuilt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['intro'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Additive and multiplicative decomposition",
                "text": "When should additive decomposition be preferred to multiplicative decomposition?",
                "options": [
                    "When the seasonal amplitude grows with the level of the series",
                    "When the seasonal amplitude stays constant over time",
                    "When the series has a strong downward trend",
                    "When the series contains missing values"
                ],
                "correctExplanation": "Additive decomposition $Y_t = T_t + S_t + R_t$ fits series whose seasonal swings keep the same size whatever the level. Multiplicative decomposition $Y_t = T_t \\times S_t \\times R_t$ fits series whose seasonal swings scale with the level.",
                "incorrectExplanation": "The choice depends on how the seasonal amplitude behaves, not on the direction of the trend or on missing values. A seasonal amplitude that grows with the level calls for the multiplicative form; a constant amplitude calls for the additive form."
            },
            "ro": {
                "title": "Descompunerea aditivă și cea multiplicativă",
                "text": "Cînd este preferabilă descompunerea aditivă celei multiplicative?",
                "options": [
                    "Cînd amplitudinea sezonieră crește odată cu nivelul seriei",
                    "Cînd amplitudinea sezonieră rămîne constantă în timp",
                    "Cînd seria are un trend descendent puternic",
                    "Cînd seria conține valori lipsă"
                ],
                "correctExplanation": "Descompunerea aditivă $Y_t = T_t + S_t + R_t$ se potrivește seriilor ale căror oscilații sezoniere păstrează aceeași mărime indiferent de nivel. Descompunerea multiplicativă $Y_t = T_t \\times S_t \\times R_t$ se potrivește seriilor ale căror oscilații sezoniere cresc proporțional cu nivelul.",
                "incorrectExplanation": "Alegerea depinde de comportamentul amplitudinii sezoniere, nu de sensul trendului sau de valorile lipsă. O amplitudine care crește odată cu nivelul cere forma multiplicativă; o amplitudine constantă cere forma aditivă."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Exponential smoothing methods",
                "text": "Which exponential smoothing method should be used for a series with both trend and seasonality?",
                "options": [
                    "Simple exponential smoothing (SES)",
                    "The Holt-Winters method",
                    "A simple moving average",
                    "Holt's linear method"
                ],
                "correctExplanation": "Holt-Winters (triple exponential smoothing) has three smoothing equations, for the level, the trend and the seasonal component, so it is built for series with both trend and seasonality.",
                "incorrectExplanation": "SES models only the level and suits series without trend or seasonality; Holt's linear method adds a trend but no seasonality; a simple moving average is a smoother, not a forecasting method with trend and seasonal components. Only Holt-Winters covers level, trend and seasonality."
            },
            "ro": {
                "title": "Metode de netezire exponențială",
                "text": "Ce metodă de netezire exponențială trebuie folosită pentru o serie care prezintă atît trend, cît și sezonalitate?",
                "options": [
                    "Netezirea exponențială simplă (SES)",
                    "Metoda Holt-Winters",
                    "Media mobilă simplă",
                    "Metoda liniară Holt"
                ],
                "correctExplanation": "Holt-Winters (netezirea exponențială triplă) are trei ecuații de netezire, pentru nivel, trend și componenta sezonieră, deci este construită pentru serii cu trend și sezonalitate.",
                "incorrectExplanation": "SES modelează doar nivelul și se potrivește seriilor fără trend și fără sezonalitate; metoda liniară Holt adaugă trendul, dar nu și sezonalitatea; media mobilă simplă este un instrument de netezire, nu o metodă de prognoză cu componente de trend și sezonalitate. Numai Holt-Winters acoperă nivelul, trendul și sezonalitatea."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Cross-validation for time series",
                "text": "Why is standard k-fold cross-validation problematic for time series data?",
                "options": [
                    "It requires too much data",
                    "It would use future observations to predict past observations",
                    "It does not work with seasonal data",
                    "It is too slow computationally"
                ],
                "correctExplanation": "Standard k-fold assigns observations to folds at random, so the model is often trained on observations that come after the ones it is evaluated on. This leaks future information and overstates forecast accuracy. Time series cross-validation keeps the chronological order.",
                "incorrectExplanation": "The problem is neither sample size, seasonality nor computing time: it is the violation of temporal order. Training on future observations to predict past ones is data leakage; rolling-origin validation avoids it."
            },
            "ro": {
                "title": "Validarea încrucișată pentru serii de timp",
                "text": "De ce este problematică validarea încrucișată k-fold standard pentru seriile de timp?",
                "options": [
                    "Necesită prea multe date",
                    "Ar folosi observații viitoare pentru a prognoza observații trecute",
                    "Nu funcționează cu date sezoniere",
                    "Este prea lentă din punct de vedere computațional"
                ],
                "correctExplanation": "K-fold standard repartizează aleator observațiile în subeșantioane, astfel încît modelul este adesea antrenat pe observații ulterioare celor pe care este evaluat. Se scurge astfel informație din viitor, iar acuratețea prognozei este supraestimată. Validarea încrucișată pentru serii de timp păstrează ordinea cronologică.",
                "incorrectExplanation": "Problema nu ține de volumul de date, de sezonalitate sau de timpul de calcul, ci de încălcarea ordinii temporale. Antrenarea pe observații viitoare pentru a prognoza observații trecute este o scurgere de informație (data leakage); validarea cu origine mobilă (rolling origin) o evită."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Smoothing parameter",
                "text": "What happens when the smoothing parameter $\\alpha$ of simple exponential smoothing moves towards 1?",
                "options": [
                    "The forecast becomes smoother and more stable",
                    "The forecast reacts more strongly to recent observations",
                    "The forecast converges to the sample mean of the series",
                    "The model gains additional parameters"
                ],
                "correctExplanation": "In SES, $\\hat{y}_{t+1} = \\alpha y_t + (1-\\alpha)\\hat{y}_t$. A value of $\\alpha$ close to 1 puts almost all the weight on the latest observation, so the forecast reacts quickly to changes but is also more volatile; at $\\alpha = 1$ it becomes the naive forecast.",
                "incorrectExplanation": "A smooth, stable forecast corresponds to a small $\\alpha$, and a forecast close to the overall mean arises as $\\alpha \\to 0$ with a long history. The number of parameters does not change with $\\alpha$. As $\\alpha \\to 1$, the weight shifts to the most recent observations."
            },
            "ro": {
                "title": "Parametrul de netezire",
                "text": "Ce se întîmplă cînd parametrul de netezire $\\alpha$ al netezirii exponențiale simple se apropie de 1?",
                "options": [
                    "Prognoza devine mai netedă și mai stabilă",
                    "Prognoza reacționează mai puternic la observațiile recente",
                    "Prognoza converge către media de selecție a seriei",
                    "Modelul capătă parametri suplimentari"
                ],
                "correctExplanation": "În SES, $\\hat{y}_{t+1} = \\alpha y_t + (1-\\alpha)\\hat{y}_t$. O valoare a lui $\\alpha$ apropiată de 1 pune aproape toată ponderea pe ultima observație, deci prognoza reacționează rapid la schimbări, dar este și mai volatilă; pentru $\\alpha = 1$ se obține prognoza naivă.",
                "incorrectExplanation": "O prognoză netedă și stabilă corespunde unui $\\alpha$ mic, iar o prognoză apropiată de media generală apare cînd $\\alpha \\to 0$ și istoricul este lung. Numărul de parametri nu depinde de $\\alpha$. Cînd $\\alpha \\to 1$, ponderea se mută pe observațiile cele mai recente."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Deterministic and stochastic trend",
                "text": "What is the correct way to handle the two types of trend?",
                "options": [
                    "Always use differencing, whatever the type of trend",
                    "Deterministic trend: regression on time; stochastic trend: differencing",
                    "Always use regression on time, whatever the type of trend",
                    "Ignore the trend, because it vanishes in long samples"
                ],
                "correctExplanation": "A deterministic trend is a fixed function of time and is removed by regressing the series on time. A stochastic trend comes from a unit root (random walk component) and is removed by differencing.",
                "incorrectExplanation": "Differencing a trend-stationary series over-differences it and creates a non-invertible MA component; regressing a unit-root series on time leaves a non-stationary residual and can produce spurious results. A trend never vanishes in long samples. The treatment must match the type of trend."
            },
            "ro": {
                "title": "Trend determinist și trend stochastic",
                "text": "Care este modul corect de tratare a celor două tipuri de trend?",
                "options": [
                    "Se folosește întotdeauna diferențierea, indiferent de tipul trendului",
                    "Trend determinist: regresie pe timp; trend stochastic: diferențiere",
                    "Se folosește întotdeauna regresia pe timp, indiferent de tipul trendului",
                    "Trendul se ignoră, deoarece dispare în eșantioanele lungi"
                ],
                "correctExplanation": "Un trend determinist este o funcție fixă de timp și se elimină prin regresia seriei pe timp. Un trend stochastic provine dintr-o rădăcină unitară (o componentă de mers aleator) și se elimină prin diferențiere.",
                "incorrectExplanation": "Diferențierea unei serii staționare în jurul unui trend o supradiferențiază și creează o componentă MA neinversabilă; regresia pe timp a unei serii cu rădăcină unitară lasă reziduuri nestaționare și poate produce rezultate false. Trendul nu dispare în eșantioanele lungi. Tratamentul trebuie să corespundă tipului de trend."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Moving average",
                "text": "What is the main purpose of a centred moving average in time series analysis?",
                "options": [
                    "To forecast future values",
                    "To extract the trend component by smoothing out noise and seasonality",
                    "To increase the variance of the series",
                    "To detect outliers"
                ],
                "correctExplanation": "A centred moving average whose window spans a full seasonal cycle averages out the seasonal effect and short-term noise, leaving an estimate of the trend (trend-cycle).",
                "incorrectExplanation": "A centred average uses observations on both sides of $t$, so it cannot forecast; it reduces rather than increases variance, and it is not an outlier detector. Its role in classical decomposition is to estimate the trend."
            },
            "ro": {
                "title": "Media mobilă",
                "text": "Care este scopul principal al unei medii mobile centrate în analiza seriilor de timp?",
                "options": [
                    "Prognoza valorilor viitoare",
                    "Extragerea componentei de trend prin netezirea zgomotului și a sezonalității",
                    "Creșterea varianței seriei",
                    "Detectarea valorilor aberante"
                ],
                "correctExplanation": "O medie mobilă centrată a cărei fereastră acoperă un ciclu sezonier complet elimină prin mediere efectul sezonier și zgomotul de termen scurt, lăsînd o estimare a trendului (trend-ciclu).",
                "incorrectExplanation": "O medie centrată folosește observații de ambele părți ale lui $t$, deci nu poate prognoza; reduce varianța, nu o crește, și nu este un instrument de detectare a valorilor aberante. Rolul ei în descompunerea clasică este estimarea trendului."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "STL decomposition",
                "text": "What is the main advantage of STL decomposition over classical decomposition?",
                "options": [
                    "It is faster to compute",
                    "It allows the seasonal component to change over time and is robust to outliers",
                    "It requires no parameters",
                    "It works only with monthly data"
                ],
                "correctExplanation": "STL (Seasonal-Trend decomposition using LOESS) lets the seasonal pattern evolve over time and has a robust option that downweights outliers, so a few extreme values do not distort the trend and seasonal estimates.",
                "incorrectExplanation": "Speed is not its selling point, it does require parameters (the seasonal and trend window lengths), and it works with any seasonal period. Its advantages are a time-varying seasonal component and robustness to outliers."
            },
            "ro": {
                "title": "Descompunerea STL",
                "text": "Care este principalul avantaj al descompunerii STL față de descompunerea clasică?",
                "options": [
                    "Se calculează mai rapid",
                    "Permite componentei sezoniere să se modifice în timp și este robustă la valori aberante",
                    "Nu necesită parametri",
                    "Funcționează doar cu date lunare"
                ],
                "correctExplanation": "STL (Seasonal-Trend decomposition using LOESS) permite tiparului sezonier să evolueze în timp și are o variantă robustă care reduce ponderea valorilor aberante, astfel încît cîteva valori extreme nu distorsionează estimările trendului și ale sezonalității.",
                "incorrectExplanation": "Viteza nu este avantajul ei, necesită parametri (lungimile ferestrelor pentru sezonalitate și trend) și funcționează cu orice perioadă sezonieră. Avantajele sînt componenta sezonieră variabilă în timp și robustețea la valori aberante."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Forecast error metrics",
                "text": "Which error metric is scale-independent and allows comparisons across different time series?",
                "options": [
                    "Mean absolute error (MAE)",
                    "Mean absolute percentage error (MAPE)",
                    "Root mean squared error (RMSE)",
                    "Sum of squared errors (SSE)"
                ],
                "correctExplanation": "MAPE expresses each error as a percentage of the actual value, so it does not depend on the units of the series and can be compared across series with different scales.",
                "incorrectExplanation": "MAE, RMSE and SSE are measured in the units of the series (or their square), so they change when the series is rescaled. MAPE is unit-free because it divides by the actual values."
            },
            "ro": {
                "title": "Indicatori ai erorii de prognoză",
                "text": "Ce indicator al erorii este independent de scală și permite comparații între serii de timp diferite?",
                "options": [
                    "Eroarea medie absolută (MAE)",
                    "Eroarea medie absolută procentuală (MAPE)",
                    "Rădăcina erorii pătratice medii (RMSE)",
                    "Suma pătratelor erorilor (SSE)"
                ],
                "correctExplanation": "MAPE exprimă fiecare eroare ca procent din valoarea efectivă, deci nu depinde de unitatea de măsură a seriei și poate fi comparată între serii cu scale diferite.",
                "incorrectExplanation": "MAE, RMSE și SSE se măsoară în unitățile seriei (sau în pătratul lor), deci se modifică atunci cînd seria este rescalată. MAPE nu are unitate de măsură, deoarece împarte la valorile efective."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Holt's linear method",
                "text": "Which components does Holt's linear method model?",
                "options": [
                    "Level only",
                    "Level and trend",
                    "Level, trend and seasonality",
                    "Seasonality only"
                ],
                "correctExplanation": "Holt's method (double exponential smoothing) has one smoothing equation for the level and one for the trend; it has no seasonal component.",
                "incorrectExplanation": "The level alone is simple exponential smoothing, and level, trend and seasonality together are Holt-Winters. Holt's method adds a trend equation to SES, nothing more."
            },
            "ro": {
                "title": "Metoda liniară Holt",
                "text": "Ce componente modelează metoda liniară Holt?",
                "options": [
                    "Doar nivelul",
                    "Nivelul și trendul",
                    "Nivelul, trendul și sezonalitatea",
                    "Doar sezonalitatea"
                ],
                "correctExplanation": "Metoda Holt (netezirea exponențială dublă) are o ecuație de netezire pentru nivel și una pentru trend; nu are componentă sezonieră.",
                "incorrectExplanation": "Doar nivelul corespunde netezirii exponențiale simple, iar nivelul, trendul și sezonalitatea împreună corespund metodei Holt-Winters. Metoda Holt adaugă la SES doar ecuația trendului."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Seasonal period",
                "text": "For monthly data with yearly seasonality, what is the seasonal period $m$?",
                "options": [
                    "$m = 4$",
                    "$m = 12$",
                    "$m = 52$",
                    "$m = 365$"
                ],
                "correctExplanation": "One yearly cycle contains 12 monthly observations, so $m = 12$.",
                "incorrectExplanation": "The other values are the periods for other frequencies with yearly seasonality: $m = 4$ for quarterly data, $m = 52$ for weekly data, $m = 365$ for daily data. Monthly data give $m = 12$."
            },
            "ro": {
                "title": "Perioada sezonieră",
                "text": "Pentru date lunare cu sezonalitate anuală, cît este perioada sezonieră $m$?",
                "options": [
                    "$m = 4$",
                    "$m = 12$",
                    "$m = 52$",
                    "$m = 365$"
                ],
                "correctExplanation": "Un ciclu anual conține 12 observații lunare, deci $m = 12$.",
                "incorrectExplanation": "Celelalte valori sînt perioadele pentru alte frecvențe cu sezonalitate anuală: $m = 4$ pentru date trimestriale, $m = 52$ pentru date săptămînale, $m = 365$ pentru date zilnice. Datele lunare au $m = 12$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Training and test split",
                "text": "In time series forecasting, how should the data be split into a training set and a test set?",
                "options": [
                    "Shuffle the observations at random and split",
                    "Use the earlier observations for training and the later ones for testing",
                    "Alternate between training and test observations",
                    "Use the middle part of the sample for testing"
                ],
                "correctExplanation": "The temporal order must be respected: the model is estimated on the past and evaluated on the future, exactly as it will be used in practice.",
                "incorrectExplanation": "Random shuffling, alternating observations or a test block in the middle all let the model see observations that come after the test period, which leaks future information. The test set must be the most recent part of the sample."
            },
            "ro": {
                "title": "Împărțirea în set de antrenare și set de test",
                "text": "În prognoza seriilor de timp, cum trebuie împărțite datele în set de antrenare și set de test?",
                "options": [
                    "Observațiile se amestecă aleator și apoi se împart",
                    "Observațiile mai vechi se folosesc pentru antrenare, iar cele mai recente pentru testare",
                    "Observațiile se alternează între antrenare și test",
                    "Partea din mijloc a eșantionului se folosește pentru testare"
                ],
                "correctExplanation": "Ordinea temporală trebuie respectată: modelul se estimează pe trecut și se evaluează pe viitor, exact cum va fi folosit în practică.",
                "incorrectExplanation": "Amestecarea aleatoare, alternarea observațiilor sau un bloc de test la mijloc permit modelului să vadă observații ulterioare perioadei de test, ceea ce înseamnă scurgere de informație din viitor. Setul de test trebuie să fie partea cea mai recentă a eșantionului."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "RMSE and MAE",
                "text": "When is RMSE preferred to MAE?",
                "options": [
                    "When all errors should be weighted equally",
                    "When large errors are particularly costly and should be penalised more",
                    "When the data contain many outliers",
                    "When a percentage-based metric is needed"
                ],
                "correctExplanation": "RMSE squares the errors before averaging, so large errors weigh disproportionately. It is the natural choice when big misses are much more costly than small ones.",
                "incorrectExplanation": "Equal weighting of errors is what MAE does, and MAE is also the more robust choice when the data contain many outliers. Neither RMSE nor MAE is a percentage metric. RMSE is chosen precisely because it penalises large errors more."
            },
            "ro": {
                "title": "RMSE și MAE",
                "text": "Cînd este preferată RMSE în locul MAE?",
                "options": [
                    "Cînd toate erorile trebuie ponderate egal",
                    "Cînd erorile mari sînt deosebit de costisitoare și trebuie penalizate mai mult",
                    "Cînd datele conțin multe valori aberante",
                    "Cînd este nevoie de un indicator procentual"
                ],
                "correctExplanation": "RMSE ridică erorile la pătrat înainte de mediere, deci erorile mari au o pondere disproporționată. Este alegerea firească atunci cînd erorile mari sînt mult mai costisitoare decît cele mici.",
                "incorrectExplanation": "Ponderarea egală a erorilor este proprietatea MAE, iar MAE este și alegerea mai robustă cînd datele conțin multe valori aberante. Nici RMSE, nici MAE nu sînt indicatori procentuali. RMSE se alege tocmai pentru că penalizează mai mult erorile mari."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Detecting seasonality",
                "text": "How can seasonality be detected visually in a time series?",
                "options": [
                    "Look for a straight-line pattern",
                    "Look for patterns that repeat at regular intervals, or use seasonal subseries plots",
                    "Check whether the mean is zero",
                    "Compute the standard deviation"
                ],
                "correctExplanation": "Seasonality shows up as a pattern that repeats at a fixed interval (every year, quarter or week). Seasonal subseries plots group the observations by season and make the pattern easy to see.",
                "incorrectExplanation": "A straight line indicates a linear trend, a zero mean says nothing about periodic behaviour, and a standard deviation is a single number with no time structure. Seasonality is a pattern that repeats at regular intervals."
            },
            "ro": {
                "title": "Detectarea sezonalității",
                "text": "Cum poate fi detectată vizual sezonalitatea într-o serie de timp?",
                "options": [
                    "Se caută un tipar de linie dreaptă",
                    "Se caută tipare care se repetă la intervale regulate sau se folosesc grafice pe subserii sezoniere",
                    "Se verifică dacă media este zero",
                    "Se calculează abaterea standard"
                ],
                "correctExplanation": "Sezonalitatea apare ca un tipar care se repetă la un interval fix (anual, trimestrial sau săptămînal). Graficele pe subserii sezoniere grupează observațiile după sezon și fac tiparul ușor de observat.",
                "incorrectExplanation": "O linie dreaptă indică un trend liniar, o medie nulă nu spune nimic despre comportamentul periodic, iar abaterea standard este un singur număr, fără structură temporală. Sezonalitatea este un tipar care se repetă la intervale regulate."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Damped trend",
                "text": "What is the purpose of the damping parameter in Holt's method?",
                "options": [
                    "To make the trend grow faster",
                    "To flatten the trend in the long run and prevent unrealistic extrapolation",
                    "To remove seasonality",
                    "To speed up computation"
                ],
                "correctExplanation": "With a damping parameter $0 < \\phi < 1$, the trend contribution to the $h$-step forecast is $(\\phi + \\phi^2 + \\dots + \\phi^h) b_t$, which converges, so long-horizon forecasts level off instead of growing linearly forever.",
                "incorrectExplanation": "Damping slows the trend down rather than accelerating it, has nothing to do with seasonality and is not a computational device. Its role is to stop the trend from being extrapolated indefinitely."
            },
            "ro": {
                "title": "Trend amortizat",
                "text": "Care este rolul parametrului de amortizare în metoda Holt?",
                "options": [
                    "Accelerează creșterea trendului",
                    "Aplatizează trendul pe termen lung și previne extrapolarea nerealistă",
                    "Elimină sezonalitatea",
                    "Crește viteza de calcul"
                ],
                "correctExplanation": "Cu un parametru de amortizare $0 < \\phi < 1$, contribuția trendului la prognoza pe $h$ pași este $(\\phi + \\phi^2 + \\dots + \\phi^h) b_t$, care converge, deci prognozele pe orizonturi lungi se stabilizează în loc să crească liniar la nesfîrșit.",
                "incorrectExplanation": "Amortizarea încetinește trendul, nu îl accelerează, nu are legătură cu sezonalitatea și nu este un artificiu de calcul. Rolul ei este să împiedice extrapolarea trendului la nesfîrșit."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Log transformation",
                "text": "When is a logarithmic transformation appropriate for time series data?",
                "options": [
                    "When the series has negative values",
                    "When the variance increases with the level, or when the patterns are multiplicative",
                    "When the data are already stationary",
                    "When the variance should be increased"
                ],
                "correctExplanation": "The logarithm stabilises a variance that grows with the level and turns multiplicative effects into additive ones: $\\ln(T_t S_t R_t) = \\ln T_t + \\ln S_t + \\ln R_t$.",
                "incorrectExplanation": "The logarithm is not defined for zero or negative values, it is not needed when the data are already stationary with constant variance, and it compresses rather than increases variability. It is used when the spread grows with the level."
            },
            "ro": {
                "title": "Transformarea logaritmică",
                "text": "Cînd este adecvată transformarea logaritmică a unei serii de timp?",
                "options": [
                    "Cînd seria are valori negative",
                    "Cînd varianța crește odată cu nivelul sau cînd tiparele sînt multiplicative",
                    "Cînd datele sînt deja staționare",
                    "Cînd se dorește creșterea varianței"
                ],
                "correctExplanation": "Logaritmul stabilizează o varianță care crește odată cu nivelul și transformă efectele multiplicative în efecte aditive: $\\ln(T_t S_t R_t) = \\ln T_t + \\ln S_t + \\ln R_t$.",
                "incorrectExplanation": "Logaritmul nu este definit pentru valori nule sau negative, nu este necesar cînd datele sînt deja staționare și au varianță constantă și comprimă variabilitatea, nu o mărește. Se folosește cînd dispersia crește odată cu nivelul."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Forecast horizon",
                "text": "What typically happens to forecast accuracy as the forecast horizon increases?",
                "options": [
                    "Accuracy improves because more data are averaged",
                    "Accuracy decreases because uncertainty accumulates",
                    "Accuracy stays constant",
                    "Accuracy depends only on the model, not on the horizon"
                ],
                "correctExplanation": "Each additional step ahead adds the uncertainty of new, unobserved shocks, so forecast error variance grows with the horizon and accuracy falls.",
                "incorrectExplanation": "A longer horizon does not bring more data, and accuracy is not constant: for virtually every model the forecast error variance grows with $h$. Prediction intervals therefore widen as the horizon lengthens."
            },
            "ro": {
                "title": "Orizontul de prognoză",
                "text": "Ce se întîmplă de obicei cu acuratețea prognozei pe măsură ce orizontul de prognoză crește?",
                "options": [
                    "Acuratețea crește, deoarece se mediază mai multe date",
                    "Acuratețea scade, deoarece incertitudinea se acumulează",
                    "Acuratețea rămîne constantă",
                    "Acuratețea depinde doar de model, nu și de orizont"
                ],
                "correctExplanation": "Fiecare pas suplimentar adaugă incertitudinea unor șocuri noi, neobservate, deci varianța erorii de prognoză crește odată cu orizontul, iar acuratețea scade.",
                "incorrectExplanation": "Un orizont mai lung nu aduce date suplimentare, iar acuratețea nu este constantă: pentru practic orice model, varianța erorii de prognoză crește odată cu $h$. De aceea intervalele de prognoză se lărgesc pe măsură ce orizontul crește."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Naive forecast",
                "text": "What is a naive forecast and why is it useful?",
                "options": [
                    "The most complex model available; useful for maximum accuracy",
                    "The last observation used as the forecast; useful as a benchmark to beat",
                    "The mean of all observations; useful for trending data",
                    "A random forecast; useful for testing"
                ],
                "correctExplanation": "The naive forecast sets $\\hat{y}_{T+h} = y_T$. It costs nothing to compute and is hard to beat for random-walk-like series, so any more elaborate model should be compared against it.",
                "incorrectExplanation": "The naive method is the simplest possible model, not the most complex. The mean forecast is a different benchmark and performs badly on trending data, and a random forecast is not a meaningful benchmark. The naive forecast repeats the last observation."
            },
            "ro": {
                "title": "Prognoza naivă",
                "text": "Ce este prognoza naivă și de ce este utilă?",
                "options": [
                    "Cel mai complex model disponibil; utilă pentru acuratețe maximă",
                    "Ultima observație folosită ca prognoză; utilă ca reper care trebuie depășit",
                    "Media tuturor observațiilor; utilă pentru date cu trend",
                    "O prognoză aleatoare; utilă pentru testare"
                ],
                "correctExplanation": "Prognoza naivă este $\\hat{y}_{T+h} = y_T$. Nu costă nimic și este greu de depășit pentru seriile apropiate de un mers aleator, deci orice model mai elaborat trebuie comparat cu ea.",
                "incorrectExplanation": "Metoda naivă este cel mai simplu model posibil, nu cel mai complex. Prognoza prin medie este un alt reper și funcționează prost pe date cu trend, iar o prognoză aleatoare nu este un reper util. Prognoza naivă repetă ultima observație."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Seasonal naive forecast",
                "text": "For monthly data, what does the seasonal naive method forecast for next January?",
                "options": [
                    "The value of last December",
                    "The value of last January (same month, previous year)",
                    "The average of all months",
                    "Zero"
                ],
                "correctExplanation": "The seasonal naive forecast is $\\hat{y}_{T+h} = y_{T+h-m}$ (for $h \\le m$): each month is forecast by the value of the same month one seasonal cycle earlier, with $m = 12$ for monthly data.",
                "incorrectExplanation": "Last December's value is the plain naive forecast, and the average of all months is the mean forecast. The seasonal version looks back exactly $m = 12$ months, to the same month of the previous year."
            },
            "ro": {
                "title": "Prognoza naivă sezonieră",
                "text": "Pentru date lunare, ce valoare prognozează metoda naivă sezonieră pentru luna ianuarie următoare?",
                "options": [
                    "Valoarea din decembrie anterior",
                    "Valoarea din ianuarie anterior (aceeași lună, anul precedent)",
                    "Media tuturor lunilor",
                    "Zero"
                ],
                "correctExplanation": "Prognoza naivă sezonieră este $\\hat{y}_{T+h} = y_{T+h-m}$ (pentru $h \\le m$): fiecare lună este prognozată prin valoarea aceleiași luni dintr-un ciclu sezonier anterior, cu $m = 12$ pentru date lunare.",
                "incorrectExplanation": "Valoarea din decembrie anterior este prognoza naivă simplă, iar media tuturor lunilor este prognoza prin medie. Varianta sezonieră privește înapoi exact $m = 12$ luni, la aceeași lună din anul precedent."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Residual analysis",
                "text": "What should the residuals of a good forecasting model look like?",
                "options": [
                    "They should show a clear trend",
                    "They should be uncorrelated with zero mean, like white noise",
                    "Their variance should increase over time",
                    "They should be exactly zero"
                ],
                "correctExplanation": "If the model has captured the systematic structure, what is left is unpredictable: uncorrelated residuals with zero mean and, ideally, constant variance.",
                "incorrectExplanation": "A trend or a growing variance in the residuals signals structure the model has missed, and residuals that are exactly zero indicate a model that interpolates the data (overfitting). Good residuals behave like white noise."
            },
            "ro": {
                "title": "Analiza reziduurilor",
                "text": "Cum trebuie să arate reziduurile unui model de prognoză bun?",
                "options": [
                    "Să prezinte un trend clar",
                    "Să fie necorelate și de medie zero, asemenea unui zgomot alb",
                    "Să aibă o varianță care crește în timp",
                    "Să fie exact zero"
                ],
                "correctExplanation": "Dacă modelul a captat structura sistematică, ceea ce rămîne este imprevizibil: reziduuri necorelate, de medie zero și, ideal, cu varianță constantă.",
                "incorrectExplanation": "Un trend sau o varianță crescătoare în reziduuri indică o structură pe care modelul nu a captat-o, iar reziduurile exact nule arată un model care interpolează datele (supraajustare). Reziduurile bune se comportă ca un zgomot alb."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "ETS framework",
                "text": "In the notation ETS(A,A,A), what do the three letters stand for?",
                "options": [
                    "Additive error, additive trend, additive seasonality",
                    "Average, average, average",
                    "Three alpha parameters",
                    "Autocorrelation at lags 1, 2 and 3"
                ],
                "correctExplanation": "ETS stands for Error, Trend, Seasonality. Each component can be additive (A), multiplicative (M) or absent (N, none); the trend can also be additive damped (Ad).",
                "incorrectExplanation": "The letters are not averages, smoothing parameters or autocorrelations: they describe the form of each component. ETS(A,A,A) has additive error, additive trend and additive seasonality."
            },
            "ro": {
                "title": "Cadrul ETS",
                "text": "În notația ETS(A,A,A), ce reprezintă cele trei litere?",
                "options": [
                    "Eroare aditivă, trend aditiv, sezonalitate aditivă",
                    "Medie, medie, medie",
                    "Trei parametri alfa",
                    "Autocorelațiile la lag-urile 1, 2 și 3"
                ],
                "correctExplanation": "ETS înseamnă Error, Trend, Seasonality (eroare, trend, sezonalitate). Fiecare componentă poate fi aditivă (A), multiplicativă (M) sau absentă (N, none); trendul poate fi și aditiv amortizat (Ad).",
                "incorrectExplanation": "Literele nu sînt medii, parametri de netezire sau autocorelații, ci descriu forma fiecărei componente. ETS(A,A,A) are eroare aditivă, trend aditiv și sezonalitate aditivă."
            }
        }
    ]
};
