// ============================================================
// Chapter 0 quiz bank: Introduction: components and exponential smoothing (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['intro'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "What a time series is",
                "text": "What distinguishes a time series from a cross-section?",
                "options": [
                    "Its observations are ordered in time and usually depend on each other",
                    "It always contains more observations than a cross-section",
                    "Its observations are independent draws from one distribution",
                    "It can only be analysed with a regression on time"
                ],
                "correctExplanation": "A time series follows one variable over time; consecutive observations are usually related, and this dependence is the information used for forecasting.",
                "incorrectExplanation": "The number of observations and the choice of method do not define a time series; the key is the order in time and the dependence between consecutive observations, which also rules out the independence assumption."
            },
            "ro": {
                "title": "Definiția seriei de timp",
                "text": "Ce deosebește o serie de timp de un set de date transversale?",
                "options": [
                    "Observațiile ei sînt ordonate în timp și depind, de obicei, unele de altele",
                    "Conține întotdeauna mai multe observații decît un set de date transversale",
                    "Observațiile ei sînt extrageri independente din aceeași distribuție",
                    "Poate fi analizată doar printr-o regresie pe timp"
                ],
                "correctExplanation": "O serie de timp urmărește o variabilă în timp; observațiile consecutive sînt de obicei legate între ele, iar această dependență este informația folosită pentru prognoză.",
                "incorrectExplanation": "Numărul de observații și metoda aleasă nu definesc o serie de timp; esențiale sînt ordinea în timp și dependența dintre observațiile consecutive, care exclude și ipoteza de independență."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Seasonality and cycle",
                "text": "What distinguishes seasonality from a cycle?",
                "options": [
                    "A cycle repeats every 12 months; seasonality has no fixed length",
                    "Seasonality repeats with a fixed, known period; a cycle has no fixed period",
                    "Seasonality appears only in financial data; cycles only in macroeconomic data",
                    "There is no difference: the two words mean the same component"
                ],
                "correctExplanation": "Seasonality has a fixed period known in advance (4 quarters, 12 months); cycles such as recessions arrive at irregular moments and last an irregular time.",
                "incorrectExplanation": "The distinction is the period: fixed and known for seasonality, irregular for cycles. It does not depend on the field, and the two are different components with different forecastability."
            },
            "ro": {
                "title": "Sezonalitate și ciclu",
                "text": "Ce deosebește sezonalitatea de un ciclu?",
                "options": [
                    "Ciclul se repetă la fiecare 12 luni; sezonalitatea nu are o durată fixă",
                    "Sezonalitatea se repetă cu o perioadă fixă și cunoscută; ciclul nu are o perioadă fixă",
                    "Sezonalitatea apare doar în datele financiare, iar ciclurile doar în cele macroeconomice",
                    "Nu există nicio diferență: cele două cuvinte desemnează aceeași componentă"
                ],
                "correctExplanation": "Sezonalitatea are o perioadă fixă, cunoscută dinainte (4 trimestre, 12 luni); ciclurile, precum recesiunile, apar în momente neregulate și durează un timp neregulat.",
                "incorrectExplanation": "Diferența ține de perioadă: fixă și cunoscută pentru sezonalitate, neregulată pentru cicluri. Nu depinde de domeniu, iar cele două sînt componente distincte, cu previzibilitate diferită."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Additive and multiplicative decomposition",
                "text": "When should additive decomposition be preferred to multiplicative decomposition?",
                "options": [
                    "When the seasonal amplitude grows with the level of the series",
                    "When the series has a strong downward trend",
                    "When the seasonal amplitude stays constant over time",
                    "When the series contains missing values"
                ],
                "correctExplanation": "Additive decomposition $y_t = T_t + S_t + R_t$ fits series whose seasonal swings keep the same size whatever the level. Multiplicative decomposition $y_t = T_t \\times S_t \\times R_t$ fits series whose seasonal swings scale with the level.",
                "incorrectExplanation": "The choice depends on how the seasonal amplitude behaves, not on the direction of the trend or on missing values. A seasonal amplitude that grows with the level calls for the multiplicative form; a constant amplitude calls for the additive form."
            },
            "ro": {
                "title": "Descompunerea aditivă și cea multiplicativă",
                "text": "Cînd este preferabilă descompunerea aditivă celei multiplicative?",
                "options": [
                    "Cînd amplitudinea sezonieră crește odată cu nivelul seriei",
                    "Cînd seria are un trend descendent puternic",
                    "Cînd amplitudinea sezonieră rămîne constantă în timp",
                    "Cînd seria conține valori lipsă"
                ],
                "correctExplanation": "Descompunerea aditivă $y_t = T_t + S_t + R_t$ se potrivește seriilor ale căror oscilații sezoniere păstrează aceeași mărime indiferent de nivel. Descompunerea multiplicativă $y_t = T_t \\times S_t \\times R_t$ se potrivește seriilor ale căror oscilații sezoniere cresc proporțional cu nivelul.",
                "incorrectExplanation": "Alegerea depinde de comportamentul amplitudinii sezoniere, nu de sensul trendului sau de valorile lipsă. O amplitudine care crește odată cu nivelul cere forma multiplicativă; o amplitudine constantă cere forma aditivă."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Seasonal factor",
                "text": "In a multiplicative decomposition of quarterly GDP, the seasonal factor of the first quarter is 0.76. What does it mean?",
                "options": [
                    "GDP falls by 0.76% every first quarter",
                    "76% of the variance of GDP is seasonal",
                    "The first quarter is 0.76 billion EUR below the trend",
                    "A typical first quarter is about 24% below the trend-cycle"
                ],
                "correctExplanation": "In the multiplicative model $y_t = T_t \\times S_t \\times R_t$, a factor of 0.76 multiplies the trend-cycle: the typical first quarter is $1 - 0.76 = 24\\%$ below it.",
                "incorrectExplanation": "A multiplicative factor is a ratio to the trend-cycle, not a percentage change, a share of variance or an amount in euros; 0.76 means 24% below the trend-cycle."
            },
            "ro": {
                "title": "Factorul sezonier",
                "text": "Într-o descompunere multiplicativă a PIB-ului trimestrial, factorul sezonier al trimestrului I este 0,76. Ce înseamnă?",
                "options": [
                    "PIB-ul scade cu 0,76% în fiecare trimestru I",
                    "76% din varianța PIB este sezonieră",
                    "Trimestrul I se află cu 0,76 miliarde EUR sub trend",
                    "Un trimestru I tipic se află cu aproximativ 24% sub trend-ciclu"
                ],
                "correctExplanation": "În modelul multiplicativ $y_t = T_t \\times S_t \\times R_t$, factorul 0,76 înmulțește trend-ciclul: trimestrul I tipic se află cu $1 - 0,76 = 24\\%$ sub acesta.",
                "incorrectExplanation": "Un factor multiplicativ este un raport față de trend-ciclu, nu o variație procentuală, o pondere a varianței sau o sumă în euro; 0,76 înseamnă 24% sub trend-ciclu."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Centred moving average",
                "text": "What is the main purpose of a centred $2 \\times m$ moving average in a decomposition?",
                "options": [
                    "To estimate the trend-cycle by averaging out the seasonal pattern and the noise",
                    "To forecast the next $m$ values",
                    "To increase the variance of the series",
                    "To detect outliers"
                ],
                "correctExplanation": "A moving average over one full season gives every season the same weight, so the seasonal effects cancel; what remains is a smooth estimate of the trend-cycle.",
                "incorrectExplanation": "A centred moving average uses future values as well as past ones, so it cannot forecast; it reduces the variance instead of increasing it, and it is not an outlier test. Its role is to estimate the trend-cycle."
            },
            "ro": {
                "title": "Media mobilă centrată",
                "text": "Care este rolul principal al unei medii mobile centrate $2 \\times m$ într-o descompunere?",
                "options": [
                    "Estimarea trend-ciclului, prin compensarea tiparului sezonier și a zgomotului",
                    "Prognoza următoarelor $m$ valori",
                    "Creșterea varianței seriei",
                    "Detectarea valorilor extreme"
                ],
                "correctExplanation": "O medie mobilă pe un sezon complet acordă aceeași pondere fiecărui sezon, deci efectele sezoniere se compensează; rămîne o estimare netedă a trend-ciclului.",
                "incorrectExplanation": "O medie mobilă centrată folosește și valori viitoare, nu doar trecute, deci nu poate prognoza; reduce varianța în loc să o crească și nu este un test pentru valori extreme. Rolul ei este estimarea trend-ciclului."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "STL decomposition",
                "text": "What is the main advantage of STL decomposition over classical decomposition?",
                "options": [
                    "It handles calendar effects such as Easter automatically",
                    "It lets the seasonal component change over time and is robust to outliers",
                    "It requires no parameters",
                    "It works only with monthly data"
                ],
                "correctExplanation": "STL (Seasonal and Trend decomposition using LOESS) lets the seasonal pattern evolve slowly, estimates the trend up to the last observation and, in its robust version, sends outliers into the remainder.",
                "incorrectExplanation": "STL needs choices such as the period and the smoothness, works with any seasonal period and does not treat calendar effects; its advantage is a seasonal pattern that may change and robustness to outliers."
            },
            "ro": {
                "title": "Descompunerea STL",
                "text": "Care este principalul avantaj al descompunerii STL față de descompunerea clasică?",
                "options": [
                    "Tratează automat efectele de calendar, precum Paștele",
                    "Permite componentei sezoniere să se modifice în timp și este robustă la valori extreme",
                    "Nu necesită parametri",
                    "Funcționează doar cu date lunare"
                ],
                "correctExplanation": "STL (Seasonal and Trend decomposition using LOESS) permite tiparului sezonier să evolueze lent, estimează trendul pînă la ultima observație și, în varianta robustă, trimite valorile extreme în componenta neregulată.",
                "incorrectExplanation": "STL cere alegeri precum perioada și gradul de netezire, funcționează cu orice perioadă sezonieră și nu tratează efectele de calendar; avantajul ei este un tipar sezonier care se poate modifica și robustețea la valori extreme."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Seasonal period",
                "text": "For monthly data with a yearly seasonal pattern, what is the seasonal period $m$?",
                "options": [
                    "$m = 4$",
                    "$m = 52$",
                    "$m = 12$",
                    "$m = 365$"
                ],
                "correctExplanation": "The pattern repeats every year and a year has 12 monthly observations, so $m = 12$; quarterly data have $m = 4$.",
                "incorrectExplanation": "The period counts the observations in one full seasonal cycle: 4 for quarterly data, 52 for weekly data, 365 for daily data; monthly data with a yearly pattern have $m = 12$."
            },
            "ro": {
                "title": "Perioada sezonieră",
                "text": "Pentru date lunare cu un tipar sezonier anual, cît este perioada sezonieră $m$?",
                "options": [
                    "$m = 4$",
                    "$m = 52$",
                    "$m = 12$",
                    "$m = 365$"
                ],
                "correctExplanation": "Tiparul se repetă în fiecare an, iar un an are 12 observații lunare, deci $m = 12$; datele trimestriale au $m = 4$.",
                "incorrectExplanation": "Perioada numără observațiile dintr-un ciclu sezonier complet: 4 pentru date trimestriale, 52 pentru date săptămînale, 365 pentru date zilnice; datele lunare cu tipar anual au $m = 12$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Growth rate of an unadjusted series",
                "text": "Unadjusted Romanian GDP falls by about a third from the fourth quarter to the next first quarter. Which growth rate should be reported?",
                "options": [
                    "The growth over the previous quarter, $100\\,(y_t / y_{t-1} - 1)$",
                    "The first difference $y_t - y_{t-1}$ in billion EUR",
                    "The sum of the last four quarterly growth rates",
                    "The growth over the same quarter of the previous year, $100\\,(y_t / y_{t-4} - 1)$"
                ],
                "correctExplanation": "Comparing a quarter with the same quarter a year earlier cancels the seasonal pattern; the fall from Q4 to Q1 is winter, not a recession.",
                "incorrectExplanation": "Changes over one quarter, in percent or in euros, are dominated by the season for unadjusted data, and adding quarterly rates mixes the seasonal swings; the year-on-year rate compares like with like."
            },
            "ro": {
                "title": "Rata de creștere a unei serii neajustate",
                "text": "PIB-ul neajustat al României scade cu aproximativ o treime din trimestrul IV în trimestrul I următor. Ce rată de creștere trebuie raportată?",
                "options": [
                    "Creșterea față de trimestrul anterior, $100\\,(y_t / y_{t-1} - 1)$",
                    "Diferența de ordinul întîi $y_t - y_{t-1}$, în miliarde EUR",
                    "Suma ultimelor patru rate trimestriale de creștere",
                    "Creșterea față de același trimestru al anului anterior, $100\\,(y_t / y_{t-4} - 1)$"
                ],
                "correctExplanation": "Comparația cu același trimestru din anul anterior anulează tiparul sezonier; scăderea din trimestrul IV în trimestrul I este iarna, nu o recesiune.",
                "incorrectExplanation": "Variațiile pe un trimestru, procentuale sau în euro, sînt dominate de sezonalitate în cazul datelor neajustate, iar suma ratelor trimestriale amestecă oscilațiile sezoniere; rata față de anul anterior compară perioade comparabile."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Log transformation",
                "text": "When is a logarithmic transformation appropriate for a time series?",
                "options": [
                    "When the variance grows with the level, or when the patterns are multiplicative",
                    "When the series has negative values",
                    "When the data are already stationary",
                    "When the variance should be increased"
                ],
                "correctExplanation": "The logarithm turns multiplicative patterns into additive ones and stabilises a variance that grows with the level; differences of logs are approximately percentage changes.",
                "incorrectExplanation": "The logarithm is not defined for negative values, it is not needed when the variance is already stable, and it reduces rather than increases the spread of large values."
            },
            "ro": {
                "title": "Transformarea logaritmică",
                "text": "Cînd este potrivită transformarea logaritmică a unei serii de timp?",
                "options": [
                    "Cînd varianța crește odată cu nivelul sau cînd tiparele sînt multiplicative",
                    "Cînd seria are valori negative",
                    "Cînd datele sînt deja staționare",
                    "Cînd se dorește creșterea varianței"
                ],
                "correctExplanation": "Logaritmul transformă tiparele multiplicative în tipare aditive și stabilizează o varianță care crește odată cu nivelul; diferențele logaritmilor sînt aproximativ variații procentuale.",
                "incorrectExplanation": "Logaritmul nu este definit pentru valori negative, nu este necesar cînd varianța este deja stabilă și reduce, nu crește, dispersia valorilor mari."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ACF of a trending series",
                "text": "The sample ACF of a monthly series decays very slowly and is still about 0.8 at lag 36. What does this suggest?",
                "options": [
                    "The series is independent noise",
                    "The series has a trend: its level stays close to its past values for years",
                    "The series has a seasonal period of 36 months",
                    "The series has no memory beyond one month"
                ],
                "correctExplanation": "A slow, almost linear decay of the ACF is the signature of a trend (or of a stochastic trend): values far apart in time are still on the same side of the mean.",
                "incorrectExplanation": "Independent noise has autocorrelations inside the band $\\pm 1.96/\\sqrt T$, a season shows peaks at multiples of its period, and a series without memory has $r_k$ close to 0 after the first lag."
            },
            "ro": {
                "title": "ACF-ul unei serii cu trend",
                "text": "ACF de selecție a unei serii lunare descrește foarte lent și este încă aproximativ 0,8 la lagul 36. Ce sugerează acest lucru?",
                "options": [
                    "Seria este un zgomot independent",
                    "Seria are un trend: nivelul ei rămîne apropiat de valorile trecute ani de zile",
                    "Seria are o perioadă sezonieră de 36 de luni",
                    "Seria nu are memorie dincolo de o lună"
                ],
                "correctExplanation": "O descreștere lentă, aproape liniară, a ACF este semnătura unui trend (sau a unui trend stochastic): valori îndepărtate în timp se află încă de aceeași parte a mediei.",
                "incorrectExplanation": "Un zgomot independent are autocorelațiile în interiorul benzii $\\pm 1,96/\\sqrt T$, o sezonalitate produce vîrfuri la multiplii perioadei, iar o serie fără memorie are $r_k$ aproape de 0 după primul lag."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The band of the correlogram",
                "text": "A series has $T = 400$ observations. Which band is drawn around zero on its correlogram?",
                "options": [
                    "$\\pm 1.96/400 = \\pm 0.0049$",
                    "$\\pm 1.96$",
                    "$\\pm 1.96/\\sqrt{400} = \\pm 0.098$",
                    "$\\pm 1/\\sqrt{1.96 \\times 400}$"
                ],
                "correctExplanation": "For independent noise the sample autocorrelations are approximately Normal with standard deviation $1/\\sqrt T$, so about 95% of them lie within $\\pm 1.96/\\sqrt T = \\pm 0.098$.",
                "incorrectExplanation": "The standard deviation of $r_k$ under independence is $1/\\sqrt T$, not $1/T$ or 1; the band is 1.96 times that standard deviation."
            },
            "ro": {
                "title": "Banda corelogramei",
                "text": "O serie are $T = 400$ de observații. Ce bandă se trasează în jurul lui zero pe corelograma ei?",
                "options": [
                    "$\\pm 1,96/400 = \\pm 0,0049$",
                    "$\\pm 1,96$",
                    "$\\pm 1,96/\\sqrt{400} = \\pm 0,098$",
                    "$\\pm 1/\\sqrt{1,96 \\times 400}$"
                ],
                "correctExplanation": "Pentru un zgomot independent, autocorelațiile de selecție urmează aproximativ distribuția Normală cu abaterea standard $1/\\sqrt T$, deci aproximativ 95% dintre ele se află în intervalul $\\pm 1,96/\\sqrt T = \\pm 0,098$.",
                "incorrectExplanation": "Abaterea standard a lui $r_k$ în ipoteza de independență este $1/\\sqrt T$, nu $1/T$ sau 1; banda este de 1,96 ori această abatere standard."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Slutsky's experiment",
                "text": "Slutsky replaced each of many independent random numbers by the sum of the last 10. What did he find?",
                "options": [
                    "The moving sums are again independent random numbers",
                    "The moving sums show an exact 10-period cycle",
                    "The moving sums have a linear trend",
                    "The moving sums show smooth waves that look like business cycles"
                ],
                "correctExplanation": "Neighbouring moving sums share 9 of their 10 shocks, so they are strongly correlated and form irregular waves: cycles can arise from random shocks alone.",
                "incorrectExplanation": "Overlapping sums cannot be independent; the waves have irregular lengths, not an exact period of 10; and the shocks have mean zero, so no trend appears."
            },
            "ro": {
                "title": "Experimentul lui Slutsky",
                "text": "Slutsky a înlocuit fiecare dintre numeroase numere aleatoare independente cu suma ultimelor 10. Ce a constatat?",
                "options": [
                    "Sumele mobile sînt din nou numere aleatoare independente",
                    "Sumele mobile au un ciclu exact de 10 perioade",
                    "Sumele mobile au un trend liniar",
                    "Sumele mobile formează valuri netede care seamănă cu ciclurile economice"
                ],
                "correctExplanation": "Sumele mobile vecine au în comun 9 din cele 10 șocuri, deci sînt puternic corelate și formează valuri neregulate: ciclurile pot apărea doar din șocuri aleatoare.",
                "incorrectExplanation": "Sumele care se suprapun nu pot fi independente; valurile au lungimi neregulate, nu o perioadă exactă de 10; iar șocurile au media zero, deci nu apare niciun trend."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Smoothing constant",
                "text": "What happens when the smoothing constant $\\alpha$ of simple exponential smoothing moves towards 1?",
                "options": [
                    "The forecast reacts more strongly to the most recent observations",
                    "The forecast becomes smoother and more stable",
                    "The forecast converges to the sample mean of the series",
                    "The model gains additional parameters"
                ],
                "correctExplanation": "In $\\ell_t = \\alpha y_t + (1-\\alpha)\\ell_{t-1}$ a large $\\alpha$ puts almost all the weight on the latest value; at $\\alpha = 1$ the forecast is the naive one.",
                "incorrectExplanation": "Smooth, stable forecasts come from a small $\\alpha$; the sample mean corresponds to equal weights on all observations; the number of parameters does not change with $\\alpha$."
            },
            "ro": {
                "title": "Constanta de netezire",
                "text": "Ce se întîmplă cînd constanta de netezire $\\alpha$ a netezirii exponențiale simple se apropie de 1?",
                "options": [
                    "Prognoza reacționează mai puternic la cele mai recente observații",
                    "Prognoza devine mai netedă și mai stabilă",
                    "Prognoza converge către media de selecție a seriei",
                    "Modelul capătă parametri suplimentari"
                ],
                "correctExplanation": "În $\\ell_t = \\alpha y_t + (1-\\alpha)\\ell_{t-1}$, un $\\alpha$ mare pune aproape toată ponderea pe ultima valoare; pentru $\\alpha = 1$, prognoza este cea naivă.",
                "incorrectExplanation": "Prognozele netede și stabile provin dintr-un $\\alpha$ mic; media de selecție corespunde unor ponderi egale pentru toate observațiile; numărul de parametri nu se schimbă odată cu $\\alpha$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Estimated smoothing constant close to 1",
                "text": "Simple exponential smoothing fitted to the monthly EUR/RON gives an estimated $\\hat\\alpha \\approx 1$. What does this say?",
                "options": [
                    "The method failed and the estimate should be ignored",
                    "The last value contains almost all the information: SES reduces to the naive forecast",
                    "The exchange rate has a strong seasonal pattern",
                    "The best forecast is the average of the whole sample"
                ],
                "correctExplanation": "With $\\alpha = 1$ the level equals the last observation, as for a random walk; this is typical of exchange rates and prices and is a finding about the series.",
                "incorrectExplanation": "An estimate near 1 is a valid result, not a failure; seasonality is not modelled by SES at all; and the sample average corresponds to the opposite extreme of very small weights on each observation."
            },
            "ro": {
                "title": "Constantă de netezire estimată apropiată de 1",
                "text": "Netezirea exponențială simplă aplicată cursului EUR/RON lunar conduce la $\\hat\\alpha \\approx 1$. Ce arată acest rezultat?",
                "options": [
                    "Metoda a eșuat, iar estimarea trebuie ignorată",
                    "Ultima valoare conține aproape toată informația: SES se reduce la prognoza naivă",
                    "Cursul de schimb are un tipar sezonier puternic",
                    "Cea mai bună prognoză este media întregului eșantion"
                ],
                "correctExplanation": "Pentru $\\alpha = 1$, nivelul este egal cu ultima observație, ca la un mers aleator; rezultatul este tipic pentru cursurile de schimb și prețuri și spune ceva despre serie.",
                "incorrectExplanation": "O estimare apropiată de 1 este un rezultat valid, nu un eșec; SES nu modelează deloc sezonalitatea; iar media eșantionului corespunde extremei opuse, cu ponderi foarte mici pentru fiecare observație."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Holt's linear method",
                "text": "Which components does Holt's linear method model?",
                "options": [
                    "Level only",
                    "Level, trend and seasonality",
                    "Level and trend",
                    "Seasonality only"
                ],
                "correctExplanation": "Holt (1957) adds a trend equation to simple exponential smoothing; its forecasts are a straight line $\\ell_T + h\\,b_T$.",
                "incorrectExplanation": "The level alone is simple exponential smoothing, and seasonality is added by the Holt-Winters method; Holt's method models the level and the trend."
            },
            "ro": {
                "title": "Metoda liniară Holt",
                "text": "Ce componente modelează metoda liniară Holt?",
                "options": [
                    "Doar nivelul",
                    "Nivelul, trendul și sezonalitatea",
                    "Nivelul și trendul",
                    "Doar sezonalitatea"
                ],
                "correctExplanation": "Holt (1957) adaugă netezirii exponențiale simple o ecuație pentru trend; prognozele ei formează o dreaptă $\\ell_T + h\\,b_T$.",
                "incorrectExplanation": "Doar nivelul înseamnă netezirea exponențială simplă, iar sezonalitatea este adăugată de metoda Holt-Winters; metoda Holt modelează nivelul și trendul."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Damped trend",
                "text": "What is the purpose of the damping parameter in Holt's method?",
                "options": [
                    "To make the trend grow faster",
                    "To remove seasonality",
                    "To speed up computation",
                    "To flatten the trend at long horizons and prevent unrealistic extrapolation"
                ],
                "correctExplanation": "A damped trend lets the forecast slope shrink as the horizon grows, so long-horizon forecasts level off instead of growing without limit.",
                "incorrectExplanation": "Damping slows the trend rather than accelerating it, it has nothing to do with seasonality, and it adds a parameter instead of saving computation."
            },
            "ro": {
                "title": "Trendul amortizat",
                "text": "Care este rolul parametrului de amortizare în metoda Holt?",
                "options": [
                    "Accelerează creșterea trendului",
                    "Elimină sezonalitatea",
                    "Crește viteza de calcul",
                    "Aplatizează trendul la orizonturi lungi și previne extrapolarea nerealistă"
                ],
                "correctExplanation": "Un trend amortizat face ca panta prognozei să scadă pe măsură ce orizontul crește, astfel încît prognozele pe termen lung se stabilizează în loc să crească nelimitat.",
                "incorrectExplanation": "Amortizarea încetinește trendul, nu îl accelerează, nu are legătură cu sezonalitatea și adaugă un parametru, în loc să reducă timpul de calcul."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Exponential smoothing with trend and season",
                "text": "Which exponential smoothing method should be used for a series with both trend and seasonality?",
                "options": [
                    "The Holt-Winters method",
                    "Simple exponential smoothing (SES)",
                    "A simple moving average",
                    "Holt's linear method"
                ],
                "correctExplanation": "Holt-Winters (1960) has three smoothing equations: level, trend and seasonal factors, so it captures both patterns.",
                "incorrectExplanation": "SES has a level only, Holt adds a trend but no season, and a moving average does not forecast trend or season; only Holt-Winters models both."
            },
            "ro": {
                "title": "Netezirea exponențială pentru trend și sezonalitate",
                "text": "Ce metodă de netezire exponențială trebuie folosită pentru o serie cu trend și sezonalitate?",
                "options": [
                    "Metoda Holt-Winters",
                    "Netezirea exponențială simplă (SES)",
                    "Media mobilă simplă",
                    "Metoda liniară Holt"
                ],
                "correctExplanation": "Holt-Winters (1960) are trei ecuații de netezire: nivel, trend și factori sezonieri, deci surprinde ambele tipare.",
                "incorrectExplanation": "SES are doar nivel, metoda Holt adaugă un trend, dar nu și sezonalitate, iar media mobilă nu prognozează nici trendul, nici sezonalitatea; doar Holt-Winters le modelează pe amîndouă."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ETS notation",
                "text": "In the notation ETS(A,A,A), what do the three letters stand for?",
                "options": [
                    "Average, average, average",
                    "Additive error, additive trend, additive seasonality",
                    "Three alpha parameters",
                    "Autocorrelation at lags 1, 2 and 3"
                ],
                "correctExplanation": "ETS stands for Error, Trend, Seasonal; each letter gives the form of one component: N (none), A (additive), A$_d$ (damped) or M (multiplicative).",
                "incorrectExplanation": "The letters are not averages, smoothing parameters or autocorrelations; they describe the form of the error, the trend and the seasonal component."
            },
            "ro": {
                "title": "Notația ETS",
                "text": "În notația ETS(A,A,A), ce reprezintă cele trei litere?",
                "options": [
                    "Medie, medie, medie",
                    "Eroare aditivă, trend aditiv, sezonalitate aditivă",
                    "Trei parametri alfa",
                    "Autocorelațiile la lagurile 1, 2 și 3"
                ],
                "correctExplanation": "ETS înseamnă Error, Trend, Seasonal (eroare, trend, sezonalitate); fiecare literă indică forma unei componente: N (absentă), A (aditivă), A$_d$ (amortizată) sau M (multiplicativă).",
                "incorrectExplanation": "Literele nu sînt medii, parametri de netezire sau autocorelații; ele descriu forma erorii, a trendului și a componentei sezoniere."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Naive forecast",
                "text": "What is the naive forecast and why is it useful?",
                "options": [
                    "The most complex model available; useful for maximum accuracy",
                    "The mean of all observations; useful for trending data",
                    "The last observation used as the forecast; a benchmark that a model must beat",
                    "A random forecast; useful for testing"
                ],
                "correctExplanation": "The naive forecast $\\hat y_{T+h} = y_T$ is optimal for a random walk and costs nothing; a model that cannot beat it on a test set is not worth using.",
                "incorrectExplanation": "The naive forecast is the simplest method, not the most complex; the mean forecast performs badly on trending data; and nothing in it is random."
            },
            "ro": {
                "title": "Prognoza naivă",
                "text": "Ce este prognoza naivă și de ce este utilă?",
                "options": [
                    "Cel mai complex model disponibil; utilă pentru acuratețe maximă",
                    "Media tuturor observațiilor; utilă pentru date cu trend",
                    "Ultima observație folosită ca prognoză; un reper pe care un model trebuie să-l depășească",
                    "O prognoză aleatoare; utilă pentru testare"
                ],
                "correctExplanation": "Prognoza naivă $\\hat y_{T+h} = y_T$ este optimă pentru un mers aleator și nu costă nimic; un model care nu o depășește pe un set de test nu merită folosit.",
                "incorrectExplanation": "Prognoza naivă este cea mai simplă metodă, nu cea mai complexă; prognoza prin medie funcționează prost pe datele cu trend; iar în ea nu există nimic aleator."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Seasonal naive forecast",
                "text": "For monthly data, what does the seasonal naive method forecast for next January?",
                "options": [
                    "The value of last December",
                    "The average of all months",
                    "Zero",
                    "The value of last January"
                ],
                "correctExplanation": "The seasonal naive forecast is $\\hat y_{T+h} = y_{T+h-m}$: the value of the same season one period earlier, here last January.",
                "incorrectExplanation": "Last December is the naive forecast, the average of all months is the mean forecast, and zero is not a forecast of a seasonal level; the seasonal naive method copies the same month of the previous year."
            },
            "ro": {
                "title": "Prognoza naivă sezonieră",
                "text": "Pentru date lunare, ce prognozează metoda naivă sezonieră pentru luna ianuarie următoare?",
                "options": [
                    "Valoarea din decembrie anterior",
                    "Media tuturor lunilor",
                    "Zero",
                    "Valoarea din ianuarie anterior"
                ],
                "correctExplanation": "Prognoza naivă sezonieră este $\\hat y_{T+h} = y_{T+h-m}$: valoarea aceluiași sezon cu o perioadă mai devreme, aici ianuarie anterior.",
                "incorrectExplanation": "Decembrie anterior este prognoza naivă, media lunilor este prognoza prin medie, iar zero nu este o prognoză a unui nivel sezonier; metoda naivă sezonieră copiază aceeași lună din anul anterior."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Training and test split",
                "text": "In time series forecasting, how should the data be split into a training set and a test set?",
                "options": [
                    "The earlier observations form the training set and the later ones the test set",
                    "The observations are shuffled at random and then split",
                    "Training and test observations alternate",
                    "The middle part of the sample is used for testing"
                ],
                "correctExplanation": "A forecast may use only the past; keeping the last observations hidden reproduces the real situation of forecasting the future.",
                "incorrectExplanation": "Random, alternating or middle splits let the method see observations that come after the ones it forecasts, so the errors look smaller than they will be in practice."
            },
            "ro": {
                "title": "Setul de antrenare și setul de test",
                "text": "În prognoza seriilor de timp, cum se împart datele în set de antrenare și set de test?",
                "options": [
                    "Observațiile mai vechi formează setul de antrenare, iar cele mai recente setul de test",
                    "Observațiile se amestecă aleator și apoi se împart",
                    "Observațiile de antrenare și de test alternează",
                    "Partea din mijloc a eșantionului se folosește pentru test"
                ],
                "correctExplanation": "O prognoză poate folosi doar trecutul; ascunderea ultimelor observații reproduce situația reală a prognozei viitorului.",
                "incorrectExplanation": "Împărțirile aleatoare, alternante sau la mijloc îi permit metodei să vadă observații care urmează celor prognozate, deci erorile par mai mici decît vor fi în practică."
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
                    "When the data contain many outliers that should be ignored",
                    "When a percentage measure is needed"
                ],
                "correctExplanation": "RMSE squares the errors before averaging, so a few large errors raise it much more than they raise the MAE.",
                "incorrectExplanation": "Equal weighting of all errors is the MAE; squaring makes RMSE more, not less, sensitive to outliers; and neither measure is a percentage."
            },
            "ro": {
                "title": "RMSE și MAE",
                "text": "Cînd este preferată RMSE în locul MAE?",
                "options": [
                    "Cînd toate erorile trebuie ponderate egal",
                    "Cînd erorile mari sînt deosebit de costisitoare și trebuie penalizate mai mult",
                    "Cînd datele conțin multe valori extreme care trebuie ignorate",
                    "Cînd este nevoie de o măsură procentuală"
                ],
                "correctExplanation": "RMSE ridică erorile la pătrat înainte de mediere, deci cîteva erori mari o cresc mult mai mult decît cresc MAE.",
                "incorrectExplanation": "Ponderarea egală a tuturor erorilor corespunde MAE; ridicarea la pătrat face RMSE mai sensibilă, nu mai puțin sensibilă, la valori extreme; iar niciuna dintre cele două nu este procentuală."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "MASE",
                "text": "A forecasting method has MASE = 0.5 on the test set. What does this mean?",
                "options": [
                    "Half of its forecasts are correct",
                    "Its forecasts are off by 0.5% on average",
                    "Its MAE is half the in-sample MAE of the seasonal naive method",
                    "It explains 50% of the variance of the test set"
                ],
                "correctExplanation": "MASE divides the MAE on the test set by the in-sample MAE of the seasonal naive method; values below 1 beat that benchmark, and the measure is comparable across series.",
                "incorrectExplanation": "MASE is a ratio of mean absolute errors: it does not count correct forecasts, it is not a percentage error, and it is not a share of explained variance."
            },
            "ro": {
                "title": "MASE",
                "text": "O metodă de prognoză are MASE = 0,5 pe setul de test. Ce înseamnă acest lucru?",
                "options": [
                    "Jumătate dintre prognozele ei sînt corecte",
                    "Prognozele ei greșesc în medie cu 0,5%",
                    "MAE-ul ei este jumătate din MAE-ul în eșantion al metodei naive sezoniere",
                    "Explică 50% din varianța setului de test"
                ],
                "correctExplanation": "MASE împarte MAE de pe setul de test la MAE în eșantion a metodei naive sezoniere; valorile sub 1 depășesc acest reper, iar măsura este comparabilă între serii.",
                "incorrectExplanation": "MASE este un raport între erori absolute medii: nu numără prognozele corecte, nu este o eroare procentuală și nu este o pondere a varianței explicate."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Forecast horizon",
                "text": "What typically happens to forecast accuracy as the forecast horizon grows?",
                "options": [
                    "It improves, because more data are averaged",
                    "It stays constant",
                    "It depends only on the model, not on the horizon",
                    "It decreases, because uncertainty accumulates"
                ],
                "correctExplanation": "Each additional step adds unknown shocks, so the forecast errors and the width of the forecast intervals grow with $h$.",
                "incorrectExplanation": "Longer horizons do not average more data; accuracy depends on both the model and the horizon, and for almost every series it worsens as the horizon grows."
            },
            "ro": {
                "title": "Orizontul de prognoză",
                "text": "Ce se întîmplă de obicei cu acuratețea prognozei pe măsură ce orizontul crește?",
                "options": [
                    "Crește, deoarece se mediază mai multe date",
                    "Rămîne constantă",
                    "Depinde doar de model, nu și de orizont",
                    "Scade, deoarece incertitudinea se acumulează"
                ],
                "correctExplanation": "Fiecare pas suplimentar adaugă șocuri necunoscute, deci erorile de prognoză și lățimea intervalelor de prognoză cresc odată cu $h$.",
                "incorrectExplanation": "Orizonturile mai lungi nu mediază mai multe date; acuratețea depinde atît de model, cît și de orizont și, pentru aproape orice serie, se înrăutățește cînd orizontul crește."
            }
        }
    ]
};
