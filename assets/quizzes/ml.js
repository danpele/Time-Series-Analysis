// ============================================================
// Chapter 9 quiz bank: Machine learning for time series (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['ml'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "Feature engineering for machine learning",
                "text": "To apply a random forest to time series forecasting, we must create:",
                "options": [
                    "Lag features and rolling statistics",
                    "A dummy variable for each observation",
                    "Fourier transforms of the series only",
                    "Only the first difference of the series"
                ],
                "correctExplanation": "A random forest has no notion of time order, so the series must be turned into a supervised learning table: lag features ($y_{t-1}, y_{t-2}, \\ldots$), rolling means and standard deviations, and calendar features (day of week, month).",
                "incorrectExplanation": "One dummy per observation would make every row unique and the model could not generalise; Fourier terms or a first difference may be useful extra inputs but on their own do not describe the recent dynamics. The core inputs are lags and rolling statistics."
            },
            "ro": {
                "title": "Feature engineering pentru machine learning",
                "text": "Pentru a aplica un random forest la prognoza seriilor de timp, trebuie să construim:",
                "options": [
                    "Variabile lag și statistici pe ferestre mobile",
                    "O variabilă dummy pentru fiecare observație",
                    "Doar transformate Fourier ale seriei",
                    "Doar prima diferență a seriei"
                ],
                "correctExplanation": "Un random forest nu cunoaște ordinea temporală, deci seria trebuie transformată într-un tabel de învățare supervizată: variabile lag ($y_{t-1}, y_{t-2}, \\ldots$), medii și abateri standard pe ferestre mobile, precum și variabile calendaristice (ziua săptămînii, luna).",
                "incorrectExplanation": "O variabilă dummy pentru fiecare observație ar face fiecare rînd unic, iar modelul nu ar putea generaliza; termenii Fourier sau prima diferență pot fi intrări suplimentare utile, dar singure nu descriu dinamica recentă. Intrările de bază sînt lag-urile și statisticile pe ferestre mobile."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Time series cross-validation",
                "text": "Why can't standard k-fold cross-validation be used for time series?",
                "options": [
                    "It is too slow for long series",
                    "It only works for classification",
                    "It requires too much data",
                    "It violates the temporal order and causes data leakage"
                ],
                "correctExplanation": "Standard k-fold assigns observations to folds at random, so the model is trained on observations that come after the validation fold. This leaks future information and overstates performance. Walk-forward validation (for example TimeSeriesSplit in scikit-learn) keeps the order.",
                "incorrectExplanation": "Speed, the type of task and sample size are not the issue: k-fold works for regression and for any sample size. The problem is that it breaks the temporal order; walk-forward validation trains only on the past."
            },
            "ro": {
                "title": "Validarea încrucișată pentru serii de timp",
                "text": "De ce nu poate fi folosită validarea încrucișată k-fold standard pentru serii de timp?",
                "options": [
                    "Este prea lentă pentru serii lungi",
                    "Funcționează doar pentru clasificare",
                    "Necesită prea multe date",
                    "Încalcă ordinea temporală și produce scurgere de informație (data leakage)"
                ],
                "correctExplanation": "K-fold standard repartizează aleator observațiile în subeșantioane, astfel încît modelul este antrenat pe observații ulterioare subeșantionului de validare. Se scurge astfel informație din viitor, iar performanța este supraestimată. Validarea walk-forward (de exemplu TimeSeriesSplit din scikit-learn) păstrează ordinea.",
                "incorrectExplanation": "Viteza, tipul problemei și volumul de date nu sînt cauza: k-fold funcționează și pentru regresie, și pentru orice volum de date. Problema este încălcarea ordinii temporale; validarea walk-forward antrenează modelul doar pe trecut."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Advantage of LSTM",
                "text": "What is the main advantage of an LSTM over a simple recurrent neural network (RNN)?",
                "options": [
                    "It is faster to train",
                    "It requires less data",
                    "It mitigates the vanishing gradient problem and can learn long-range dependencies",
                    "It is easier to interpret"
                ],
                "correctExplanation": "The LSTM cell state, controlled by the forget, input and output gates, is updated additively, so gradients can flow over many time steps without vanishing. This lets the network learn long-range dependencies. Exploding gradients are usually handled separately, by gradient clipping.",
                "incorrectExplanation": "An LSTM has more parameters than a simple RNN, so it is slower to train, typically needs more data and is no easier to interpret. Its advantage is the gated cell state, which mitigates vanishing gradients."
            },
            "ro": {
                "title": "Avantajul LSTM",
                "text": "Care este principalul avantaj al unei rețele LSTM față de o rețea neuronală recurentă (RNN) simplă?",
                "options": [
                    "Se antrenează mai rapid",
                    "Necesită mai puține date",
                    "Atenuează problema gradienților care se anulează (vanishing gradient) și poate învăța dependențe pe termen lung",
                    "Este mai ușor de interpretat"
                ],
                "correctExplanation": "Starea celulei LSTM, controlată de porțile de uitare, de intrare și de ieșire, se actualizează aditiv, astfel încît gradienții se pot propaga pe mulți pași de timp fără să se anuleze. Rețeaua poate învăța astfel dependențe pe termen lung. Gradienții care explodează se tratează de obicei separat, prin gradient clipping.",
                "incorrectExplanation": "O rețea LSTM are mai mulți parametri decît o RNN simplă, deci se antrenează mai lent, are nevoie de regulă de mai multe date și nu este mai ușor de interpretat. Avantajul ei este starea celulei controlată de porți, care atenuează anularea gradienților."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Random forest as an ensemble",
                "text": "Compared with a single decision tree, a random forest reduces overfitting by:",
                "options": [
                    "Using deeper trees",
                    "Averaging the predictions of many trees trained on bootstrap samples",
                    "Using fewer observations",
                    "Training only on the full data set"
                ],
                "correctExplanation": "Bagging (bootstrap aggregating) plus a random subset of features at each split makes the trees diverse; averaging many weakly correlated trees reduces variance while leaving the bias roughly unchanged.",
                "incorrectExplanation": "Deeper trees increase overfitting, discarding observations removes information, and a single fit on the full data set is exactly what a lone tree does. The gain comes from averaging many trees grown on bootstrap samples with random feature subsets."
            },
            "ro": {
                "title": "Random forest ca ansamblu",
                "text": "Comparativ cu un singur arbore de decizie, un random forest reduce overfitting-ul prin:",
                "options": [
                    "Folosirea unor arbori mai adînci",
                    "Medierea predicțiilor mai multor arbori antrenați pe eșantioane bootstrap",
                    "Folosirea unui număr mai mic de observații",
                    "Antrenarea exclusiv pe întregul set de date"
                ],
                "correctExplanation": "Bagging-ul (bootstrap aggregating), împreună cu alegerea aleatoare a unui subset de variabile la fiecare ramificare, face arborii diferiți între ei; medierea multor arbori slab corelați reduce varianța, lăsînd deplasarea (bias) aproximativ neschimbată.",
                "incorrectExplanation": "Arborii mai adînci accentuează overfitting-ul, renunțarea la observații pierde informație, iar o singură estimare pe întregul set de date este exact ce face un arbore izolat. Cîștigul vine din medierea multor arbori crescuți pe eșantioane bootstrap, cu subseturi aleatoare de variabile."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Data leakage",
                "text": "Which of the following is an example of data leakage in machine learning for time series?",
                "options": [
                    "Scaling the data with statistics computed on the whole sample (training and test)",
                    "Using lag features",
                    "Using rolling-window features",
                    "Splitting the data chronologically"
                ],
                "correctExplanation": "If the mean, standard deviation, minimum or maximum used for scaling are computed on the whole sample, the training data carry information about the test period. The scaler must be fitted on the training data only and then applied to both sets.",
                "incorrectExplanation": "Lag and rolling-window features use only past values, so they are legitimate, and a chronological split is the correct practice. Leakage occurs when statistics from the test period enter the training step, as in full-sample scaling."
            },
            "ro": {
                "title": "Scurgerea de informație",
                "text": "Care dintre următoarele este un exemplu de scurgere de informație (data leakage) în machine learning pentru serii de timp?",
                "options": [
                    "Scalarea datelor cu statistici calculate pe întregul eșantion (antrenare și test)",
                    "Folosirea variabilelor lag",
                    "Folosirea variabilelor pe ferestre mobile",
                    "Împărțirea cronologică a datelor"
                ],
                "correctExplanation": "Dacă media, abaterea standard, minimul sau maximul folosite la scalare se calculează pe întregul eșantion, datele de antrenare conțin informație despre perioada de test. Scalarea trebuie estimată doar pe datele de antrenare și apoi aplicată ambelor seturi.",
                "incorrectExplanation": "Variabilele lag și cele pe ferestre mobile folosesc doar valori trecute, deci sînt legitime, iar împărțirea cronologică este practica corectă. Scurgerea apare cînd statistici din perioada de test intră în etapa de antrenare, ca la scalarea pe întregul eșantion."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "LSTM gates",
                "text": "Which LSTM gate decides what information to discard from the cell state?",
                "options": [
                    "The input gate",
                    "The output gate",
                    "The update gate",
                    "The forget gate"
                ],
                "correctExplanation": "The forget gate $f_t = \\sigma(W_f[h_{t-1}, x_t] + b_f)$ multiplies the previous cell state: values close to 0 mean forget, values close to 1 mean keep.",
                "incorrectExplanation": "The input gate controls what new information enters the cell state, the output gate controls what is exposed as the hidden state, and the update gate belongs to the GRU, not the LSTM. Discarding information is the role of the forget gate."
            },
            "ro": {
                "title": "Porțile LSTM",
                "text": "Ce poartă a unei rețele LSTM decide ce informație se elimină din starea celulei?",
                "options": [
                    "Poarta de intrare (input gate)",
                    "Poarta de ieșire (output gate)",
                    "Poarta de actualizare (update gate)",
                    "Poarta de uitare (forget gate)"
                ],
                "correctExplanation": "Poarta de uitare $f_t = \\sigma(W_f[h_{t-1}, x_t] + b_f)$ înmulțește starea anterioară a celulei: valorile apropiate de 0 înseamnă uitare, iar cele apropiate de 1 înseamnă păstrare.",
                "incorrectExplanation": "Poarta de intrare controlează ce informație nouă intră în starea celulei, poarta de ieșire controlează ce se transmite în starea ascunsă, iar poarta de actualizare aparține rețelei GRU, nu LSTM. Eliminarea informației este rolul porții de uitare."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Scaling the inputs of a network",
                "text": "Before training a neural network (an MLP or an LSTM), the inputs should be:",
                "options": [
                    "Log-transformed",
                    "Differenced twice",
                    "Scaled, for example to $[0,1]$ or $[-1,1]$, or standardised",
                    "Converted to integers"
                ],
                "correctExplanation": "The sigmoid and tanh activations work on bounded ranges, so scaled inputs give faster convergence and numerical stability. The scaler must be fitted on the training data only and then applied to the validation and test data.",
                "incorrectExplanation": "A log transform or differencing may be useful in specific cases but is not the generic requirement, and integer conversion only loses information. The standard step is scaling (MinMaxScaler or StandardScaler) fitted on the training set."
            },
            "ro": {
                "title": "Scalarea intrărilor unei rețele",
                "text": "Înainte de antrenarea unei rețele neuronale (MLP sau LSTM), intrările trebuie:",
                "options": [
                    "Transformate logaritmic",
                    "Diferențiate de două ori",
                    "Scalate, de exemplu la $[0,1]$ sau $[-1,1]$, ori standardizate",
                    "Convertite în numere întregi"
                ],
                "correctExplanation": "Funcțiile de activare sigmoid și tanh lucrează pe intervale mărginite, deci intrările scalate asigură o convergență mai rapidă și stabilitate numerică. Scalarea se estimează doar pe datele de antrenare și apoi se aplică datelor de validare și de test.",
                "incorrectExplanation": "Transformarea logaritmică sau diferențierea pot fi utile în anumite situații, dar nu sînt cerința generală, iar conversia în numere întregi doar pierde informație. Pasul standard este scalarea (MinMaxScaler sau StandardScaler) estimată pe setul de antrenare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Early stopping",
                "text": "Early stopping monitors the validation loss in order to:",
                "options": [
                    "Speed up each iteration",
                    "Stop training when the model starts to overfit",
                    "Select the best features",
                    "Adjust the learning rate"
                ],
                "correctExplanation": "When the validation loss stops improving for a given number of epochs (the patience), training stops and the best weights are kept. This prevents the network from fitting noise in the training data.",
                "incorrectExplanation": "Early stopping does not make individual iterations faster, does not select features and does not change the learning rate (that is the job of a learning-rate scheduler). It halts training once validation performance stops improving."
            },
            "ro": {
                "title": "Oprirea timpurie",
                "text": "Oprirea timpurie (early stopping) urmărește pierderea pe setul de validare pentru a:",
                "options": [
                    "Accelera fiecare iterație",
                    "Opri antrenarea atunci cînd modelul începe să facă overfitting",
                    "Selecta cele mai bune variabile",
                    "Ajusta rata de învățare"
                ],
                "correctExplanation": "Cînd pierderea pe setul de validare nu se mai îmbunătățește timp de un număr dat de epoci (patience), antrenarea se oprește și se păstrează cele mai bune ponderi. Astfel rețeaua nu ajunge să modeleze zgomotul din datele de antrenare.",
                "incorrectExplanation": "Early stopping nu accelerează iterațiile, nu selectează variabile și nu modifică rata de învățare (acesta este rolul unui learning-rate scheduler). El oprește antrenarea cînd performanța pe setul de validare nu se mai îmbunătățește."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Recursive and direct forecasts",
                "text": "A forecaster needs forecasts for horizons 1 to 14 days. Which description matches the direct strategy?",
                "options": [
                    "A separate model is trained for each horizon $h$, with target $y_{t+h}$",
                    "One one-step model is applied 14 times, feeding its forecasts back as inputs",
                    "One model returns the vector of all 14 forecasts at once",
                    "The last observed value is used for every horizon"
                ],
                "correctExplanation": "The direct strategy fits one model $\\hat f_h$ per horizon on the table whose target is $y_{t+h}$. Errors do not accumulate through the horizons, at the cost of $H$ models.",
                "incorrectExplanation": "Feeding forecasts back is the recursive strategy, one model with $H$ outputs is the multi-output (MIMO) strategy, and repeating the last value is the naive forecast. The direct strategy trains one model per horizon."
            },
            "ro": {
                "title": "Prognoze recursive și directe",
                "text": "Avem nevoie de prognoze pentru orizonturile de 1 pînă la 14 zile. Ce descriere corespunde strategiei directe?",
                "options": [
                    "Pentru fiecare orizont $h$ se antrenează un model separat, cu ținta $y_{t+h}$",
                    "Un singur model pe un pas se aplică de 14 ori, cu propriile prognoze ca intrări",
                    "Un singur model întoarce deodată vectorul celor 14 prognoze",
                    "Ultima valoare observată se folosește pentru toate orizonturile"
                ],
                "correctExplanation": "Strategia directă estimează cîte un model $\\hat f_h$ pentru fiecare orizont, pe tabelul a cărui țintă este $y_{t+h}$. Erorile nu se acumulează de la un orizont la altul, cu prețul a $H$ modele.",
                "incorrectExplanation": "Folosirea prognozelor ca intrări este strategia recursivă, un model cu $H$ ieșiri este strategia MIMO, iar repetarea ultimei valori este prognoza naivă. Strategia directă antrenează cîte un model pentru fiecare orizont."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Rows of a lag table",
                "text": "A series has $n = 100$ observations. With $p = 7$ lags ($y_t, \\dots, y_{t-6}$) and horizon $h = 3$, how many complete rows does the supervised table have?",
                "options": [
                    "93",
                    "97",
                    "100",
                    "91"
                ],
                "correctExplanation": "The first complete origin is $t = 7$ (it needs $y_1, \\dots, y_7$) and the last is $t = 97$ (its target is $y_{100}$): $n - p - h + 1 = 100 - 7 - 3 + 1 = 91$ rows.",
                "incorrectExplanation": "93 forgets the horizon, 97 forgets the lags and 100 ignores both. The count is $n - p - h + 1 = 91$."
            },
            "ro": {
                "title": "Rîndurile unui tabel de decalaje",
                "text": "O serie are $n = 100$ de observații. Cu $p = 7$ decalaje ($y_t, \\dots, y_{t-6}$) și orizontul $h = 3$, cîte rînduri complete are tabelul de învățare supervizată?",
                "options": [
                    "93",
                    "97",
                    "100",
                    "91"
                ],
                "correctExplanation": "Prima origine completă este $t = 7$ (are nevoie de $y_1, \\dots, y_7$), iar ultima este $t = 97$ (ținta ei este $y_{100}$): $n - p - h + 1 = 100 - 7 - 3 + 1 = 91$ de rînduri.",
                "incorrectExplanation": "93 ignoră orizontul, 97 ignoră decalajele, iar 100 le ignoră pe amîndouă. Numărul este $n - p - h + 1 = 91$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "A centred moving average as a feature",
                "text": "A 7-day moving average centred on day $t$ is used as a feature to forecast $y_{t+1}$. What is the problem?",
                "options": [
                    "Moving averages cannot be used by tree-based models",
                    "A 7-day window is too short for daily data",
                    "It contains $y_{t+1}$, $y_{t+2}$ and $y_{t+3}$: future values leak into the features",
                    "There is no problem, because the average smooths the noise"
                ],
                "correctExplanation": "A centred window uses three values after $t$, including the target itself. Validation scores become excellent and disappear in real use. A trailing mean of $y_{t-6}, \\dots, y_t$ is the correct feature.",
                "incorrectExplanation": "Trees can use any numeric feature and the window length is a tuning choice; smoothing does not remove the leak. The centred window contains future values, including the target."
            },
            "ro": {
                "title": "O medie mobilă centrată ca variabilă explicativă",
                "text": "O medie mobilă pe 7 zile, centrată în ziua $t$, este folosită ca variabilă pentru prognoza lui $y_{t+1}$. Care este problema?",
                "options": [
                    "Mediile mobile nu pot fi folosite de modelele bazate pe arbori",
                    "O fereastră de 7 zile este prea scurtă pentru date zilnice",
                    "Conține $y_{t+1}$, $y_{t+2}$ și $y_{t+3}$: valori viitoare intră în variabilele explicative",
                    "Nu există nicio problemă, deoarece media netezește zgomotul"
                ],
                "correctExplanation": "O fereastră centrată folosește trei valori de după $t$, inclusiv ținta. Scorurile la validare devin excelente și dispar la utilizarea reală. Variabila corectă este media valorilor $y_{t-6}, \\dots, y_t$.",
                "incorrectExplanation": "Arborii pot folosi orice variabilă numerică, iar lungimea ferestrei este o alegere de reglaj; netezirea nu elimină scurgerea de informație. Fereastra centrată conține valori viitoare, inclusiv ținta."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Trees and trending series",
                "text": "A random forest is trained on the monthly level of a price index up to 2020 (maximum 111) and forecasts it one month ahead until 2026, when the index reaches 173. What happens?",
                "options": [
                    "The forecasts follow the index, because random forests extrapolate trends",
                    "The forecasts stay near the largest level seen in training, about 111",
                    "The forecasts become negative",
                    "The forest refuses to forecast outside the training range"
                ],
                "correctExplanation": "A tree predicts means of training targets, so its forecast can never exceed the largest training value. On the Romanian HICP the forest on levels stayed near 110.8 (RMSE 36.9); the same forest on monthly changes had RMSE 0.71.",
                "incorrectExplanation": "Random forests do not extrapolate trends, cannot produce values below the smallest target here, and give a forecast for any input. The forecast stays at the training maximum: model changes, not levels."
            },
            "ro": {
                "title": "Arborii și seriile cu trend",
                "text": "Un random forest este antrenat pe nivelul lunar al unui indice de prețuri pînă în 2020 (maximum 111) și îl prognozează cu o lună înainte pînă în 2026, cînd indicele ajunge la 173. Ce se întîmplă?",
                "options": [
                    "Prognozele urmăresc indicele, deoarece random forest extrapolează trendurile",
                    "Prognozele rămîn în jurul celui mai mare nivel din antrenare, circa 111",
                    "Prognozele devin negative",
                    "Modelul refuză să prognozeze în afara intervalului de antrenare"
                ],
                "correctExplanation": "Un arbore prognozează medii ale țintelor din antrenare, deci prognoza lui nu poate depăși cea mai mare valoare de antrenare. Pentru IAPC-ul României, random forest pe niveluri a rămas în jurul valorii 110,8 (RMSE 36,9); același model pe variațiile lunare a avut RMSE 0,71.",
                "incorrectExplanation": "Random forest nu extrapolează trendurile, nu poate da aici valori sub cea mai mică țintă și oferă o prognoză pentru orice intrare. Prognoza rămîne la maximul din antrenare: modelăm variații, nu niveluri."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Ridge and lasso",
                "text": "Which statement about ridge and lasso regression on lag features is correct?",
                "options": [
                    "The lasso can set some coefficients exactly to zero; ridge only shrinks them",
                    "Ridge can set some coefficients exactly to zero; the lasso only shrinks them",
                    "Both set the same coefficients to zero for any penalty",
                    "Neither needs standardised features"
                ],
                "correctExplanation": "The absolute-value penalty $\\lambda\\sum|\\beta_j|$ has a kink at zero, so the lasso selects lags; the squared penalty $\\lambda\\sum\\beta_j^2$ of ridge shrinks every coefficient smoothly towards zero.",
                "incorrectExplanation": "Ridge never produces exact zeros, the two penalties select differently, and both penalties depend on the scale of the features, so the features must be standardised. Only the lasso performs selection."
            },
            "ro": {
                "title": "Ridge și lasso",
                "text": "Ce afirmație despre regresia ridge și lasso pe decalaje este corectă?",
                "options": [
                    "Lasso poate face unii coeficienți exact zero; ridge doar îi contractă",
                    "Ridge poate face unii coeficienți exact zero; lasso doar îi contractă",
                    "Ambele fac zero aceiași coeficienți, pentru orice penalizare",
                    "Niciuna nu are nevoie de variabile standardizate"
                ],
                "correctExplanation": "Penalizarea cu valoarea absolută, $\\lambda\\sum|\\beta_j|$, are un colț în zero, deci lasso selectează decalajele; penalizarea pătratică $\\lambda\\sum\\beta_j^2$ a regresiei ridge contractă lin toți coeficienții spre zero.",
                "incorrectExplanation": "Ridge nu produce niciodată zerouri exacte, cele două penalizări selectează diferit, iar ambele depind de scala variabilelor, deci variabilele trebuie standardizate. Doar lasso face selecție."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Ridge in one dimension",
                "text": "One standardised feature has $S_{xy} = \\sum x_ty_t = 90$ and $S_{xx} = \\sum x_t^2 = 100$. What is the ridge coefficient with $\\lambda = 50$?",
                "options": [
                    "0.90",
                    "0.70",
                    "1.80",
                    "0.60"
                ],
                "correctExplanation": "Ridge in one dimension: $\\hat\\beta = S_{xy}/(S_{xx} + \\lambda) = 90/150 = 0.60$, against the OLS value $90/100 = 0.90$.",
                "incorrectExplanation": "0.90 is the OLS coefficient, 0.70 is the lasso coefficient with $\\lambda = 40$, and 1.80 divides by $\\lambda$ alone. The ridge coefficient is $90/(100 + 50) = 0.60$."
            },
            "ro": {
                "title": "Ridge într-o dimensiune",
                "text": "O variabilă standardizată are $S_{xy} = \\sum x_ty_t = 90$ și $S_{xx} = \\sum x_t^2 = 100$. Care este coeficientul ridge pentru $\\lambda = 50$?",
                "options": [
                    "0,90",
                    "0,70",
                    "1,80",
                    "0,60"
                ],
                "correctExplanation": "Ridge într-o dimensiune: $\\hat\\beta = S_{xy}/(S_{xx} + \\lambda) = 90/150 = 0{,}60$, față de valoarea OLS $90/100 = 0{,}90$.",
                "incorrectExplanation": "0,90 este coeficientul OLS, 0,70 este coeficientul lasso pentru $\\lambda = 40$, iar 1,80 împarte doar la $\\lambda$. Coeficientul ridge este $90/(100 + 50) = 0{,}60$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The learning rate in gradient boosting",
                "text": "In gradient boosting, the learning rate $\\nu$ is reduced from 0.3 to 0.05. What usually happens?",
                "options": [
                    "Fewer trees are needed to reach the same validation error",
                    "The model can no longer overfit, whatever the number of trees",
                    "More trees are needed, and the best validation error is usually slightly lower",
                    "The learning rate has no effect on the forecasts"
                ],
                "correctExplanation": "Each tree adds only $\\nu$ times its correction, so a small $\\nu$ needs more trees. On the load data the best validation MSE was 0.110 after 72 trees with $\\nu = 0.3$ and 0.098 after 256 trees with $\\nu = 0.05$.",
                "incorrectExplanation": "A smaller step needs more, not fewer, trees; with enough trees the model can still overfit, which is why early stopping is used; and $\\nu$ changes the path of the fit. A small learning rate trades speed for a slightly better optimum."
            },
            "ro": {
                "title": "Rata de învățare în gradient boosting",
                "text": "În gradient boosting, rata de învățare $\\nu$ scade de la 0,3 la 0,05. Ce se întîmplă de obicei?",
                "options": [
                    "Sînt necesari mai puțini arbori pentru aceeași eroare la validare",
                    "Modelul nu mai poate face overfitting, oricare ar fi numărul de arbori",
                    "Sînt necesari mai mulți arbori, iar cea mai bună eroare la validare este de obicei puțin mai mică",
                    "Rata de învățare nu influențează prognozele"
                ],
                "correctExplanation": "Fiecare arbore adaugă doar $\\nu$ din corecția lui, deci un $\\nu$ mic cere mai mulți arbori. Pe datele de consum, cea mai bună MSE la validare a fost 0,110 după 72 de arbori cu $\\nu = 0{,}3$ și 0,098 după 256 de arbori cu $\\nu = 0{,}05$.",
                "incorrectExplanation": "Un pas mai mic cere mai mulți arbori, nu mai puțini; cu destui arbori modelul poate face în continuare overfitting, de aceea se folosește oprirea timpurie; iar $\\nu$ schimbă traiectoria estimării. O rată de învățare mică sacrifică viteza pentru un optim puțin mai bun."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Permutation importance",
                "text": "How is the permutation importance of a feature measured?",
                "options": [
                    "By counting how often the feature is used in the splits of the training trees",
                    "By shuffling the feature in the test data and measuring how much the forecast error increases",
                    "By the size of its coefficient in a linear regression",
                    "By the correlation between the feature and the target"
                ],
                "correctExplanation": "Shuffling breaks the link between the feature and the target while keeping its distribution; the increase in the test error measures how much the model relies on it, out of sample and for any type of model.",
                "incorrectExplanation": "Split counts and impurity decreases are computed in training and favour continuous features; coefficients and correlations describe linear relations only. Permutation importance is the increase in test error after shuffling."
            },
            "ro": {
                "title": "Importanța prin permutare",
                "text": "Cum se măsoară importanța prin permutare a unei variabile?",
                "options": [
                    "Numărăm de cîte ori variabila apare în împărțirile arborilor de antrenare",
                    "Amestecăm variabila în datele de test și măsurăm cît crește eroarea de prognoză",
                    "Prin mărimea coeficientului ei într-o regresie liniară",
                    "Prin corelația dintre variabilă și țintă"
                ],
                "correctExplanation": "Amestecarea rupe legătura dintre variabilă și țintă, păstrîndu-i distribuția; creșterea erorii de test arată cît se bazează modelul pe ea, în afara eșantionului și pentru orice tip de model.",
                "incorrectExplanation": "Numărul de împărțiri și scăderea impurității se calculează la antrenare și favorizează variabilele continue; coeficienții și corelațiile descriu doar relații liniare. Importanța prin permutare este creșterea erorii de test după amestecare."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Reading a MASE",
                "text": "On the 26 load origins of Chapter 4, gradient boosting has MASE 0.97 and DHR has MASE 0.89. Which reading is correct?",
                "options": [
                    "Both beat the in-sample seasonal naive method, and DHR is more accurate than gradient boosting",
                    "Gradient boosting is 3% more accurate than DHR",
                    "Both are worse than the seasonal naive method",
                    "MASE values cannot be compared across models"
                ],
                "correctExplanation": "MASE divides the MAE by the in-sample MAE of the seasonal naive method; values below 1 beat that benchmark, and on the same test data the smaller MASE is the more accurate model.",
                "incorrectExplanation": "MASE is not measured relative to DHR, values below 1 mean better than the benchmark, and on a common scale MASE values of different models are directly comparable. DHR has the smaller MASE."
            },
            "ro": {
                "title": "Interpretarea MASE",
                "text": "Pe cele 26 de origini pentru consum din Capitolul 4, gradient boosting are MASE 0,97 și DHR are MASE 0,89. Ce interpretare este corectă?",
                "options": [
                    "Ambele depășesc metoda naivă sezonieră în eșantion, iar DHR este mai precis decît gradient boosting",
                    "Gradient boosting este cu 3% mai precis decît DHR",
                    "Ambele sînt mai slabe decît metoda naivă sezonieră",
                    "Valorile MASE nu se pot compara între modele"
                ],
                "correctExplanation": "MASE împarte MAE la MAE în eșantion a metodei naive sezoniere; valorile sub 1 depășesc acea metodă de referință, iar pe aceleași date de test modelul cu MASE mai mic este mai precis.",
                "incorrectExplanation": "MASE nu se măsoară relativ la DHR, valorile sub 1 înseamnă mai bine decît metoda de referință, iar pe o scală comună valorile MASE ale modelelor diferite sînt direct comparabile. DHR are MASE mai mic."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "A Diebold-Mariano p-value",
                "text": "Ridge and DHR have almost the same MAE on the load origins; the Diebold-Mariano test of equal accuracy gives p = 0.94. What do we conclude?",
                "options": [
                    "Ridge is significantly more accurate than DHR",
                    "DHR is significantly more accurate than ridge",
                    "The two models produce identical forecasts",
                    "The data give no evidence that one model is more accurate than the other"
                ],
                "correctExplanation": "A large p-value means that the mean loss difference is small compared with its sampling variability over the origins: equal accuracy is not rejected.",
                "incorrectExplanation": "A large p-value supports neither direction, and equal accuracy on average does not mean identical forecasts. The test simply finds no evidence of a difference."
            },
            "ro": {
                "title": "O valoare p a testului Diebold–Mariano",
                "text": "Ridge și DHR au aproape același MAE pe originile pentru consum; testul Diebold–Mariano de acuratețe egală dă p = 0,94. Ce concluzionăm?",
                "options": [
                    "Ridge este semnificativ mai precis decît DHR",
                    "DHR este semnificativ mai precis decît ridge",
                    "Cele două modele dau prognoze identice",
                    "Datele nu oferă dovezi că unul dintre modele este mai precis decît celălalt"
                ],
                "correctExplanation": "O valoare p mare înseamnă că diferența medie a pierderilor este mică față de variabilitatea ei de la o origine la alta: acuratețea egală nu este respinsă.",
                "incorrectExplanation": "O valoare p mare nu susține niciun sens, iar acuratețea egală în medie nu înseamnă prognoze identice. Testul nu găsește dovezi ale unei diferențe."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Global models",
                "text": "What is a global forecasting model?",
                "options": [
                    "A model that forecasts world aggregates, such as global GDP",
                    "A model with one separate set of parameters for every series",
                    "One model estimated on the pooled data of many related series and used to forecast each of them",
                    "A model that uses all available lags of a single series"
                ],
                "correctExplanation": "A global model shares its parameters across series, so a flexible learner such as gradient boosting gets many more rows. The winners of M4 and M5 were global; for Romanian inflation the global GB pooled 27 EU countries.",
                "incorrectExplanation": "The word refers to pooling across series, not to world data; one parameter set per series is a local model, and using many lags of one series is still local. A global model is fitted once on many series."
            },
            "ro": {
                "title": "Modele globale",
                "text": "Ce este un model global de prognoză?",
                "options": [
                    "Un model care prognozează agregate mondiale, de exemplu PIB-ul global",
                    "Un model cu cîte un set separat de parametri pentru fiecare serie",
                    "Un singur model estimat pe datele combinate ale mai multor serii înrudite și folosit pentru prognoza fiecăreia",
                    "Un model care folosește toate decalajele disponibile ale unei singure serii"
                ],
                "correctExplanation": "Un model global are parametri comuni pentru toate seriile, deci un algoritm flexibil precum gradient boosting primește mult mai multe rînduri. Cîștigătorii M4 și M5 au fost globali; pentru inflația României, GB global a combinat 27 de țări UE.",
                "incorrectExplanation": "Termenul se referă la combinarea seriilor, nu la date mondiale; un set de parametri pentru fiecare serie înseamnă un model local, iar folosirea multor decalaje ale unei serii este tot locală. Un model global se estimează o singură dată, pe multe serii."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The pinball loss",
                "text": "A 90% quantile forecast of tomorrow's load is $q = 6.4$ GW and the outcome is $y = 6.6$ GW. What is the pinball loss $L_{0.9}(y, q)$?",
                "options": [
                    "0.02",
                    "0.18",
                    "0.20",
                    "0.04"
                ],
                "correctExplanation": "Since $y \\ge q$: $L = \\tau(y - q) = 0.9 \\cdot 0.2 = 0.18$. An outcome above a high quantile is penalised nine times more per unit than one below it.",
                "incorrectExplanation": "0.02 uses the weight $1 - \\tau$ on the wrong side, 0.20 is the absolute error and 0.04 is the loss for $y = 6.0$. With $y$ above $q$ the weight is $\\tau = 0.9$."
            },
            "ro": {
                "title": "Funcția de pierdere pinball",
                "text": "O prognoză cuantilică de 90% pentru consumul de mîine este $q = 6{,}4$ GW, iar valoarea realizată este $y = 6{,}6$ GW. Cît este funcția pinball $L_{0{,}9}(y, q)$?",
                "options": [
                    "0,02",
                    "0,18",
                    "0,20",
                    "0,04"
                ],
                "correctExplanation": "Deoarece $y \\ge q$: $L = \\tau(y - q) = 0{,}9 \\cdot 0{,}2 = 0{,}18$. O valoare peste o cuantilă mare este penalizată de nouă ori mai mult pe unitate decît una sub ea.",
                "incorrectExplanation": "0,02 folosește ponderea $1 - \\tau$ pe partea greșită, 0,20 este eroarea absolută, iar 0,04 este pierderea pentru $y = 6{,}0$. Cu $y$ peste $q$, ponderea este $\\tau = 0{,}9$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Split conformal prediction",
                "text": "A split conformal interval at level 80% uses $n = 9$ absolute calibration errors. Which order statistic gives the half-width $\\hat q$?",
                "options": [
                    "The 8th smallest",
                    "The 7th smallest",
                    "The 9th smallest (the largest)",
                    "The median, the 5th smallest"
                ],
                "correctExplanation": "The rank is $\\lceil (n + 1)(1 - \\alpha) \\rceil = \\lceil 10 \\cdot 0.8 \\rceil = 8$; the interval is $\\hat y \\pm \\hat q$ and covers the next value with probability at least 80% if the errors are exchangeable.",
                "incorrectExplanation": "The 7th smallest ignores the $+1$ correction, the largest error gives a wider interval than needed, and the median gives about 50% coverage. The rank is $\\lceil (n + 1)(1 - \\alpha) \\rceil = 8$."
            },
            "ro": {
                "title": "Predicția conformală prin împărțire",
                "text": "Un interval conformal de 80% folosește $n = 9$ erori absolute de calibrare. Ce statistică de ordine dă semilățimea $\\hat q$?",
                "options": [
                    "A 8-a cea mai mică valoare",
                    "A 7-a cea mai mică valoare",
                    "A 9-a cea mai mică valoare (cea mai mare)",
                    "Mediana, a 5-a cea mai mică valoare"
                ],
                "correctExplanation": "Rangul este $\\lceil (n + 1)(1 - \\alpha) \\rceil = \\lceil 10 \\cdot 0{,}8 \\rceil = 8$; intervalul este $\\hat y \\pm \\hat q$ și acoperă următoarea valoare cu probabilitatea de cel puțin 80% dacă erorile sînt interschimbabile.",
                "incorrectExplanation": "A 7-a valoare ignoră corecția $+1$, cea mai mare eroare dă un interval mai larg decît este nevoie, iar mediana dă o acoperire de circa 50%. Rangul este $\\lceil (n + 1)(1 - \\alpha) \\rceil = 8$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The right baseline for the sign of returns",
                "text": "From 2008 to 2026 the S&P 500 rose on 54.4% of the test days. A logit model predicts the sign of the next daily return with 53.8% accuracy. What is the correct conclusion?",
                "options": [
                    "The model has skill, because its accuracy is above 50%",
                    "The model is profitable, because it is right on most days",
                    "Accuracy cannot be computed for the sign of returns",
                    "The model is worse than always predicting \"up\", so it has no useful skill"
                ],
                "correctExplanation": "The baseline is the majority class of the training data (\"always up\"), which is right on the share of up days. The model is below that baseline; its AUC is close to 0.5.",
                "incorrectExplanation": "Comparing with 50% ignores that markets rise more often than they fall, and an accuracy below the baseline cannot be a profitable signal. The right comparison is with \"always up\"."
            },
            "ro": {
                "title": "Reperul corect pentru semnul randamentelor",
                "text": "Între 2008 și 2026, S&P 500 a crescut în 54,4% din zilele de test. Un model logit prognozează semnul randamentului zilnic următor cu acuratețea de 53,8%. Care este concluzia corectă?",
                "options": [
                    "Modelul are capacitate predictivă, deoarece acuratețea lui depășește 50%",
                    "Modelul este profitabil, deoarece are dreptate în majoritatea zilelor",
                    "Acuratețea nu se poate calcula pentru semnul randamentelor",
                    "Modelul este mai slab decît „întotdeauna în creștere”, deci nu are nicio capacitate utilă"
                ],
                "correctExplanation": "Reperul este clasa majoritară din antrenare („întotdeauna în creștere”), care are dreptate în proporția zilelor de creștere. Modelul este sub acest reper; AUC-ul lui este aproape de 0,5.",
                "incorrectExplanation": "Comparația cu 50% ignoră faptul că piețele cresc mai des decît scad, iar o acuratețe sub reper nu poate fi un semnal profitabil. Comparația corectă este cu „întotdeauna în creștere”."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The M4 competition",
                "text": "What did the M4 competition (100,000 series, 2018) find about pure machine-learning methods?",
                "options": [
                    "They were the most accurate methods on all frequencies",
                    "They won because they were trained separately on every series",
                    "None of them beat the combination of exponential smoothing methods, and only one beat Naive2",
                    "They were not allowed in the competition"
                ],
                "correctExplanation": "The 6 pure ML methods ranked at best 23 (OWA 0.915) against Comb (0.898); the winner was a hybrid of exponential smoothing and an LSTM trained across all series (OWA 0.821).",
                "incorrectExplanation": "Pure ML methods were allowed but performed poorly; training separately on every short series is precisely what limited them. The winner was a hybrid global model, and 12 of the 17 best methods were combinations."
            },
            "ro": {
                "title": "Competiția M4",
                "text": "Ce a arătat competiția M4 (100\\,000 de serii, 2018) despre metodele pure de învățare automată?",
                "options": [
                    "Au fost cele mai precise metode la toate frecvențele",
                    "Au cîștigat pentru că au fost antrenate separat pe fiecare serie",
                    "Niciuna nu a depășit combinația metodelor de netezire exponențială, iar doar una a depășit Naive2",
                    "Nu au fost admise în competiție"
                ],
                "correctExplanation": "Cele 6 metode ML pure s-au clasat cel mai bine pe locul 23 (OWA 0,915), față de Comb (0,898); cîștigătorul a fost un model hibrid de netezire exponențială și LSTM, antrenat pe toate seriile (OWA 0,821).",
                "incorrectExplanation": "Metodele ML pure au fost admise, dar au avut rezultate slabe; antrenarea separată pe fiecare serie scurtă este exact ce le-a limitat. Cîștigătorul a fost un model hibrid global, iar 12 dintre cele mai bune 17 metode au fost combinații."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The M5 competition",
                "text": "Which approach dominated the M5 accuracy competition on Walmart unit sales?",
                "options": [
                    "One ARIMA model fitted separately to each of the 42,840 series",
                    "Gradient-boosted trees (LightGBM) trained on many pooled series, with calendar and price features",
                    "Deep LSTM networks trained on each product separately",
                    "The seasonal naive forecast"
                ],
                "correctExplanation": "Most top methods used LightGBM; the winner averaged LightGBM models pooled by store, category and department, both recursive and direct, and was 22.4% more accurate than the best benchmark.",
                "incorrectExplanation": "Local ARIMA and per-product networks lost to the pooled tree ensembles, and the seasonal naive method was only a benchmark. M5 was won by global LightGBM models."
            },
            "ro": {
                "title": "Competiția M5",
                "text": "Ce abordare a dominat competiția de acuratețe M5 pe vînzările Walmart?",
                "options": [
                    "Cîte un model ARIMA estimat separat pentru fiecare dintre cele 42\\,840 de serii",
                    "Arbori cu gradient boosting (LightGBM) antrenați pe multe serii combinate, cu variabile de calendar și de preț",
                    "Rețele LSTM adînci antrenate separat pentru fiecare produs",
                    "Prognoza naivă sezonieră"
                ],
                "correctExplanation": "Cele mai multe metode de top au folosit LightGBM; cîștigătorul a făcut media unor modele LightGBM combinate pe magazin, categorie și departament, recursive și directe, și a fost cu 22,4% mai precis decît cea mai bună metodă de referință.",
                "incorrectExplanation": "ARIMA local și rețelele pe fiecare produs au pierdut în fața ansamblurilor de arbori combinate, iar metoda naivă sezonieră a fost doar un reper. M5 a fost cîștigată de modele globale LightGBM."
            }
        }
    ]
};
