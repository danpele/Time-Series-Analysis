// ============================================================
// Chapter 9 quiz bank: Machine learning for time series (EN + RO)
// 14 questions ported from the 2025/2026 site; 14 drawn per attempt.
// The bank grows to 24 questions when the chapter is rebuilt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['ml'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Feature engineering for machine learning",
                "text": "To apply a random forest to time series forecasting, we must create:",
                "options": [
                    "A dummy variable for each observation",
                    "Lag features and rolling statistics",
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
                    "O variabilă dummy pentru fiecare observație",
                    "Variabile lag și statistici pe ferestre mobile",
                    "Doar transformate Fourier ale seriei",
                    "Doar prima diferență a seriei"
                ],
                "correctExplanation": "Un random forest nu cunoaște ordinea temporală, deci seria trebuie transformată într-un tabel de învățare supervizată: variabile lag ($y_{t-1}, y_{t-2}, \\ldots$), medii și abateri standard pe ferestre mobile, precum și variabile calendaristice (ziua săptămînii, luna).",
                "incorrectExplanation": "O variabilă dummy pentru fiecare observație ar face fiecare rînd unic, iar modelul nu ar putea generaliza; termenii Fourier sau prima diferență pot fi intrări suplimentare utile, dar singure nu descriu dinamica recentă. Intrările de bază sînt lag-urile și statisticile pe ferestre mobile."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Time series cross-validation",
                "text": "Why can't standard k-fold cross-validation be used for time series?",
                "options": [
                    "It is too slow for long series",
                    "It violates the temporal order and causes data leakage",
                    "It only works for classification",
                    "It requires too much data"
                ],
                "correctExplanation": "Standard k-fold assigns observations to folds at random, so the model is trained on observations that come after the validation fold. This leaks future information and overstates performance. Walk-forward validation (for example TimeSeriesSplit in scikit-learn) keeps the order.",
                "incorrectExplanation": "Speed, the type of task and sample size are not the issue: k-fold works for regression and for any sample size. The problem is that it breaks the temporal order; walk-forward validation trains only on the past."
            },
            "ro": {
                "title": "Validarea încrucișată pentru serii de timp",
                "text": "De ce nu poate fi folosită validarea încrucișată k-fold standard pentru serii de timp?",
                "options": [
                    "Este prea lentă pentru serii lungi",
                    "Încalcă ordinea temporală și produce scurgere de informație (data leakage)",
                    "Funcționează doar pentru clasificare",
                    "Necesită prea multe date"
                ],
                "correctExplanation": "K-fold standard repartizează aleator observațiile în subeșantioane, astfel încît modelul este antrenat pe observații ulterioare subeșantionului de validare. Se scurge astfel informație din viitor, iar performanța este supraestimată. Validarea walk-forward (de exemplu TimeSeriesSplit din scikit-learn) păstrează ordinea.",
                "incorrectExplanation": "Viteza, tipul problemei și volumul de date nu sînt cauza: k-fold funcționează și pentru regresie, și pentru orice volum de date. Problema este încălcarea ordinii temporale; validarea walk-forward antrenează modelul doar pe trecut."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Advantage of LSTM",
                "text": "What is the main advantage of an LSTM over a simple recurrent neural network (RNN)?",
                "options": [
                    "It is faster to train",
                    "It mitigates the vanishing gradient problem and can learn long-range dependencies",
                    "It requires less data",
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
                    "Atenuează problema gradienților care se anulează (vanishing gradient) și poate învăța dependențe pe termen lung",
                    "Necesită mai puține date",
                    "Este mai ușor de interpretat"
                ],
                "correctExplanation": "Starea celulei LSTM, controlată de porțile de uitare, de intrare și de ieșire, se actualizează aditiv, astfel încît gradienții se pot propaga pe mulți pași de timp fără să se anuleze. Rețeaua poate învăța astfel dependențe pe termen lung. Gradienții care explodează se tratează de obicei separat, prin gradient clipping.",
                "incorrectExplanation": "O rețea LSTM are mai mulți parametri decît o RNN simplă, deci se antrenează mai lent, are nevoie de regulă de mai multe date și nu este mai ușor de interpretat. Avantajul ei este starea celulei controlată de porți, care atenuează anularea gradienților."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Scaling the data",
                "text": "Before training an LSTM, the data should be:",
                "options": [
                    "Log-transformed",
                    "Scaled, for example to $[0,1]$ or $[-1,1]$, or standardised",
                    "Differenced twice",
                    "Converted to integers"
                ],
                "correctExplanation": "The sigmoid and tanh activations work on bounded ranges, so scaled inputs give faster convergence and numerical stability. The scaler must be fitted on the training data only and then applied to the validation and test data.",
                "incorrectExplanation": "A log transform or differencing may be useful in specific cases but is not the generic requirement, and integer conversion only loses information. The standard step is scaling (MinMaxScaler or StandardScaler) fitted on the training set."
            },
            "ro": {
                "title": "Scalarea datelor",
                "text": "Înainte de antrenarea unei rețele LSTM, datele trebuie:",
                "options": [
                    "Transformate logaritmic",
                    "Scalate, de exemplu la $[0,1]$ sau $[-1,1]$, ori standardizate",
                    "Diferențiate de două ori",
                    "Convertite în numere întregi"
                ],
                "correctExplanation": "Funcțiile de activare sigmoid și tanh lucrează pe intervale mărginite, deci intrările scalate asigură o convergență mai rapidă și stabilitate numerică. Scalarea se estimează doar pe datele de antrenare și apoi se aplică datelor de validare și de test.",
                "incorrectExplanation": "Transformarea logaritmică sau diferențierea pot fi utile în anumite situații, dar nu sînt cerința generală, iar conversia în numere întregi doar pierde informație. Pasul standard este scalarea (MinMaxScaler sau StandardScaler) estimată pe setul de antrenare."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "LSTM hyperparameters",
                "text": "Which of the following is NOT a typical LSTM hyperparameter?",
                "options": [
                    "Number of units (neurons) per layer",
                    "Input sequence length",
                    "Learning rate",
                    "Fractional differencing parameter $d$"
                ],
                "correctExplanation": "The fractional differencing parameter $d$ belongs to ARFIMA, not to neural networks. Typical LSTM hyperparameters are the architecture (layers, units), sequence length, learning rate, batch size and dropout rate.",
                "incorrectExplanation": "Number of units, sequence length and learning rate are all chosen by the user before training, so they are hyperparameters. The parameter $d$ is estimated in ARFIMA models and plays no role in an LSTM."
            },
            "ro": {
                "title": "Hiperparametri LSTM",
                "text": "Care dintre următoarele NU este un hiperparametru tipic al unei rețele LSTM?",
                "options": [
                    "Numărul de unități (neuroni) pe strat",
                    "Lungimea secvenței de intrare",
                    "Rata de învățare (learning rate)",
                    "Parametrul de diferențiere fracționară $d$"
                ],
                "correctExplanation": "Parametrul de diferențiere fracționară $d$ aparține modelului ARFIMA, nu rețelelor neuronale. Hiperparametrii tipici ai unei rețele LSTM sînt arhitectura (straturi, unități), lungimea secvenței, rata de învățare, dimensiunea lotului (batch size) și rata de dropout.",
                "incorrectExplanation": "Numărul de unități, lungimea secvenței și rata de învățare sînt alese înainte de antrenare, deci sînt hiperparametri. Parametrul $d$ se estimează în modelele ARFIMA și nu are niciun rol într-o rețea LSTM."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Feature importance",
                "text": "In a random forest for time series, feature importance helps us to:",
                "options": [
                    "Eliminate all low-importance variables automatically",
                    "Identify which lags and features are most predictive",
                    "Establish Granger causality",
                    "Compute confidence intervals"
                ],
                "correctExplanation": "Feature importance ranks lags and engineered features by their contribution to predictive accuracy. It measures predictive power, not causality, and correlated features can share or split importance.",
                "incorrectExplanation": "Importance scores do not remove variables by themselves, do not test Granger causality and do not deliver confidence intervals. They show which inputs carry predictive information."
            },
            "ro": {
                "title": "Importanța variabilelor",
                "text": "Într-un random forest pentru serii de timp, importanța variabilelor (feature importance) ne ajută să:",
                "options": [
                    "Eliminăm automat toate variabilele cu importanță scăzută",
                    "Identificăm lag-urile și variabilele cu cea mai mare putere predictivă",
                    "Stabilim cauzalitatea Granger",
                    "Calculăm intervale de încredere"
                ],
                "correctExplanation": "Importanța variabilelor ordonează lag-urile și variabilele construite după contribuția lor la acuratețea prognozei. Ea măsoară puterea predictivă, nu cauzalitatea, iar variabilele corelate își pot împărți importanța.",
                "incorrectExplanation": "Scorurile de importanță nu elimină singure variabile, nu testează cauzalitatea Granger și nu furnizează intervale de încredere. Ele arată care intrări conțin informație predictivă."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Model selection",
                "text": "When comparing ARFIMA, random forest and LSTM for forecasting:",
                "options": [
                    "LSTM always wins because it is deep learning",
                    "ARFIMA is always best for financial data",
                    "The best model depends on the data and on the requirements of the application",
                    "A random forest cannot be used for time series"
                ],
                "correctExplanation": "ARFIMA captures long memory and is interpretable; a random forest captures non-linear effects and gives feature importance; an LSTM can learn complex patterns from long sequences but needs more data. The choice must be validated with time series cross-validation.",
                "incorrectExplanation": "No model dominates in all settings: deep learning often loses to simple models on short or noisy series, ARFIMA is not universally best for financial data, and a random forest works well once lag features are built. Choose on the basis of the data, interpretability needs and out-of-sample performance."
            },
            "ro": {
                "title": "Alegerea modelului",
                "text": "La compararea modelelor ARFIMA, random forest și LSTM pentru prognoză:",
                "options": [
                    "LSTM cîștigă întotdeauna, pentru că este deep learning",
                    "ARFIMA este întotdeauna cel mai bun pentru date financiare",
                    "Cel mai bun model depinde de date și de cerințele aplicației",
                    "Un random forest nu poate fi folosit pentru serii de timp"
                ],
                "correctExplanation": "ARFIMA captează memoria lungă și este interpretabil; un random forest captează efecte neliniare și oferă importanța variabilelor; o rețea LSTM poate învăța tipare complexe din secvențe lungi, dar are nevoie de mai multe date. Alegerea trebuie validată prin validare încrucișată pentru serii de timp.",
                "incorrectExplanation": "Niciun model nu domină în toate situațiile: deep learning pierde adesea în fața modelelor simple pe serii scurte sau zgomotoase, ARFIMA nu este universal cel mai bun pentru date financiare, iar un random forest funcționează bine odată ce sînt construite variabilele lag. Alegerea se face pe baza datelor, a nevoii de interpretabilitate și a performanței în afara eșantionului."
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
                "text": "Comparativ cu un singur arbore de decizie, un random forest reduce supraajustarea prin:",
                "options": [
                    "Folosirea unor arbori mai adînci",
                    "Medierea predicțiilor mai multor arbori antrenați pe eșantioane bootstrap",
                    "Folosirea unui număr mai mic de observații",
                    "Antrenarea exclusiv pe întregul set de date"
                ],
                "correctExplanation": "Bagging-ul (bootstrap aggregating), împreună cu alegerea aleatoare a unui subset de variabile la fiecare ramificare, face arborii diferiți între ei; medierea multor arbori slab corelați reduce varianța, lăsînd distorsiunea (bias) aproximativ neschimbată.",
                "incorrectExplanation": "Arborii mai adînci accentuează supraajustarea, renunțarea la observații pierde informație, iar o singură estimare pe întregul set de date este exact ce face un arbore izolat. Cîștigul vine din medierea multor arbori crescuți pe eșantioane bootstrap, cu subseturi aleatoare de variabile."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Data leakage",
                "text": "Which of the following is an example of data leakage in machine learning for time series?",
                "options": [
                    "Using lag features",
                    "Scaling the data with statistics computed on the whole sample (training and test)",
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
                    "Folosirea variabilelor lag",
                    "Scalarea datelor cu statistici calculate pe întregul eșantion (antrenare și test)",
                    "Folosirea variabilelor pe ferestre mobile",
                    "Împărțirea cronologică a datelor"
                ],
                "correctExplanation": "Dacă media, abaterea standard, minimul sau maximul folosite la scalare se calculează pe întregul eșantion, datele de antrenare conțin informație despre perioada de test. Scalarea trebuie estimată doar pe datele de antrenare și apoi aplicată ambelor seturi.",
                "incorrectExplanation": "Variabilele lag și cele pe ferestre mobile folosesc doar valori trecute, deci sînt legitime, iar împărțirea cronologică este practica corectă. Scurgerea apare cînd statistici din perioada de test intră în etapa de antrenare, ca la scalarea pe întregul eșantion."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "LSTM gates",
                "text": "Which LSTM gate decides what information to discard from the cell state?",
                "options": [
                    "The input gate",
                    "The forget gate",
                    "The output gate",
                    "The update gate"
                ],
                "correctExplanation": "The forget gate $f_t = \\sigma(W_f[h_{t-1}, x_t] + b_f)$ multiplies the previous cell state: values close to 0 mean forget, values close to 1 mean keep.",
                "incorrectExplanation": "The input gate controls what new information enters the cell state, the output gate controls what is exposed as the hidden state, and the update gate belongs to the GRU, not the LSTM. Discarding information is the role of the forget gate."
            },
            "ro": {
                "title": "Porțile LSTM",
                "text": "Ce poartă a unei rețele LSTM decide ce informație se elimină din starea celulei?",
                "options": [
                    "Poarta de intrare (input gate)",
                    "Poarta de uitare (forget gate)",
                    "Poarta de ieșire (output gate)",
                    "Poarta de actualizare (update gate)"
                ],
                "correctExplanation": "Poarta de uitare $f_t = \\sigma(W_f[h_{t-1}, x_t] + b_f)$ înmulțește starea anterioară a celulei: valorile apropiate de 0 înseamnă uitare, iar cele apropiate de 1 înseamnă păstrare.",
                "incorrectExplanation": "Poarta de intrare controlează ce informație nouă intră în starea celulei, poarta de ieșire controlează ce se transmite în starea ascunsă, iar poarta de actualizare aparține rețelei GRU, nu LSTM. Eliminarea informației este rolul porții de uitare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Dropout regularisation",
                "text": "In an LSTM, dropout helps to:",
                "options": [
                    "Speed up training",
                    "Prevent overfitting by randomly setting units to zero during training",
                    "Increase model capacity",
                    "Handle missing data"
                ],
                "correctExplanation": "During training, dropout sets a random fraction of units to zero at each step, which prevents units from co-adapting and forces more robust representations. This regularises the network and reduces overfitting.",
                "incorrectExplanation": "Dropout usually slows convergence slightly, it reduces rather than increases effective capacity, and it has nothing to do with missing observations. It is a regularisation technique against overfitting."
            },
            "ro": {
                "title": "Regularizarea prin dropout",
                "text": "Într-o rețea LSTM, dropout-ul ajută la:",
                "options": [
                    "Accelerarea antrenării",
                    "Prevenirea supraajustării prin anularea aleatoare a unor unități în timpul antrenării",
                    "Creșterea capacității modelului",
                    "Tratarea datelor lipsă"
                ],
                "correctExplanation": "În timpul antrenării, dropout-ul anulează la fiecare pas o fracțiune aleatoare de unități, ceea ce împiedică adaptarea lor reciprocă și forțează reprezentări mai robuste. Rețeaua este astfel regularizată, iar supraajustarea scade.",
                "incorrectExplanation": "Dropout-ul încetinește de obicei ușor convergența, reduce capacitatea efectivă în loc să o mărească și nu are legătură cu observațiile lipsă. Este o tehnică de regularizare împotriva supraajustării."
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
                "title": "Early stopping",
                "text": "Early stopping urmărește pierderea pe setul de validare pentru a:",
                "options": [
                    "Accelera fiecare iterație",
                    "Opri antrenarea atunci cînd modelul începe să supraajusteze",
                    "Selecta cele mai bune variabile",
                    "Ajusta rata de învățare"
                ],
                "correctExplanation": "Cînd pierderea pe setul de validare nu se mai îmbunătățește timp de un număr dat de epoci (patience), antrenarea se oprește și se păstrează cele mai bune ponderi. Astfel rețeaua nu ajunge să modeleze zgomotul din datele de antrenare.",
                "incorrectExplanation": "Early stopping nu accelerează iterațiile, nu selectează variabile și nu modifică rata de învățare (acesta este rolul unui learning-rate scheduler). El oprește antrenarea cînd performanța pe setul de validare nu se mai îmbunătățește."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Input sequence length",
                "text": "Choosing the LSTM input sequence length (lookback window) involves:",
                "options": [
                    "Always using all available history",
                    "Balancing the amount of history the model sees against computational cost and overfitting risk",
                    "Always using exactly 30 time steps",
                    "Matching the seasonal period, and nothing else"
                ],
                "correctExplanation": "Longer sequences give the network more history but increase computation and the number of noisy inputs, and they do not help if the relevant dependencies are short. The lookback is a hyperparameter tuned with domain knowledge and validation.",
                "incorrectExplanation": "There is no universal value such as 30 steps, using all history is rarely optimal, and the seasonal period is only one useful reference point. The lookback is tuned to the temporal dependence of the data."
            },
            "ro": {
                "title": "Lungimea secvenței de intrare",
                "text": "Alegerea lungimii secvenței de intrare a unei rețele LSTM (fereastra lookback) presupune:",
                "options": [
                    "Folosirea întotdeauna a întregului istoric disponibil",
                    "Un echilibru între cît istoric vede modelul, costul de calcul și riscul de supraajustare",
                    "Folosirea întotdeauna a exact 30 de pași de timp",
                    "Potrivirea cu perioada sezonieră și nimic altceva"
                ],
                "correctExplanation": "Secvențele mai lungi oferă rețelei mai mult istoric, dar cresc costul de calcul și numărul de intrări zgomotoase și nu ajută dacă dependențele relevante sînt scurte. Fereastra lookback este un hiperparametru ales pe baza cunoașterii domeniului și a validării.",
                "incorrectExplanation": "Nu există o valoare universală, de tipul 30 de pași, folosirea întregului istoric este rareori optimă, iar perioada sezonieră este doar un reper util. Fereastra se alege în funcție de dependența temporală din date."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Directional accuracy",
                "text": "Directional accuracy measures:",
                "options": [
                    "The magnitude of forecast errors",
                    "How often the model correctly predicts up and down movements",
                    "The correlation between forecasts and actual values",
                    "The variance of forecast errors"
                ],
                "correctExplanation": "Directional accuracy is the share of periods in which $\\text{sign}(\\Delta \\hat{y}_t) = \\text{sign}(\\Delta y_t)$. It matters for trading decisions even when RMSE is high.",
                "incorrectExplanation": "Magnitude and variance of errors are captured by MAE, RMSE and similar metrics, and correlation measures linear co-movement rather than agreement in sign. Directional accuracy counts how often the predicted direction is right."
            },
            "ro": {
                "title": "Acuratețea direcțională",
                "text": "Acuratețea direcțională măsoară:",
                "options": [
                    "Mărimea erorilor de prognoză",
                    "Cît de des prognozează corect modelul mișcările în sus și în jos",
                    "Corelația dintre prognoze și valorile efective",
                    "Varianța erorilor de prognoză"
                ],
                "correctExplanation": "Acuratețea direcțională este proporția perioadelor în care $\\text{sign}(\\Delta \\hat{y}_t) = \\text{sign}(\\Delta y_t)$. Ea contează pentru deciziile de tranzacționare chiar și atunci cînd RMSE este mare.",
                "incorrectExplanation": "Mărimea și varianța erorilor sînt descrise de MAE, RMSE și indicatori similari, iar corelația măsoară co-mișcarea liniară, nu potrivirea semnului. Acuratețea direcțională numără cît de des direcția prognozată este cea corectă."
            }
        }
    ]
};
