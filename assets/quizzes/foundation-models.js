// ============================================================
// Chapter 11 quiz bank: Foundation models for time series (EN + RO)
// 10 questions ported from the 2025/2026 site; 10 drawn per attempt.
// The bank grows to 24 questions when the chapter is rebuilt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['foundation-models'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Self-attention mechanism",
                "text": "What is the formula for scaled dot-product attention in the Transformer (Vaswani et al., 2017)?",
                "options": [
                    "$\\text{Attention}(Q,K,V) = \\text{softmax}(QK^\\top) V$",
                    "$\\text{Attention}(Q,K,V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V$",
                    "$\\text{Attention}(Q,K,V) = \\sigma(QK^\\top + b) V$",
                    "$\\text{Attention}(Q,K,V) = Q \\, \\text{softmax}(K^\\top V)$"
                ],
                "correctExplanation": "Each query is compared with every key through a dot product, the scores are divided by $\\sqrt{d_k}$, turned into weights by the softmax, and used to average the values. Dividing by $\\sqrt{d_k}$ keeps the scores from growing with the key dimension, so the softmax does not saturate and gradients stay usable.",
                "incorrectExplanation": "Without the $\\sqrt{d_k}$ factor the dot products grow with the dimension and the softmax saturates; a sigmoid does not produce weights that sum to one; and the softmax applies to query-key scores, not to $K^\\top V$. The correct form is $\\text{softmax}(QK^\\top/\\sqrt{d_k})V$."
            },
            "ro": {
                "title": "Mecanismul de self-attention",
                "text": "Care este formula atenției de tip scaled dot-product din Transformer (Vaswani et al., 2017)?",
                "options": [
                    "$\\text{Attention}(Q,K,V) = \\text{softmax}(QK^\\top) V$",
                    "$\\text{Attention}(Q,K,V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V$",
                    "$\\text{Attention}(Q,K,V) = \\sigma(QK^\\top + b) V$",
                    "$\\text{Attention}(Q,K,V) = Q \\, \\text{softmax}(K^\\top V)$"
                ],
                "correctExplanation": "Fiecare interogare (query) este comparată cu fiecare cheie (key) printr-un produs scalar, scorurile sînt împărțite la $\\sqrt{d_k}$, transformate în ponderi prin softmax și folosite pentru medierea valorilor (values). Împărțirea la $\\sqrt{d_k}$ împiedică creșterea scorurilor odată cu dimensiunea cheilor, astfel încît softmax nu se saturează, iar gradienții rămîn utilizabili.",
                "incorrectExplanation": "Fără factorul $\\sqrt{d_k}$, produsele scalare cresc odată cu dimensiunea, iar softmax se saturează; funcția sigmoid nu produce ponderi cu suma 1; iar softmax se aplică scorurilor dintre interogări și chei, nu lui $K^\\top V$. Forma corectă este $\\text{softmax}(QK^\\top/\\sqrt{d_k})V$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Architecture of Chronos",
                "text": "Which type of Transformer architecture does Chronos (Amazon) use?",
                "options": [
                    "Decoder-only (GPT-style)",
                    "Encoder-decoder (T5)",
                    "Encoder-only (BERT-style)",
                    "A recurrent network (LSTM)"
                ],
                "correctExplanation": "Chronos reuses the T5 encoder-decoder architecture essentially unchanged; the novelty is that time series values are scaled and quantised into tokens, so forecasting becomes language modelling over this vocabulary.",
                "incorrectExplanation": "Decoder-only designs are used by other models (for example TimesFM), encoder-only masked designs by Moirai, and Chronos is not recurrent. Chronos is built on the T5 encoder-decoder."
            },
            "ro": {
                "title": "Arhitectura Chronos",
                "text": "Ce tip de arhitectură Transformer folosește Chronos (Amazon)?",
                "options": [
                    "Decoder-only (stil GPT)",
                    "Encoder-decoder (T5)",
                    "Encoder-only (stil BERT)",
                    "O rețea recurentă (LSTM)"
                ],
                "correctExplanation": "Chronos preia practic nemodificată arhitectura encoder-decoder T5; noutatea constă în faptul că valorile seriei sînt scalate și cuantizate în tokenuri, astfel încît prognoza devine modelare a limbajului pe acest vocabular.",
                "incorrectExplanation": "Arhitecturile decoder-only sînt folosite de alte modele (de exemplu TimesFM), cele encoder-only cu mascare de Moirai, iar Chronos nu este o rețea recurentă. Chronos folosește arhitectura encoder-decoder T5."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Tokenisation by patching",
                "text": "A time series has 512 time steps and we use patch size $P = 32$ with non-overlapping patches. How many tokens result?",
                "options": [
                    "512 tokens",
                    "16 tokens",
                    "32 tokens",
                    "256 tokens"
                ],
                "correctExplanation": "Patching groups $P$ consecutive points into one token: $512 / 32 = 16$ tokens. Attention cost falls from $O(T^2)$ to $O((T/P)^2)$.",
                "incorrectExplanation": "512 tokens would mean one token per time step (no patching), 32 is the patch size itself, and 256 would correspond to patches of 2 points. With $P = 32$ the series becomes $512/32 = 16$ tokens."
            },
            "ro": {
                "title": "Tokenizare prin patching",
                "text": "O serie de timp are 512 pași de timp și folosim patch-uri disjuncte de dimensiune $P = 32$. Cîte tokenuri rezultă?",
                "options": [
                    "512 tokenuri",
                    "16 tokenuri",
                    "32 de tokenuri",
                    "256 de tokenuri"
                ],
                "correctExplanation": "Patching-ul grupează $P$ puncte consecutive într-un singur token: $512 / 32 = 16$ tokenuri. Costul atenției scade de la $O(T^2)$ la $O((T/P)^2)$.",
                "incorrectExplanation": "512 tokenuri ar însemna cîte un token pentru fiecare pas de timp (fără patching), 32 este chiar dimensiunea patch-ului, iar 256 ar corespunde unor patch-uri de 2 puncte. Cu $P = 32$, seria devine $512/32 = 16$ tokenuri."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Zero-shot forecasting and fine-tuning",
                "text": "When is zero-shot forecasting preferable to fine-tuning?",
                "options": [
                    "When millions of observations from the target domain are available",
                    "When target data are scarce, rapid prototyping is needed or many heterogeneous series must be forecast",
                    "When causal interpretability is required",
                    "When the target series has strong domain-specific patterns absent from the pre-training corpus"
                ],
                "correctExplanation": "Zero-shot use suits cold-start problems, quick prototypes and deployment across many series without training a model for each. In the Chronos benchmarks, zero-shot forecasts were competitive with, and on many data sets better than, models trained on the target data.",
                "incorrectExplanation": "Abundant target data and patterns missing from the pre-training corpus are exactly the cases where fine-tuning pays off, and neither approach provides causal interpretability. Zero-shot is attractive when data are scarce or speed and breadth matter."
            },
            "ro": {
                "title": "Prognoza zero-shot și fine-tuning",
                "text": "Cînd este preferabilă prognoza zero-shot față de fine-tuning?",
                "options": [
                    "Cînd sînt disponibile milioane de observații din domeniul țintă",
                    "Cînd datele țintă sînt puține, este nevoie de un prototip rapid sau trebuie prognozate multe serii eterogene",
                    "Cînd este necesară interpretabilitatea cauzală",
                    "Cînd seria țintă are tipare specifice domeniului, absente din corpusul de pre-antrenare"
                ],
                "correctExplanation": "Utilizarea zero-shot se potrivește problemelor de tip cold-start, prototipurilor rapide și aplicării pe multe serii fără antrenarea unui model pentru fiecare. În evaluările Chronos, prognozele zero-shot au fost competitive cu modelele antrenate pe datele țintă și, pe multe seturi de date, mai bune decît acestea.",
                "incorrectExplanation": "Datele țintă abundente și tiparele absente din corpusul de pre-antrenare sînt tocmai situațiile în care fine-tuning-ul merită, iar niciuna dintre abordări nu oferă interpretabilitate cauzală. Zero-shot este atractiv cînd datele sînt puține sau cînd contează viteza și acoperirea."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Tokenisation in Chronos",
                "text": "How does Chronos turn continuous time series values into discrete tokens?",
                "options": [
                    "Direct embedding of the raw values",
                    "Mean scaling of the context, followed by uniform binning into a fixed vocabulary of 4096 tokens",
                    "Differencing the series and one-hot encoding",
                    "Writing the series as text and using the standard GPT tokeniser"
                ],
                "correctExplanation": "Chronos divides the context by its mean absolute value, $\\tilde{x}_t = x_t / s$ with $s = \\frac{1}{C}\\sum_{t=1}^{C}|x_t|$, maps the scaled values into uniform bins (4094 bins plus two special tokens, 4096 in total) and models the token sequence with T5 and a cross-entropy loss.",
                "incorrectExplanation": "Chronos does not embed raw values, does not difference the series and does not use a text tokeniser. It mean-scales the context and quantises the values into a fixed set of bins, so forecasting becomes classification over tokens."
            },
            "ro": {
                "title": "Tokenizarea în Chronos",
                "text": "Cum transformă Chronos valorile continue ale unei serii de timp în tokenuri discrete?",
                "options": [
                    "Prin embedding direct al valorilor brute",
                    "Prin scalarea contextului cu media valorilor absolute, urmată de împărțirea în intervale egale ale unui vocabular fix de 4096 de tokenuri",
                    "Prin diferențierea seriei și codificare one-hot",
                    "Prin scrierea seriei ca text și folosirea tokenizatorului GPT standard"
                ],
                "correctExplanation": "Chronos împarte contextul la media valorilor absolute, $\\tilde{x}_t = x_t / s$, cu $s = \\frac{1}{C}\\sum_{t=1}^{C}|x_t|$, repartizează valorile scalate în intervale egale (4094 de intervale plus două tokenuri speciale, în total 4096) și modelează șirul de tokenuri cu T5 și o funcție de pierdere cross-entropy.",
                "incorrectExplanation": "Chronos nu folosește embedding-ul valorilor brute, nu diferențiază seria și nu folosește un tokenizator de text. El scalează contextul și cuantizează valorile într-un set fix de intervale, astfel încît prognoza devine o clasificare pe tokenuri."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "LoRA (low-rank adaptation)",
                "text": "Roughly what share of the parameters is trained in LoRA fine-tuning?",
                "options": [
                    "100%: all parameters are updated",
                    "About 0.1–1%: only small low-rank matrices ($W' = W + BA$, $r \\ll d$)",
                    "50%: half of the network layers",
                    "10–20%: only the last layers"
                ],
                "correctExplanation": "LoRA (Hu et al., 2022) freezes the pre-trained weights and learns a low-rank update $W' = W + BA$, with $B \\in \\mathbb{R}^{d \\times r}$, $A \\in \\mathbb{R}^{r \\times k}$ and $r \\ll \\min(d, k)$. Typically well under 1% of the parameters are trained, with performance close to full fine-tuning.",
                "incorrectExplanation": "Updating all parameters is full fine-tuning, and retraining whole layers (half the network or the last layers) is layer-wise transfer learning. LoRA trains only the small low-rank factors, a tiny fraction of the parameters."
            },
            "ro": {
                "title": "LoRA (low-rank adaptation)",
                "text": "Aproximativ ce proporție din parametri se antrenează într-un fine-tuning cu LoRA?",
                "options": [
                    "100%: toți parametrii sînt actualizați",
                    "Circa 0,1–1%: doar matrice mici de rang redus ($W' = W + BA$, $r \\ll d$)",
                    "50%: jumătate din straturile rețelei",
                    "10–20%: doar ultimele straturi"
                ],
                "correctExplanation": "LoRA (Hu et al., 2022) îngheață ponderile pre-antrenate și învață o actualizare de rang redus $W' = W + BA$, cu $B \\in \\mathbb{R}^{d \\times r}$, $A \\in \\mathbb{R}^{r \\times k}$ și $r \\ll \\min(d, k)$. De regulă se antrenează mult sub 1% din parametri, cu performanță apropiată de fine-tuning-ul complet.",
                "incorrectExplanation": "Actualizarea tuturor parametrilor înseamnă fine-tuning complet, iar reantrenarea unor straturi întregi (jumătate din rețea sau ultimele straturi) este transfer learning pe straturi. LoRA antrenează doar factorii mici de rang redus, o fracțiune infimă din parametri."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Universal design of Moirai",
                "text": "What distinguishes Moirai (Salesforce) among time series foundation models?",
                "options": [
                    "It is the only model with API access",
                    "It handles any number of variables, any frequency and any prediction length",
                    "It has the most parameters of any such model",
                    "It is pre-trained exclusively on synthetic data"
                ],
                "correctExplanation": "Moirai is a masked encoder-only Transformer with multi-patch-size projections, an any-variate attention mechanism and a mixture output distribution, which makes it the closest to a universal forecaster. It was pre-trained on LOTSA, about 27 billion observations from nine domains.",
                "incorrectExplanation": "API access is not a modelling property, Moirai is not the largest model (its largest version has about 311M parameters, fewer than Chronos-Large with 710M), and LOTSA consists of real data sets. Its distinctive feature is flexibility across variables, frequencies and horizons."
            },
            "ro": {
                "title": "Arhitectura universală Moirai",
                "text": "Ce distinge Moirai (Salesforce) între modelele fundaționale pentru serii de timp?",
                "options": [
                    "Este singurul model accesibil printr-un API",
                    "Tratează orice număr de variabile, orice frecvență și orice orizont de prognoză",
                    "Are cei mai mulți parametri dintre toate modelele de acest tip",
                    "Este pre-antrenat exclusiv pe date sintetice"
                ],
                "correctExplanation": "Moirai este un Transformer encoder-only, antrenat prin mascare, cu proiecții pentru mai multe dimensiuni de patch, un mecanism de atenție pentru orice număr de variabile și o distribuție de ieșire de tip mixtură, ceea ce îl apropie cel mai mult de un model universal de prognoză. A fost pre-antrenat pe LOTSA, circa 27 de miliarde de observații din nouă domenii.",
                "incorrectExplanation": "Accesul printr-un API nu este o proprietate a modelului, Moirai nu este cel mai mare model (varianta cea mai mare are circa 311M parametri, mai puțin decît Chronos-Large, cu 710M), iar LOTSA conține date reale. Trăsătura distinctivă este flexibilitatea în privința variabilelor, frecvențelor și orizonturilor."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Limitations of foundation models",
                "text": "Which of the following is NOT a major limitation of current time series foundation models?",
                "options": [
                    "Many are univariate only",
                    "They cannot produce probabilistic forecasts",
                    "They do not reason causally",
                    "They are black boxes that are hard to interpret"
                ],
                "correctExplanation": "Probabilistic forecasting is a strength: Chronos samples from a categorical distribution over tokens, Lag-Llama outputs Student-t parameters and Moirai outputs a mixture distribution. The real limitations are the frequent univariate restriction, the absence of causal reasoning and the lack of interpretability.",
                "incorrectExplanation": "Univariate restrictions, no causal reasoning and black-box behaviour are genuine limitations of current models. The statement that they cannot produce probabilistic forecasts is false."
            },
            "ro": {
                "title": "Limitele modelelor fundaționale",
                "text": "Care dintre următoarele NU este o limită importantă a modelelor fundaționale actuale pentru serii de timp?",
                "options": [
                    "Multe sînt doar univariate",
                    "Nu pot produce prognoze probabilistice",
                    "Nu fac raționamente cauzale",
                    "Sînt cutii negre, greu de interpretat"
                ],
                "correctExplanation": "Prognoza probabilistică este un punct forte: Chronos extrage eșantioane dintr-o distribuție categorială pe tokenuri, Lag-Llama produce parametrii unei distribuții Student-t, iar Moirai o distribuție de tip mixtură. Limitele reale sînt restricția frecventă la cazul univariat, absența raționamentului cauzal și lipsa de interpretabilitate.",
                "incorrectExplanation": "Restricția la cazul univariat, absența raționamentului cauzal și comportamentul de cutie neagră sînt limite reale ale modelelor actuale. Falsă este afirmația că nu pot produce prognoze probabilistice."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Output distribution of Lag-Llama",
                "text": "Which output distribution does Lag-Llama use instead of probabilities over discrete tokens?",
                "options": [
                    "The Normal distribution $N(\\mu, \\sigma^2)$",
                    "The Student-t distribution with parameters $(\\mu, \\sigma, \\nu)$",
                    "A uniform distribution on an interval",
                    "A categorical distribution over 4096 bins"
                ],
                "correctExplanation": "Lag-Llama outputs the parameters $(\\mu, \\sigma, \\nu)$ of a Student-t distribution for the next value. The heavy tails make it suitable for data with frequent extremes, such as financial series.",
                "incorrectExplanation": "A Normal output would understate tail risk, a uniform distribution has no realistic shape, and the categorical distribution over 4096 bins is the Chronos approach. Lag-Llama uses a parametric Student-t head."
            },
            "ro": {
                "title": "Distribuția de ieșire a Lag-Llama",
                "text": "Ce distribuție de ieșire folosește Lag-Llama în locul probabilităților pe tokenuri discrete?",
                "options": [
                    "Distribuția Normală $N(\\mu, \\sigma^2)$",
                    "Distribuția Student-t cu parametrii $(\\mu, \\sigma, \\nu)$",
                    "O distribuție uniformă pe un interval",
                    "O distribuție categorială pe 4096 de intervale"
                ],
                "correctExplanation": "Lag-Llama produce parametrii $(\\mu, \\sigma, \\nu)$ ai unei distribuții Student-t pentru valoarea următoare. Cozile groase o fac potrivită pentru date cu valori extreme frecvente, cum sînt seriile financiare.",
                "incorrectExplanation": "O ieșire Normală ar subestima riscul din cozi, o distribuție uniformă nu are o formă realistă, iar distribuția categorială pe 4096 de intervale este abordarea Chronos. Lag-Llama folosește o ieșire parametrică Student-t."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Scaling hypothesis",
                "text": "What has been found about scaling laws for time series foundation models compared with NLP?",
                "options": [
                    "Larger models are always much better",
                    "The evidence is mixed: data diversity and quality matter at least as much as parameter count",
                    "Scaling works exactly as in NLP",
                    "Smaller models are always better"
                ],
                "correctExplanation": "Gains from size are smaller than in NLP: for example, Chronos-Large (710M parameters) improves only modestly on Chronos-Base (200M). Time series are less diverse than text and many patterns are captured by autoregressive and seasonal structure, so the breadth and quality of the training data matter a great deal.",
                "incorrectExplanation": "Neither extreme holds: bigger models are not always much better, nor are smaller ones always superior, and the clean power laws of NLP have not been replicated. The current evidence is mixed, with data playing a central role."
            },
            "ro": {
                "title": "Ipoteza de scalare",
                "text": "Ce s-a constatat despre legile de scalare (scaling laws) ale modelelor fundaționale pentru serii de timp, comparativ cu NLP?",
                "options": [
                    "Modelele mai mari sînt întotdeauna mult mai bune",
                    "Dovezile sînt mixte: diversitatea și calitatea datelor contează cel puțin la fel de mult ca numărul de parametri",
                    "Scalarea funcționează exact ca în NLP",
                    "Modelele mai mici sînt întotdeauna mai bune"
                ],
                "correctExplanation": "Cîștigurile obținute prin mărirea modelului sînt mai mici decît în NLP: de exemplu, Chronos-Large (710M parametri) aduce doar o îmbunătățire modestă față de Chronos-Base (200M). Seriile de timp sînt mai puțin diverse decît textul, iar multe tipare sînt captate de structura autoregresivă și sezonieră, așa că amploarea și calitatea datelor de antrenare contează foarte mult.",
                "incorrectExplanation": "Niciuna dintre extreme nu este adevărată: modelele mai mari nu sînt întotdeauna mult mai bune, nici cele mai mici întotdeauna superioare, iar legile de putere clare din NLP nu au fost reproduse. Dovezile actuale sînt mixte, iar datele au un rol central."
            }
        }
    ]
};
