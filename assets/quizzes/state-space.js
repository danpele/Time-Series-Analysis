// ============================================================
// Chapter 10 quiz bank: State space models, Kalman filter and Markov switching (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['state-space'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "The two equations",
                "text": "In the state space form $y_t = Z\\alpha_t + \\varepsilon_t$, $\\alpha_{t+1} = T\\alpha_t + R\\eta_t$, which equation describes how the unobserved state evolves?",
                "options": [
                    "The measurement equation",
                    "The transition equation",
                    "Both equations equally",
                    "Neither: the state is constant"
                ],
                "correctExplanation": "The transition equation $\\alpha_{t+1} = T\\alpha_t + R\\eta_t$ moves the hidden state forward in time, as a VAR(1); the measurement equation links the state to the observed $y_t$.",
                "incorrectExplanation": "The measurement equation only says how $y_t$ is generated from the state, and the state is not constant unless $Q = 0$. The dynamics of the state are in the transition equation."
            },
            "ro": {
                "title": "Cele două ecuații",
                "text": "În forma în spațiul stărilor $y_t = Z\\alpha_t + \\varepsilon_t$, $\\alpha_{t+1} = T\\alpha_t + R\\eta_t$, ce ecuație descrie evoluția stării neobservate?",
                "options": [
                    "Ecuația de măsurare",
                    "Ecuația de tranziție",
                    "Ambele ecuații în egală măsură",
                    "Niciuna: starea este constantă"
                ],
                "correctExplanation": "Ecuația de tranziție $\\alpha_{t+1} = T\\alpha_t + R\\eta_t$ mută starea ascunsă înainte în timp, ca un VAR(1); ecuația de măsurare leagă starea de $y_t$ observat.",
                "incorrectExplanation": "Ecuația de măsurare spune doar cum se generează $y_t$ din stare, iar starea nu este constantă decît dacă $Q = 0$. Dinamica stării se află în ecuația de tranziție."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Signal-to-noise ratio",
                "text": "In the local level model $y_t = \\mu_t + \\varepsilon_t$, $\\mu_{t+1} = \\mu_t + \\eta_t$, what does $q = \\sigma^2_\\eta/\\sigma^2_\\varepsilon = 0$ imply?",
                "options": [
                    "The series is a pure random walk",
                    "The series is a local linear trend",
                    "The level is constant: white noise around a fixed mean",
                    "The variance of $y_t$ is zero"
                ],
                "correctExplanation": "With $\\sigma^2_\\eta = 0$ the level never moves, so $y_t$ is white noise around a constant mean; a large $q$ goes towards a random walk.",
                "incorrectExplanation": "A random walk corresponds to $q \\to \\infty$, a local linear trend needs a slope state, and $y_t$ still has the noise variance $\\sigma^2_\\varepsilon$. With $q = 0$ the level is constant."
            },
            "ro": {
                "title": "Raportul semnal--zgomot",
                "text": "În modelul local level $y_t = \\mu_t + \\varepsilon_t$, $\\mu_{t+1} = \\mu_t + \\eta_t$, ce implică $q = \\sigma^2_\\eta/\\sigma^2_\\varepsilon = 0$?",
                "options": [
                    "Seria este un mers aleator pur",
                    "Seria este un local linear trend",
                    "Nivelul este constant: zgomot alb în jurul unei medii fixe",
                    "Varianța lui $y_t$ este zero"
                ],
                "correctExplanation": "Cu $\\sigma^2_\\eta = 0$ nivelul nu se mișcă niciodată, deci $y_t$ este zgomot alb în jurul unei medii constante; un $q$ mare tinde spre un mers aleator.",
                "incorrectExplanation": "Mersul aleator corespunde lui $q \\to \\infty$, local linear trend are nevoie de o stare pentru pantă, iar $y_t$ păstrează varianța zgomotului $\\sigma^2_\\varepsilon$. Cu $q = 0$ nivelul este constant."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The Kalman gain",
                "text": "In the local level model, what is the Kalman gain $K_t$ of the update step?",
                "options": [
                    "$\\sigma^2_\\varepsilon/(P_t + \\sigma^2_\\varepsilon)$",
                    "$\\sigma^2_\\eta/\\sigma^2_\\varepsilon$",
                    "$P_t + \\sigma^2_\\varepsilon$",
                    "$P_t/(P_t + \\sigma^2_\\varepsilon)$"
                ],
                "correctExplanation": "$K_t = P_t/F_t$ with $F_t = P_t + \\sigma^2_\\varepsilon$: the share of the prediction variance in the total variance of the prediction error.",
                "incorrectExplanation": "The first expression is $1 - K_t$, the weight of the old prediction; $\\sigma^2_\\eta/\\sigma^2_\\varepsilon$ is the signal-to-noise ratio $q$; $P_t + \\sigma^2_\\varepsilon$ is $F_t$, the variance of $v_t$."
            },
            "ro": {
                "title": "Cîștigul Kalman",
                "text": "În modelul local level, cît este cîștigul Kalman $K_t$ al pasului de actualizare?",
                "options": [
                    "$\\sigma^2_\\varepsilon/(P_t + \\sigma^2_\\varepsilon)$",
                    "$\\sigma^2_\\eta/\\sigma^2_\\varepsilon$",
                    "$P_t + \\sigma^2_\\varepsilon$",
                    "$P_t/(P_t + \\sigma^2_\\varepsilon)$"
                ],
                "correctExplanation": "$K_t = P_t/F_t$, cu $F_t = P_t + \\sigma^2_\\varepsilon$: ponderea varianței predicției în varianța totală a erorii de predicție.",
                "incorrectExplanation": "Prima expresie este $1 - K_t$, ponderea predicției vechi; $\\sigma^2_\\eta/\\sigma^2_\\varepsilon$ este raportul semnal--zgomot $q$; $P_t + \\sigma^2_\\varepsilon$ este $F_t$, varianța lui $v_t$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Noisy measurements",
                "text": "The measurement noise $\\sigma^2_\\varepsilon$ increases while $P_t$ stays the same. What happens to the Kalman gain?",
                "options": [
                    "It falls: each observation moves the estimate less",
                    "It rises: noisy data deserve more weight",
                    "It does not change",
                    "It becomes negative"
                ],
                "correctExplanation": "With $K_t = P_t/(P_t + \\sigma^2_\\varepsilon)$ a larger noise variance lowers the gain: the filter trusts the model more and the noisy observation less.",
                "incorrectExplanation": "The gain depends on $\\sigma^2_\\varepsilon$ and always stays between 0 and 1; more noise means less weight on the data, not more."
            },
            "ro": {
                "title": "Măsurători zgomotoase",
                "text": "Zgomotul de măsurare $\\sigma^2_\\varepsilon$ crește, iar $P_t$ rămîne același. Ce se întîmplă cu cîștigul Kalman?",
                "options": [
                    "Scade: fiecare observație mută estimarea mai puțin",
                    "Crește: datele zgomotoase merită o pondere mai mare",
                    "Nu se schimbă",
                    "Devine negativ"
                ],
                "correctExplanation": "Cu $K_t = P_t/(P_t + \\sigma^2_\\varepsilon)$, o varianță mai mare a zgomotului reduce cîștigul: filtrul are mai multă încredere în model și mai puțină în observația zgomotoasă.",
                "incorrectExplanation": "Cîștigul depinde de $\\sigma^2_\\varepsilon$ și rămîne mereu între 0 și 1; mai mult zgomot înseamnă o pondere mai mică pentru date, nu mai mare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "One update by hand",
                "text": "Local level: prediction $a_t = 10$, $P_t = 4$, $\\sigma^2_\\varepsilon = 4$, and the observation is $y_t = 12$. What is the filtered level $a_{t|t}$?",
                "options": [
                    "10",
                    "11",
                    "12",
                    "10.5"
                ],
                "correctExplanation": "$F_t = 8$, $K_t = 0.5$, $v_t = 2$, so $a_{t|t} = 10 + 0.5 \\cdot 2 = 11$: halfway between the prediction and the observation.",
                "incorrectExplanation": "10 ignores the observation and 12 ignores the prediction; 10.5 would need $K_t = 0.25$. With equal variances the gain is 0.5 and the filtered level is 11."
            },
            "ro": {
                "title": "O actualizare de mînă",
                "text": "Local level: predicția $a_t = 10$, $P_t = 4$, $\\sigma^2_\\varepsilon = 4$, iar observația este $y_t = 12$. Cît este nivelul filtrat $a_{t|t}$?",
                "options": [
                    "10",
                    "11",
                    "12",
                    "10,5"
                ],
                "correctExplanation": "$F_t = 8$, $K_t = 0{,}5$, $v_t = 2$, deci $a_{t|t} = 10 + 0{,}5 \\cdot 2 = 11$: la jumătatea distanței dintre predicție și observație.",
                "incorrectExplanation": "10 ignoră observația, iar 12 ignoră predicția; 10,5 ar cere $K_t = 0{,}25$. Cu varianțe egale cîștigul este 0,5, iar nivelul filtrat este 11."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "A missing observation",
                "text": "In the local level filter $y_t$ is missing. What does the filter do at $t$?",
                "options": [
                    "It sets $y_t$ to the sample mean and updates",
                    "It stops and restarts with a diffuse prior",
                    "It skips the update: $a_{t+1} = a_t$ and $P_{t+1} = P_t + \\sigma^2_\\eta$",
                    "It removes period $t$ and joins $t-1$ to $t+1$"
                ],
                "correctExplanation": "Without an observation there is nothing to update: the prediction is carried forward and its variance grows by $\\sigma^2_\\eta$; the likelihood simply omits that term.",
                "incorrectExplanation": "Imputing the mean distorts the level, restarting throws away information, and joining the periods ignores the extra level shock. The filter just skips the update step."
            },
            "ro": {
                "title": "O observație lipsă",
                "text": "În filtrul local level lipsește $y_t$. Ce face filtrul la momentul $t$?",
                "options": [
                    "Înlocuiește $y_t$ cu media eșantionului și actualizează",
                    "Se oprește și repornește cu o distribuție a priori difuză",
                    "Sare peste actualizare: $a_{t+1} = a_t$ și $P_{t+1} = P_t + \\sigma^2_\\eta$",
                    "Elimină perioada $t$ și leagă $t-1$ de $t+1$"
                ],
                "correctExplanation": "Fără observație nu are ce actualiza: predicția este dusă mai departe, iar varianța ei crește cu $\\sigma^2_\\eta$; verosimilitatea omite pur și simplu acel termen.",
                "incorrectExplanation": "Înlocuirea cu media distorsionează nivelul, repornirea aruncă informație, iar legarea perioadelor ignoră șocul suplimentar al nivelului. Filtrul sare doar peste pasul de actualizare."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Exponential smoothing and the Kalman filter",
                "text": "In the steady state of the local level filter, $a_{t+1} = \\bar K y_t + (1 - \\bar K)a_t$. What does this show?",
                "options": [
                    "The filter becomes a random walk forecast",
                    "The filter becomes the sample mean",
                    "The gain converges to 1",
                    "Simple exponential smoothing is the steady-state Kalman filter, with $\\alpha = \\bar K$"
                ],
                "correctExplanation": "The recursion is exactly SES with smoothing weight $\\alpha = \\bar K$: exponential smoothing gives optimal forecasts when the data follow a local level model (Muth, 1960).",
                "incorrectExplanation": "A random walk forecast needs $\\bar K = 1$ and the sample mean needs $\\bar K \\to 0$; the steady-state gain lies strictly between 0 and 1 for $0 < q < \\infty$."
            },
            "ro": {
                "title": "Netezirea exponențială și filtrul Kalman",
                "text": "În starea de echilibru a filtrului local level, $a_{t+1} = \\bar K y_t + (1 - \\bar K)a_t$. Ce arată aceasta?",
                "options": [
                    "Filtrul devine prognoza mersului aleator",
                    "Filtrul devine media eșantionului",
                    "Cîștigul converge la 1",
                    "Netezirea exponențială simplă este filtrul Kalman în echilibru, cu $\\alpha = \\bar K$"
                ],
                "correctExplanation": "Recurența este exact SES cu ponderea de netezire $\\alpha = \\bar K$: netezirea exponențială dă prognoze optime cînd datele urmează un model local level (Muth, 1960).",
                "incorrectExplanation": "Prognoza mersului aleator cere $\\bar K = 1$, iar media eșantionului cere $\\bar K \\to 0$; cîștigul de echilibru este strict între 0 și 1 pentru $0 < q < \\infty$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "From SES to the local level",
                "text": "An SES model has $\\alpha = 0.5$. What is the signal-to-noise ratio $q$ of the equivalent local level model?",
                "options": [
                    "$q = \\alpha^2/(1 - \\alpha) = 0.5$",
                    "$q = \\alpha = 0.5$ by definition",
                    "$q = 1 - \\alpha = 0.5$ by definition",
                    "$q = 0.25$"
                ],
                "correctExplanation": "From $\\sigma^2_\\eta = \\bar K\\bar P$ and $\\bar P = \\alpha\\sigma^2_\\varepsilon/(1 - \\alpha)$ we get $q = \\alpha^2/(1 - \\alpha) = 0.25/0.5 = 0.5$. Here the number equals $\\alpha$ by coincidence.",
                "incorrectExplanation": "$q$ is not $\\alpha$ or $1 - \\alpha$ in general (for $\\alpha = 0.3$, $q = 0.129$), and 0.25 is only $\\alpha^2$. The map is $q = \\alpha^2/(1 - \\alpha)$."
            },
            "ro": {
                "title": "De la SES la local level",
                "text": "Un model SES are $\\alpha = 0{,}5$. Cît este raportul semnal--zgomot $q$ al modelului local level echivalent?",
                "options": [
                    "$q = \\alpha^2/(1 - \\alpha) = 0{,}5$",
                    "$q = \\alpha = 0{,}5$, prin definiție",
                    "$q = 1 - \\alpha = 0{,}5$, prin definiție",
                    "$q = 0{,}25$"
                ],
                "correctExplanation": "Din $\\sigma^2_\\eta = \\bar K\\bar P$ și $\\bar P = \\alpha\\sigma^2_\\varepsilon/(1 - \\alpha)$ rezultă $q = \\alpha^2/(1 - \\alpha) = 0{,}25/0{,}5 = 0{,}5$. Aici valoarea coincide întîmplător cu $\\alpha$.",
                "incorrectExplanation": "$q$ nu este, în general, $\\alpha$ sau $1 - \\alpha$ (pentru $\\alpha = 0{,}3$, $q = 0{,}129$), iar 0,25 este doar $\\alpha^2$. Relația este $q = \\alpha^2/(1 - \\alpha)$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The likelihood of a state space model",
                "text": "How does the Kalman filter deliver the Gaussian log-likelihood?",
                "options": [
                    "From the smoothed states only",
                    "By counting the missing values",
                    "From the prediction errors: $-\\frac12\\sum_t(\\log 2\\pi + \\log F_t + v_t^2/F_t)$",
                    "It cannot: state space models are estimated by OLS"
                ],
                "correctExplanation": "The prediction-error decomposition writes the joint density as a product of one-step densities $N(Z_ta_t, F_t)$; the filter gives $v_t$ and $F_t$, and an optimiser maximises the sum over the parameters.",
                "incorrectExplanation": "The smoothed states and the missing values do not give the likelihood, and OLS does not apply to unobserved states. The likelihood comes from the one-step prediction errors and their variances."
            },
            "ro": {
                "title": "Verosimilitatea unui model în spațiul stărilor",
                "text": "Cum dă filtrul Kalman log-verosimilitatea gaussiană?",
                "options": [
                    "Doar din stările netezite",
                    "Numărînd valorile lipsă",
                    "Din erorile de predicție: $-\\frac12\\sum_t(\\log 2\\pi + \\log F_t + v_t^2/F_t)$",
                    "Nu poate: modelele în spațiul stărilor se estimează prin OLS"
                ],
                "correctExplanation": "Descompunerea erorilor de predicție scrie densitatea comună ca produs de densități la un pas $N(Z_ta_t, F_t)$; filtrul dă $v_t$ și $F_t$, iar un optimizator maximizează suma după parametri.",
                "incorrectExplanation": "Stările netezite și valorile lipsă nu dau verosimilitatea, iar OLS nu se aplică stărilor neobservate. Verosimilitatea provine din erorile de predicție la un pas și din varianțele lor."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Diffuse initialisation",
                "text": "Why is a non-stationary state (such as a random-walk level) started with a very large $P_1$?",
                "options": [
                    "To make the gain zero at the start",
                    "Because the stationary variance is negative",
                    "To force the filter to ignore the first observation",
                    "Because we know nothing about the initial level: a diffuse prior lets the first observation determine it"
                ],
                "correctExplanation": "A random walk has no stationary distribution, so the initial level is unknown; with $P_1$ huge, $K_1 \\approx 1$ and the first observation sets the level; its likelihood term is left out.",
                "incorrectExplanation": "A large $P_1$ makes the gain close to 1, not 0, so the first observation counts fully; the stationary variance of a random walk does not exist rather than being negative."
            },
            "ro": {
                "title": "Inițializarea difuză",
                "text": "De ce pornește o stare nestaționară (de exemplu un nivel de tip mers aleator) cu un $P_1$ foarte mare?",
                "options": [
                    "Pentru ca la început cîștigul să fie zero",
                    "Pentru că varianța staționară este negativă",
                    "Pentru a forța filtrul să ignore prima observație",
                    "Pentru că nu știm nimic despre nivelul inițial: o distribuție a priori difuză lasă prima observație să îl determine"
                ],
                "correctExplanation": "Un mers aleator nu are distribuție staționară, deci nivelul inițial este necunoscut; cu $P_1$ uriaș, $K_1 \\approx 1$, iar prima observație fixează nivelul; termenul ei din verosimilitate este omis.",
                "incorrectExplanation": "Un $P_1$ mare face cîștigul apropiat de 1, nu de 0, deci prima observație contează complet; varianța staționară a unui mers aleator nu există, nu este negativă."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Forecasting with the local level model",
                "text": "What do the $h$-step forecasts of a local level model look like?",
                "options": [
                    "Flat at $a_{n+1}$, with a variance that grows linearly in $h$",
                    "A straight line with the last slope",
                    "Converging to the sample mean",
                    "Identical to the last observation $y_n$, with constant variance"
                ],
                "correctExplanation": "Forecasting treats future values as missing: the level is carried forward, so $\\hat y_{n+h} = a_{n+1}$, with variance $P_{n+1} + (h-1)\\sigma^2_\\eta + \\sigma^2_\\varepsilon$.",
                "incorrectExplanation": "A sloped line needs a local linear trend, convergence to the mean needs a stationary model, and $a_{n+1}$ differs from $y_n$ unless the gain is 1; the variance grows with $h$."
            },
            "ro": {
                "title": "Prognoza cu modelul local level",
                "text": "Cum arată prognozele la $h$ pași ale unui model local level?",
                "options": [
                    "Constante, egale cu $a_{n+1}$, cu o varianță care crește liniar în $h$",
                    "O dreaptă cu ultima pantă",
                    "Converg spre media eșantionului",
                    "Egale cu ultima observație $y_n$, cu varianță constantă"
                ],
                "correctExplanation": "Prognoza tratează valorile viitoare ca lipsă: nivelul este dus mai departe, deci $\\hat y_{n+h} = a_{n+1}$, cu varianța $P_{n+1} + (h-1)\\sigma^2_\\eta + \\sigma^2_\\varepsilon$.",
                "incorrectExplanation": "O dreaptă înclinată cere un local linear trend, convergența spre medie cere un model staționar, iar $a_{n+1}$ diferă de $y_n$ dacă cîștigul nu este 1; varianța crește cu $h$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "AR(2) in state space form",
                "text": "With state $\\alpha_t = (y_t, y_{t-1})'$, what is the transition matrix $T$ of an AR(2) $y_t = \\phi_1y_{t-1} + \\phi_2y_{t-2} + u_t$?",
                "options": [
                    "$\\begin{pmatrix}1 & 1\\\\ 0 & 1\\end{pmatrix}$",
                    "$\\begin{pmatrix}\\phi_1 & \\phi_2\\\\ 1 & 0\\end{pmatrix}$",
                    "$\\begin{pmatrix}\\phi_1 & 0\\\\ 0 & \\phi_2\\end{pmatrix}$",
                    "$\\begin{pmatrix}0 & 1\\\\ \\phi_2 & \\phi_1\\end{pmatrix}$ with $H = 1$"
                ],
                "correctExplanation": "The first row gives $y_{t+1} = \\phi_1y_t + \\phi_2y_{t-1} + u_{t+1}$, the second row shifts $y_t$ down: the companion matrix; $Z = (1, 0)$ and $H = 0$.",
                "incorrectExplanation": "The first matrix is the local linear trend; a diagonal matrix would make two separate AR(1) processes; the last matrix has the wrong order for this state and adds a noise that an AR(2) does not have."
            },
            "ro": {
                "title": "AR(2) în forma în spațiul stărilor",
                "text": "Cu starea $\\alpha_t = (y_t, y_{t-1})'$, care este matricea de tranziție $T$ a unui AR(2) $y_t = \\phi_1y_{t-1} + \\phi_2y_{t-2} + u_t$?",
                "options": [
                    "$\\begin{pmatrix}1 & 1\\\\ 0 & 1\\end{pmatrix}$",
                    "$\\begin{pmatrix}\\phi_1 & \\phi_2\\\\ 1 & 0\\end{pmatrix}$",
                    "$\\begin{pmatrix}\\phi_1 & 0\\\\ 0 & \\phi_2\\end{pmatrix}$",
                    "$\\begin{pmatrix}0 & 1\\\\ \\phi_2 & \\phi_1\\end{pmatrix}$, cu $H = 1$"
                ],
                "correctExplanation": "Primul rînd dă $y_{t+1} = \\phi_1y_t + \\phi_2y_{t-1} + u_{t+1}$, al doilea mută $y_t$ în jos: matricea companion; $Z = (1, 0)$ și $H = 0$.",
                "incorrectExplanation": "Prima matrice este cea a modelului local linear trend; o matrice diagonală ar da două procese AR(1) separate; ultima matrice are ordinea greșită pentru această stare și adaugă un zgomot pe care AR(2) nu îl are."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The HP filter and state space",
                "text": "In which sense is the Hodrick--Prescott filter a state space method?",
                "options": [
                    "It is the filtered (one-sided) trend of a local level model",
                    "It is the OLS trend of a linear regression on time",
                    "It is the smoothed trend of a smooth-trend plus white-noise model with $\\lambda = \\sigma^2_\\varepsilon/\\sigma^2_\\zeta$",
                    "It is a Markov-switching model with two regimes"
                ],
                "correctExplanation": "The HP trend solves the same problem as the Kalman smoother of a UC model whose trend is an integrated random walk and whose gap is white noise; $\\lambda = 1600$ fixes the variance ratio instead of estimating it.",
                "incorrectExplanation": "HP is two-sided (a smoother, not a filter), it is not a straight-line trend unless $\\lambda \\to \\infty$, and it has no regimes."
            },
            "ro": {
                "title": "Filtrul HP și spațiul stărilor",
                "text": "În ce sens este filtrul Hodrick--Prescott o metodă în spațiul stărilor?",
                "options": [
                    "Este trendul filtrat (unilateral) al unui model local level",
                    "Este trendul OLS al unei regresii liniare pe timp",
                    "Este trendul netezit al unui model cu trend neted plus zgomot alb, cu $\\lambda = \\sigma^2_\\varepsilon/\\sigma^2_\\zeta$",
                    "Este un model Markov switching cu două regimuri"
                ],
                "correctExplanation": "Trendul HP rezolvă aceeași problemă ca netezitorul Kalman al unui model UC al cărui trend este un mers aleator integrat și a cărui deviație este zgomot alb; $\\lambda = 1600$ fixează raportul varianțelor în loc să îl estimeze.",
                "incorrectExplanation": "HP este bilateral (un netezitor, nu un filtru), nu este un trend liniar decît dacă $\\lambda \\to \\infty$ și nu are regimuri."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The critique of Hamilton (2018)",
                "text": "Which alternative to the HP filter does Hamilton (2018) propose?",
                "options": [
                    "A larger $\\lambda$, such as 129,600",
                    "A Markov-switching model of GDP growth",
                    "A centred moving average of 8 quarters",
                    "The residual of an OLS regression of $y_{t+8}$ on a constant and $y_t, y_{t-1}, y_{t-2}, y_{t-3}$"
                ],
                "correctExplanation": "The regression filter defines the cycle as what could not be predicted two years earlier; it is one-sided by construction and avoids the spurious cycles and end-point revisions of HP.",
                "incorrectExplanation": "Changing $\\lambda$ keeps the problems of a two-sided smoother, a centred average also uses future data, and a regime model answers a different question."
            },
            "ro": {
                "title": "Critica lui Hamilton (2018)",
                "text": "Ce alternativă la filtrul HP propune Hamilton (2018)?",
                "options": [
                    "Un $\\lambda$ mai mare, de exemplu 129 600",
                    "Un model Markov switching pentru creșterea PIB",
                    "O medie mobilă centrată de 8 trimestre",
                    "Reziduul unei regresii OLS a lui $y_{t+8}$ pe o constantă și $y_t, y_{t-1}, y_{t-2}, y_{t-3}$"
                ],
                "correctExplanation": "Filtrul de regresie definește ciclul ca ceea ce nu putea fi prezis cu doi ani înainte; este unilateral prin construcție și evită ciclurile aparente și revizuirile de la capătul eșantionului ale HP.",
                "incorrectExplanation": "Schimbarea lui $\\lambda$ păstrează problemele unui netezitor bilateral, o medie centrată folosește și ea date viitoare, iar un model cu regimuri răspunde la altă întrebare."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Real-time output gaps",
                "text": "For Romania in 2008Q3 the HP gap computed in real time was about 2%, while the HP gap computed with today's data is about 9%. What is the lesson?",
                "options": [
                    "Two-sided gaps are revised heavily at the end of the sample; real-time judgements need one-sided (filtered) estimates",
                    "The economy was not overheating in 2008",
                    "The HP filter is wrong only for Romania",
                    "Real-time data are always more accurate than later data"
                ],
                "correctExplanation": "The last points of a two-sided filter change as new quarters arrive; the overheating was invisible in real time. Policy needs filtered estimates with their uncertainty.",
                "incorrectExplanation": "Hindsight shows the overheating clearly; the end-point problem affects every two-sided filter and every country; real-time estimates are the less accurate ones."
            },
            "ro": {
                "title": "Deviații PIB în timp real",
                "text": "Pentru România, în T3 2008 deviația HP calculată în timp real era de aproximativ 2%, iar deviația HP calculată cu datele de azi este de aproximativ 9%. Care este lecția?",
                "options": [
                    "Deviațiile bilaterale se revizuiesc puternic la capătul eșantionului; judecățile în timp real cer estimări unilaterale (filtrate)",
                    "Economia nu era supraîncălzită în 2008",
                    "Filtrul HP este greșit doar pentru România",
                    "Datele în timp real sînt întotdeauna mai exacte decît datele ulterioare"
                ],
                "correctExplanation": "Ultimele puncte ale unui filtru bilateral se schimbă pe măsură ce sosesc trimestre noi; supraîncălzirea era invizibilă în timp real. Politica economică are nevoie de estimări filtrate, cu incertitudinea lor.",
                "incorrectExplanation": "Retrospectiv supraîncălzirea se vede clar; problema capătului de eșantion afectează orice filtru bilateral și orice țară; estimările în timp real sînt cele mai puțin exacte."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Regression with time-varying parameters",
                "text": "In the TVP regression $y_t = \\alpha + \\beta_tx_t + \\varepsilon_t$, $\\beta_{t+1} = \\beta_t + \\eta_t$, what happens when $\\sigma^2_\\eta = 0$?",
                "options": [
                    "The beta becomes a random walk",
                    "The Kalman filter computes the ordinary least squares estimate recursively",
                    "The model cannot be estimated",
                    "The beta equals the rolling-window OLS beta"
                ],
                "correctExplanation": "With no shocks to $\\beta_t$ the coefficient is constant; the filter then updates the OLS estimate one observation at a time (recursive least squares).",
                "incorrectExplanation": "A random-walk beta needs $\\sigma^2_\\eta > 0$; the model is still estimable; a rolling window drops old observations, which the constant-coefficient filter does not."
            },
            "ro": {
                "title": "Regresie cu parametri variabili în timp",
                "text": "În regresia TVP $y_t = \\alpha + \\beta_tx_t + \\varepsilon_t$, $\\beta_{t+1} = \\beta_t + \\eta_t$, ce se întîmplă cînd $\\sigma^2_\\eta = 0$?",
                "options": [
                    "Beta devine un mers aleator",
                    "Filtrul Kalman calculează recursiv estimarea prin cele mai mici pătrate",
                    "Modelul nu poate fi estimat",
                    "Beta este egal cu beta OLS pe fereastră mobilă"
                ],
                "correctExplanation": "Fără șocuri pentru $\\beta_t$, coeficientul este constant; filtrul actualizează atunci estimarea OLS cîte o observație (cele mai mici pătrate recursive).",
                "incorrectExplanation": "Un beta de tip mers aleator cere $\\sigma^2_\\eta > 0$; modelul rămîne estimabil; o fereastră mobilă elimină observațiile vechi, ceea ce filtrul cu coeficient constant nu face."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Expected duration of a regime",
                "text": "In a two-regime Markov chain the probability of staying in regime 1 is $p_{11} = 0.8$. What is the expected duration of regime 1?",
                "options": [
                    "0.8 periods",
                    "4 periods",
                    "5 periods",
                    "1.25 periods"
                ],
                "correctExplanation": "The time spent in a regime is geometric, so the expected duration is $1/(1 - p_{11}) = 1/0.2 = 5$ periods.",
                "incorrectExplanation": "0.8 is the probability itself, 4 is $p_{11}/(1 - p_{11})$, a common error, and 1.25 is $1/p_{11}$. The expected duration is $1/(1 - p_{11})$."
            },
            "ro": {
                "title": "Durata așteptată a unui regim",
                "text": "Într-un lanț Markov cu două regimuri probabilitatea de a rămîne în regimul 1 este $p_{11} = 0{,}8$. Cît este durata așteptată a regimului 1?",
                "options": [
                    "0,8 perioade",
                    "4 perioade",
                    "5 perioade",
                    "1,25 perioade"
                ],
                "correctExplanation": "Timpul petrecut într-un regim este geometric, deci durata așteptată este $1/(1 - p_{11}) = 1/0{,}2 = 5$ perioade.",
                "incorrectExplanation": "0,8 este chiar probabilitatea, 4 este $p_{11}/(1 - p_{11})$, o greșeală frecventă, iar 1,25 este $1/p_{11}$. Durata așteptată este $1/(1 - p_{11})$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Ergodic probabilities",
                "text": "With $p_{11} = 0.75$ (recession) and $p_{22} = 0.95$ (expansion), what share of quarters is spent in recession in the long run?",
                "options": [
                    "1/4",
                    "3/4",
                    "1/20",
                    "1/6"
                ],
                "correctExplanation": "$\\pi_1 = (1 - p_{22})/(2 - p_{11} - p_{22}) = 0.05/0.30 = 1/6$: on average one quarter in six is a recession quarter.",
                "incorrectExplanation": "1/4 and 3/4 confuse the ergodic probability with the transition probabilities, and 1/20 is $1 - p_{22}$. The long-run share is $0.05/0.30 = 1/6$."
            },
            "ro": {
                "title": "Probabilități ergodice",
                "text": "Cu $p_{11} = 0{,}75$ (recesiune) și $p_{22} = 0{,}95$ (expansiune), ce fracțiune din trimestre este petrecută în recesiune pe termen lung?",
                "options": [
                    "1/4",
                    "3/4",
                    "1/20",
                    "1/6"
                ],
                "correctExplanation": "$\\pi_1 = (1 - p_{22})/(2 - p_{11} - p_{22}) = 0{,}05/0{,}30 = 1/6$: în medie un trimestru din șase este trimestru de recesiune.",
                "incorrectExplanation": "1/4 și 3/4 confundă probabilitatea ergodică cu probabilitățile de tranziție, iar 1/20 este $1 - p_{22}$. Fracțiunea pe termen lung este $0{,}05/0{,}30 = 1/6$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The Hamilton filter",
                "text": "How does the Hamilton filter update the probability of regime 1 when $y_t$ arrives?",
                "options": [
                    "By Bayes' rule: predicted probability times the density of $y_t$ in regime 1, divided by the total density",
                    "By the Kalman gain $P_t/(P_t + \\sigma^2)$",
                    "By setting it to 1 if $y_t < 0$",
                    "By averaging the last four probabilities"
                ],
                "correctExplanation": "$\\Pr(S_t = 1 \\mid Y_t) = \\Pr(S_t = 1 \\mid Y_{t-1})f_1(y_t)/\\sum_j\\Pr(S_t = j \\mid Y_{t-1})f_j(y_t)$; the denominator is the likelihood contribution of $y_t$.",
                "incorrectExplanation": "The Kalman gain belongs to continuous states, a sign rule ignores the densities, and an average of past probabilities ignores the new observation. The update is Bayes' rule."
            },
            "ro": {
                "title": "Filtrul Hamilton",
                "text": "Cum actualizează filtrul Hamilton probabilitatea regimului 1 cînd sosește $y_t$?",
                "options": [
                    "Prin regula lui Bayes: probabilitatea prezisă înmulțită cu densitatea lui $y_t$ în regimul 1, împărțită la densitatea totală",
                    "Prin cîștigul Kalman $P_t/(P_t + \\sigma^2)$",
                    "Punînd-o egală cu 1 dacă $y_t < 0$",
                    "Prin media ultimelor patru probabilități"
                ],
                "correctExplanation": "$\\Pr(S_t = 1 \\mid Y_t) = \\Pr(S_t = 1 \\mid Y_{t-1})f_1(y_t)/\\sum_j\\Pr(S_t = j \\mid Y_{t-1})f_j(y_t)$; numitorul este contribuția lui $y_t$ la verosimilitate.",
                "incorrectExplanation": "Cîștigul Kalman aparține stărilor continue, o regulă după semn ignoră densitățile, iar media probabilităților trecute ignoră noua observație. Actualizarea este regula lui Bayes."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Calling a recession in real time",
                "text": "A forecaster wants to know, at the end of each quarter, whether the economy is in recession. Which output of a Markov-switching model should she use?",
                "options": [
                    "The smoothed probabilities",
                    "The filtered probabilities $\\Pr(S_t = j \\mid y_1, \\dots, y_t)$",
                    "The ergodic probabilities",
                    "The transition probabilities $p_{ij}$"
                ],
                "correctExplanation": "Only the filtered probabilities use the information available at time $t$; in 2008 they crossed 0.5 three quarters after the smoothed ones.",
                "incorrectExplanation": "Smoothed probabilities use future data, ergodic probabilities are long-run averages, and transition probabilities do not depend on the data of the current quarter."
            },
            "ro": {
                "title": "Anunțarea unei recesiuni în timp real",
                "text": "Un prognozator vrea să știe, la sfîrșitul fiecărui trimestru, dacă economia este în recesiune. Ce rezultat al unui model Markov switching trebuie să folosească?",
                "options": [
                    "Probabilitățile netezite",
                    "Probabilitățile filtrate $\\Pr(S_t = j \\mid y_1, \\dots, y_t)$",
                    "Probabilitățile ergodice",
                    "Probabilitățile de tranziție $p_{ij}$"
                ],
                "correctExplanation": "Doar probabilitățile filtrate folosesc informația disponibilă la momentul $t$; în 2008 ele au trecut de 0,5 cu trei trimestre după cele netezite.",
                "incorrectExplanation": "Probabilitățile netezite folosesc date viitoare, probabilitățile ergodice sînt medii pe termen lung, iar probabilitățile de tranziție nu depind de datele trimestrului curent."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "What regime did the model find?",
                "text": "A two-regime model with switching mean and variance, fitted to US GDP growth 1947--2019, gives regimes lasting about 40 quarters with very different variances. What did it most likely find?",
                "options": [
                    "The NBER recessions",
                    "Seasonality",
                    "The Great Moderation: high volatility before the mid-1980s and low volatility after",
                    "The COVID-19 pandemic"
                ],
                "correctExplanation": "With a switching variance the likelihood prefers the large and lasting fall in volatility around 1984 to the short recessions; the regimes must be checked against known events.",
                "incorrectExplanation": "Recessions last about 3--4 quarters, not 40; the data are seasonally adjusted; and 2020 lies outside 1947--2019."
            },
            "ro": {
                "title": "Ce regim a găsit modelul?",
                "text": "Un model cu două regimuri, cu medie și varianță variabile, estimat pe creșterea PIB din SUA în 1947--2019, dă regimuri de aproximativ 40 de trimestre, cu varianțe foarte diferite. Ce a găsit cel mai probabil?",
                "options": [
                    "Recesiunile NBER",
                    "Sezonalitatea",
                    "Marea Moderație: volatilitate mare înainte de mijlocul anilor 1980 și volatilitate mică după",
                    "Pandemia de COVID-19"
                ],
                "correctExplanation": "Cu o varianță variabilă, verosimilitatea preferă scăderea mare și durabilă a volatilității din jurul anului 1984 recesiunilor scurte; regimurile trebuie verificate în raport cu evenimente cunoscute.",
                "incorrectExplanation": "Recesiunile durează aproximativ 3--4 trimestre, nu 40; datele sînt ajustate sezonier; iar 2020 este în afara intervalului 1947--2019."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Regimes, GARCH and long memory",
                "text": "Series simulated from a two-regime model with constant variance inside each regime give a GARCH(1,1) persistence $\\alpha + \\beta$ of about 0.9 and a positive estimate of $d$. What does this show?",
                "options": [
                    "The simulation contains a GARCH process",
                    "Regime models are always better than GARCH",
                    "Long memory and regimes are the same thing",
                    "Shifts in the level of variance can imitate GARCH persistence and long memory"
                ],
                "correctExplanation": "Lamoureux and Lastrapes (1990) and Diebold and Inoue (2001): regime shifts produce slowly decaying autocorrelations of $|r_t|$, high $\\alpha + \\beta$ and $\\hat d > 0$ although there is no GARCH or fractional integration.",
                "incorrectExplanation": "The simulated series contain no GARCH by construction; the result does not rank the models; the two mechanisms differ but can produce similar statistics, so models must be compared by likelihood and forecasts."
            },
            "ro": {
                "title": "Regimuri, GARCH și memorie lungă",
                "text": "Seriile simulate dintr-un model cu două regimuri, cu varianță constantă în fiecare regim, dau o persistență GARCH(1,1) $\\alpha + \\beta$ de aproximativ 0,9 și o estimare pozitivă a lui $d$. Ce arată aceasta?",
                "options": [
                    "Simularea conține un proces GARCH",
                    "Modelele cu regimuri sînt întotdeauna mai bune decît GARCH",
                    "Memoria lungă și regimurile sînt același lucru",
                    "Schimbările nivelului varianței pot imita persistența GARCH și memoria lungă"
                ],
                "correctExplanation": "Lamoureux și Lastrapes (1990) și Diebold și Inoue (2001): schimbările de regim produc autocorelații ale lui $|r_t|$ care scad lent, $\\alpha + \\beta$ mare și $\\hat d > 0$, deși nu există GARCH sau integrare fracționară.",
                "incorrectExplanation": "Seriile simulate nu conțin GARCH prin construcție; rezultatul nu clasifică modelele; cele două mecanisme diferă, dar pot produce statistici asemănătoare, deci modelele se compară după verosimilitate și prognoze."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Dynamic factor models and nowcasting",
                "text": "Why can a dynamic factor model estimated by the Kalman filter produce a nowcast when the latest values of some indicators are not yet published?",
                "options": [
                    "The filter treats the unpublished values as missing and updates the common factor with the indicators already available",
                    "It replaces them with zeros",
                    "It waits until all series are available",
                    "It uses only GDP"
                ],
                "correctExplanation": "Missing values simply skip part of the update; the ragged edge of the data set is handled automatically, which is the basis of nowcasting (Giannone, Reichlin and Small, 2008).",
                "incorrectExplanation": "Zeros would bias the factor, waiting defeats the purpose of a nowcast, and GDP is the quarterly target, not the monthly input."
            },
            "ro": {
                "title": "Modele cu factori dinamici și nowcasting",
                "text": "De ce poate un model cu factori dinamici, estimat cu filtrul Kalman, să dea un nowcast cînd ultimele valori ale unor indicatori nu sînt încă publicate?",
                "options": [
                    "Filtrul tratează valorile nepublicate ca lipsă și actualizează factorul comun cu indicatorii deja disponibili",
                    "Le înlocuiește cu zero",
                    "Așteaptă pînă cînd toate seriile sînt disponibile",
                    "Folosește doar PIB-ul"
                ],
                "correctExplanation": "Valorile lipsă sar peste o parte a actualizării; marginea neregulată a setului de date este tratată automat, ceea ce stă la baza nowcasting-ului (Giannone, Reichlin și Small, 2008).",
                "incorrectExplanation": "Valorile zero ar deplasa factorul, așteptarea anulează scopul unui nowcast, iar PIB-ul este ținta trimestrială, nu datele lunare de intrare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Several maxima of the likelihood",
                "text": "Hamilton's MS-AR(4) on US GDP growth 1951--1984 has two local maxima with log-likelihoods $-189.47$ and $-189.16$: a recession regime of about four quarters, and a regime of isolated sharp quarters. How should you choose?",
                "options": [
                    "Always take the higher likelihood, without further checks",
                    "Compare the economic meaning and robustness (dates, samples, starting values): a difference of 0.3 in log-likelihood cannot decide",
                    "Average the two solutions",
                    "Choose the solution with the shorter regime"
                ],
                "correctExplanation": "The two maxima are almost equally likely; the recession regime matches the NBER dates (94% of recession quarters found against 10%), so meaning and robustness decide.",
                "incorrectExplanation": "A tiny likelihood difference is not decisive, averaging two maxima gives no valid model, and the length of the regime is not a criterion in itself."
            },
            "ro": {
                "title": "Mai multe maxime ale verosimilității",
                "text": "MS-AR(4) al lui Hamilton pe creșterea PIB din SUA în 1951--1984 are două maxime locale, cu log-verosimilitățile $-189{,}47$ și $-189{,}16$: un regim de recesiune de aproximativ patru trimestre și un regim de trimestre izolate cu scăderi bruște. Cum alegeți?",
                "options": [
                    "Întotdeauna maximul cu verosimilitatea mai mare, fără alte verificări",
                    "Comparați înțelesul economic și robustețea (date, eșantioane, valori de pornire): o diferență de 0,3 în log-verosimilitate nu poate decide",
                    "Faceți media celor două soluții",
                    "Alegeți soluția cu regimul mai scurt"
                ],
                "correctExplanation": "Cele două maxime sînt aproape la fel de verosimile; regimul de recesiune se potrivește cu datările NBER (94% din trimestrele de recesiune găsite, față de 10%), deci decid înțelesul și robustețea.",
                "incorrectExplanation": "O diferență foarte mică de verosimilitate nu este decisivă, media a două maxime nu dă un model valid, iar lungimea regimului nu este un criteriu în sine."
            }
        }
    ]
};
