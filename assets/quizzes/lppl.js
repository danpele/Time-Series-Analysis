// ============================================================
// Chapter 13 quiz bank: Speculative bubbles: LPPL models (EN + RO)
// 10 questions ported from the 2025/2026 site; 10 drawn per attempt.
// The bank grows to 24 questions when the chapter is rebuilt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['lppl'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Super-exponential growth",
                "text": "What is the main mathematical signature of a speculative bubble in the LPPL framework?",
                "options": [
                    "The price grows linearly in time",
                    "The log-price grows faster than linearly (super-exponential growth)",
                    "Volatility is constant",
                    "Returns follow the Normal distribution"
                ],
                "correctExplanation": "In a bubble, $\\ln P(t) \\approx A + B(t_c - t)^m$ with $B < 0$ and $0 < m < 1$. The growth rate of the price itself increases as $t \\to t_c$, so growth is faster than exponential and ends in a finite-time singularity.",
                "incorrectExplanation": "Linear price growth is slower than exponential, and constant volatility or Normal returns are statements about risk, not about the growth path. The LPPL signature is an accelerating growth rate: log-price grows faster than linearly."
            },
            "ro": {
                "title": "Creștere super-exponențială",
                "text": "Care este principala semnătură matematică a unei bule speculative în cadrul LPPL?",
                "options": [
                    "Prețul crește liniar în timp",
                    "Logaritmul prețului crește mai rapid decît liniar (creștere super-exponențială)",
                    "Volatilitatea este constantă",
                    "Randamentele urmează distribuția Normală"
                ],
                "correctExplanation": "Într-o bulă, $\\ln P(t) \\approx A + B(t_c - t)^m$, cu $B < 0$ și $0 < m < 1$. Rata de creștere a prețului crește ea însăși cînd $t \\to t_c$, deci creșterea este mai rapidă decît exponențială și se încheie cu o singularitate în timp finit.",
                "incorrectExplanation": "Creșterea liniară a prețului este mai lentă decît cea exponențială, iar volatilitatea constantă sau randamentele normale privesc riscul, nu traiectoria de creștere. Semnătura LPPL este o rată de creștere în accelerare: logaritmul prețului crește mai rapid decît liniar."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Ordered phase of the Ising model",
                "text": "In the financial analogy of the Ising model, what does the regime $T < T_c$ (ordered phase) correspond to?",
                "options": [
                    "An efficient market with random trading",
                    "Strong herding: a bubble regime in which one side dominates",
                    "High-frequency trading with no directional bias",
                    "A bear market with falling prices"
                ],
                "correctExplanation": "Below $T_c$ the spins align: most agents take the same decision (buy or sell). In market terms this is herding, the mechanism behind bubble formation.",
                "incorrectExplanation": "Random, independent trading corresponds to the disordered phase $T > T_c$; high-frequency trading is a market microstructure topic, and alignment can occur on the buy side as well as the sell side. The ordered phase means strong herding."
            },
            "ro": {
                "title": "Faza ordonată a modelului Ising",
                "text": "În analogia financiară a modelului Ising, cărei situații îi corespunde regimul $T < T_c$ (faza ordonată)?",
                "options": [
                    "O piață eficientă, cu tranzacții aleatoare",
                    "Un comportament de turmă (herding) puternic: regim de bulă în care o tabără domină",
                    "Tranzacționare de înaltă frecvență fără direcție predominantă",
                    "O piață bear, cu prețuri în scădere"
                ],
                "correctExplanation": "Sub $T_c$ spinii se aliniază: majoritatea agenților iau aceeași decizie (cumpără sau vînd). În termeni de piață, acesta este comportamentul de turmă (herding), mecanismul din spatele formării bulelor.",
                "incorrectExplanation": "Tranzacțiile aleatoare și independente corespund fazei dezordonate $T > T_c$; tranzacționarea de înaltă frecvență ține de microstructura pieței, iar alinierea poate avea loc atît la cumpărare, cît și la vînzare. Faza ordonată înseamnă herding puternic."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Susceptibility at the critical point",
                "text": "Why does the divergence of the susceptibility $\\chi \\to \\infty$ at the critical point $T_c$ matter for financial markets?",
                "options": [
                    "The market becomes insensitive to news",
                    "The market is perfectly efficient",
                    "Even a small perturbation can trigger a market-wide cascade",
                    "Volatility drops to zero"
                ],
                "correctExplanation": "A diverging susceptibility means the aggregate response to a small external field becomes unboundedly large. In a market close to criticality, a minor piece of news can flip large clusters of traders at once.",
                "incorrectExplanation": "Near the critical point the market is maximally sensitive, not insensitive, it is fragile rather than efficient, and volatility rises rather than vanishes. The relevance is that small shocks can trigger large cascades."
            },
            "ro": {
                "title": "Susceptibilitatea în punctul critic",
                "text": "De ce contează pentru piețele financiare divergența susceptibilității, $\\chi \\to \\infty$, în punctul critic $T_c$?",
                "options": [
                    "Piața devine insensibilă la știri",
                    "Piața este perfect eficientă",
                    "Chiar și o perturbație mică poate declanșa o cascadă la nivelul întregii piețe",
                    "Volatilitatea scade la zero"
                ],
                "correctExplanation": "O susceptibilitate divergentă înseamnă că răspunsul agregat la un cîmp extern mic devine oricît de mare. Într-o piață apropiată de punctul critic, o știre minoră poate schimba simultan decizia unor grupuri mari de traderi.",
                "incorrectExplanation": "Lîngă punctul critic piața este maxim sensibilă, nu insensibilă, este fragilă, nu eficientă, iar volatilitatea crește, nu dispare. Relevanța constă în faptul că șocuri mici pot declanșa cascade mari."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Discrete scale invariance",
                "text": "What does discrete scale invariance (DSI) produce in the LPPL model?",
                "options": [
                    "Constant exponential growth",
                    "Log-periodic oscillations with a preferred scaling ratio $\\lambda$",
                    "Normally distributed returns",
                    "Zero correlation between returns"
                ],
                "correctExplanation": "DSI turns the power law into $(t_c - t)^m[1 + C\\cos(\\omega \\ln(t_c - t) + \\phi)]$, i.e. oscillations periodic in $\\ln(t_c - t)$ with preferred scaling ratio $\\lambda = e^{2\\pi/\\omega}$. Successive oscillation cycles shorten by the factor $\\lambda$ as $t \\to t_c$.",
                "incorrectExplanation": "Constant exponential growth has no oscillations, and the distribution or correlation of returns is not what DSI describes. DSI generates oscillations that are periodic in log-time, with ratio $\\lambda$ between successive cycles."
            },
            "ro": {
                "title": "Invarianța de scală discretă",
                "text": "Ce produce invarianța de scală discretă (DSI) în modelul LPPL?",
                "options": [
                    "O creștere exponențială constantă",
                    "Oscilații log-periodice cu un raport de scală preferat $\\lambda$",
                    "Randamente cu distribuție Normală",
                    "Corelație nulă între randamente"
                ],
                "correctExplanation": "DSI transformă legea de putere în $(t_c - t)^m[1 + C\\cos(\\omega \\ln(t_c - t) + \\phi)]$, adică oscilații periodice în $\\ln(t_c - t)$, cu raportul de scală preferat $\\lambda = e^{2\\pi/\\omega}$. Ciclurile succesive de oscilație se scurtează cu factorul $\\lambda$ cînd $t \\to t_c$.",
                "incorrectExplanation": "O creștere exponențială constantă nu are oscilații, iar distribuția sau corelația randamentelor nu sînt descrise de DSI. DSI generează oscilații periodice în timp logaritmic, cu raportul $\\lambda$ între ciclurile succesive."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The parameter B in the LPPL equation",
                "text": "Why is the condition $B < 0$ essential in the LPPL model of a bubble?",
                "options": [
                    "It ensures that the price falls over time",
                    "It ensures super-exponential price growth as $t \\to t_c$",
                    "It removes the log-periodic oscillations",
                    "It forces $t_c$ to lie in the past"
                ],
                "correctExplanation": "With $B < 0$ and $0 < m < 1$, the term $B(t_c - t)^m$ rises towards 0 as $t \\to t_c$, and its slope $-Bm(t_c - t)^{m-1}$ grows without bound, so the log-price rises at an accelerating rate.",
                "incorrectExplanation": "$B < 0$ produces a rising, not a falling, price path; the oscillations are governed by $C$ and $\\omega$, and $t_c$ is estimated, not constrained to the past. The sign of $B$ is what makes the growth super-exponential."
            },
            "ro": {
                "title": "Parametrul B din ecuația LPPL",
                "text": "De ce este esențială condiția $B < 0$ în modelul LPPL al unei bule?",
                "options": [
                    "Asigură scăderea prețului în timp",
                    "Asigură creșterea super-exponențială a prețului cînd $t \\to t_c$",
                    "Elimină oscilațiile log-periodice",
                    "Obligă $t_c$ să se afle în trecut"
                ],
                "correctExplanation": "Cu $B < 0$ și $0 < m < 1$, termenul $B(t_c - t)^m$ crește spre 0 cînd $t \\to t_c$, iar panta lui, $-Bm(t_c - t)^{m-1}$, crește nemărginit, deci logaritmul prețului crește într-un ritm accelerat.",
                "incorrectExplanation": "$B < 0$ produce o traiectorie crescătoare a prețului, nu una descrescătoare; oscilațiile sînt guvernate de $C$ și $\\omega$, iar $t_c$ se estimează, fără restricția de a se afla în trecut. Semnul lui $B$ face creșterea super-exponențială."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Partial linearisation",
                "text": "What advantage does partial linearisation (slaving the linear parameters) bring to LPPL estimation?",
                "options": [
                    "It removes all 7 parameters from the optimisation",
                    "It reduces the nonlinear search from 7 to 3 dimensions $(t_c, m, \\omega)$, with OLS for the rest",
                    "It guarantees a unique global optimum",
                    "It removes the need for market data"
                ],
                "correctExplanation": "For fixed $(t_c, m, \\omega)$, the parameters $(A, B, C_1, C_2)$ enter linearly and are obtained exactly by OLS. Only a 3-dimensional nonlinear search remains, which makes the fit far more stable.",
                "incorrectExplanation": "Three nonlinear parameters still have to be searched, the objective can still have several local minima, and the model is always fitted to market data. The gain is the reduction from 7 to 3 nonlinear dimensions."
            },
            "ro": {
                "title": "Liniarizarea parțială",
                "text": "Ce avantaj aduce liniarizarea parțială (slaving-ul parametrilor liniari) în estimarea LPPL?",
                "options": [
                    "Elimină toți cei 7 parametri din optimizare",
                    "Reduce căutarea neliniară de la 7 la 3 dimensiuni $(t_c, m, \\omega)$, restul parametrilor obținîndu-se prin OLS",
                    "Garantează un optim global unic",
                    "Elimină nevoia de date de piață"
                ],
                "correctExplanation": "Pentru $(t_c, m, \\omega)$ fixați, parametrii $(A, B, C_1, C_2)$ intră liniar și se obțin exact prin OLS. Rămîne doar o căutare neliniară în 3 dimensiuni, ceea ce face estimarea mult mai stabilă.",
                "incorrectExplanation": "Cei trei parametri neliniari trebuie în continuare căutați, funcția obiectiv poate avea în continuare mai multe minime locale, iar modelul se estimează întotdeauna pe date de piață. Cîștigul este reducerea de la 7 la 3 dimensiuni neliniare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Filter condition on m",
                "text": "Why is the filter condition $0.1 \\leq m \\leq 0.9$ imposed on a valid bubble signal?",
                "options": [
                    "It ensures that growth is slower than exponential",
                    "It ensures growth faster than exponential but not an almost instantaneous jump",
                    "It removes the oscillations from the model",
                    "It forces the price to revert to the mean"
                ],
                "correctExplanation": "With $0 < m < 1$ the log-price stays finite at $t_c$ while its growth rate diverges, which is super-exponential growth. For $m \\geq 1$ the growth rate no longer diverges, and as $m \\to 0$ the fitted curve degenerates into an almost instantaneous jump. The bounds 0.1 and 0.9 keep the fit away from both degenerate cases.",
                "incorrectExplanation": "Growth slower than exponential and mean reversion contradict the bubble hypothesis, and the oscillations depend on $\\omega$ and $C$, not on $m$. The condition on $m$ keeps the fitted growth super-exponential without becoming a near-jump."
            },
            "ro": {
                "title": "Condiția de filtrare pentru m",
                "text": "De ce se impune unui semnal valid de bulă condiția de filtrare $0{,}1 \\leq m \\leq 0{,}9$?",
                "options": [
                    "Asigură o creștere mai lentă decît cea exponențială",
                    "Asigură o creștere mai rapidă decît cea exponențială, dar nu un salt aproape instantaneu",
                    "Elimină oscilațiile din model",
                    "Obligă prețul să revină la medie"
                ],
                "correctExplanation": "Pentru $0 < m < 1$, logaritmul prețului rămîne finit în $t_c$, în timp ce rata lui de creștere diverge, adică o creștere super-exponențială. Pentru $m \\geq 1$ rata de creștere nu mai diverge, iar cînd $m \\to 0$ curba estimată degenerează într-un salt aproape instantaneu. Limitele 0,1 și 0,9 țin estimarea departe de ambele cazuri degenerate.",
                "incorrectExplanation": "Creșterea mai lentă decît cea exponențială și revenirea la medie contrazic ipoteza de bulă, iar oscilațiile depind de $\\omega$ și $C$, nu de $m$. Condiția asupra lui $m$ păstrează creșterea estimată super-exponențială, fără ca ea să devină un salt."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "LPPLS confidence indicator",
                "text": "How is the LPPLS confidence indicator (CI) constructed?",
                "options": [
                    "By computing daily volatility",
                    "By fitting LPPLS over many estimation windows ending at the current date and computing the fraction of fits that pass all filter conditions",
                    "By comparing the current price with a moving average",
                    "By applying the ADF test to returns"
                ],
                "correctExplanation": "For each date, LPPLS is fitted on many windows of different lengths that end at that date. The CI is the share of fits that satisfy all the filter conditions (on $m$, $\\omega$, $t_c$, the number of oscillations, damping and so on). A high CI means the bubble signature is robust to the choice of window.",
                "incorrectExplanation": "Volatility, moving-average rules and unit-root tests do not use the LPPLS fits at all. The CI measures how many window-specific LPPLS fits are valid."
            },
            "ro": {
                "title": "Indicatorul de încredere LPPLS",
                "text": "Cum se construiește indicatorul de încredere LPPLS (CI)?",
                "options": [
                    "Prin calculul volatilității zilnice",
                    "Prin estimarea LPPLS pe multe ferestre care se încheie la data curentă și calculul proporției de estimări care satisfac toate condițiile de filtrare",
                    "Prin compararea prețului curent cu o medie mobilă",
                    "Prin aplicarea testului ADF asupra randamentelor"
                ],
                "correctExplanation": "Pentru fiecare dată, LPPLS se estimează pe multe ferestre de lungimi diferite care se încheie la acea dată. CI este proporția estimărilor care satisfac toate condițiile de filtrare (asupra lui $m$, $\\omega$, $t_c$, numărului de oscilații, amortizării etc.). Un CI ridicat arată că semnătura de bulă nu depinde de alegerea ferestrei.",
                "incorrectExplanation": "Volatilitatea, regulile cu medii mobile și testele de rădăcină unitară nu folosesc deloc estimările LPPLS. CI măsoară cîte dintre estimările LPPLS pe ferestre diferite sînt valide."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Negative control",
                "text": "Why is a negative control (for example the COVID-19 crash of 2020) important when validating LPPL?",
                "options": [
                    "It shows that LPPL can predict any type of crash",
                    "It checks whether LPPL avoids false alarms before crashes caused by exogenous shocks",
                    "It shows that the market was efficient in 2020",
                    "It confirms that the COVID-19 crash was an endogenous bubble"
                ],
                "correctExplanation": "LPPL claims to detect endogenous bubbles. A crash triggered by an exogenous shock should not be preceded by a strong LPPL signature; if the method flags such episodes as often as genuine bubbles, its signals are not specific.",
                "incorrectExplanation": "LPPL is not meant to predict every crash, a negative control says nothing about market efficiency, and its purpose is to test an episode treated as exogenous, not to relabel it as a bubble. It checks the specificity of the method."
            },
            "ro": {
                "title": "Control negativ",
                "text": "De ce este important un control negativ (de exemplu crahul COVID-19 din 2020) în validarea LPPL?",
                "options": [
                    "Arată că LPPL poate prezice orice tip de crah",
                    "Verifică dacă LPPL evită alarmele false înaintea crahurilor provocate de șocuri exogene",
                    "Arată că piața era eficientă în 2020",
                    "Confirmă că crahul COVID-19 a fost o bulă endogenă"
                ],
                "correctExplanation": "LPPL își propune să detecteze bule endogene. Un crah declanșat de un șoc exogen nu ar trebui precedat de o semnătură LPPL puternică; dacă metoda semnalează astfel de episoade la fel de des ca bulele reale, semnalele ei nu sînt specifice.",
                "incorrectExplanation": "LPPL nu urmărește să prezică orice crah, un control negativ nu spune nimic despre eficiența pieței, iar scopul lui este testarea unui episod considerat exogen, nu reclasificarea lui drept bulă. Controlul negativ verifică specificitatea metodei."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Phase transition in the Ising model",
                "text": "What happens to the magnetisation $|M|$ at the critical temperature $T_c$ in the 2D Ising model?",
                "options": [
                    "$|M|$ stays at 1 (perfect order persists)",
                    "$|M|$ falls continuously to 0 (second-order phase transition)",
                    "$|M|$ jumps discontinuously from 1 to 0 (first-order transition)",
                    "$|M|$ oscillates between 0 and 1"
                ],
                "correctExplanation": "The 2D Ising model has a continuous (second-order) transition at $T_c = 2/\\ln(1+\\sqrt{2}) \\approx 2.269$ (with $J = k_B = 1$). The spontaneous magnetisation vanishes as $|M| \\sim (T_c - T)^{1/8}$ (Onsager, 1944; Yang, 1952).",
                "incorrectExplanation": "Order does not persist at $T_c$, there is no jump because the transition is continuous, not first-order, and the equilibrium magnetisation does not oscillate. $|M|$ falls continuously to zero."
            },
            "ro": {
                "title": "Tranziția de fază în modelul Ising",
                "text": "Ce se întîmplă cu magnetizarea $|M|$ la temperatura critică $T_c$ în modelul Ising bidimensional?",
                "options": [
                    "$|M|$ rămîne egală cu 1 (ordinea perfectă persistă)",
                    "$|M|$ scade continuu la 0 (tranziție de fază de ordinul doi)",
                    "$|M|$ sare discontinuu de la 1 la 0 (tranziție de ordinul întîi)",
                    "$|M|$ oscilează între 0 și 1"
                ],
                "correctExplanation": "Modelul Ising bidimensional are o tranziție continuă (de ordinul doi) la $T_c = 2/\\ln(1+\\sqrt{2}) \\approx 2{,}269$ (cu $J = k_B = 1$). Magnetizarea spontană se anulează după legea $|M| \\sim (T_c - T)^{1/8}$ (Onsager, 1944; Yang, 1952).",
                "incorrectExplanation": "Ordinea nu persistă la $T_c$, nu există un salt, deoarece tranziția este continuă, nu de ordinul întîi, iar magnetizarea de echilibru nu oscilează. $|M|$ scade continuu la zero."
            }
        }
    ]
};
