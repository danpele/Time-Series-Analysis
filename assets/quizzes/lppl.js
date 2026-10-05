// ============================================================
// Chapter 13 quiz bank: Speculative bubbles: LPPL models (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['lppl'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "Growth of a rational bubble",
                "text": "With no-arbitrage pricing and a discount rate $r$, how must a rational bubble component $B_t = P_t - F_t$ evolve in expectation?",
                "options": [
                    "$E_t[B_{t+1}] = (1+r)B_t$",
                    "$E_t[B_{t+1}] = B_t$",
                    "$E_t[B_{t+1}] = 0$",
                    "$E_t[B_{t+1}] = (1-r)B_t$"
                ],
                "correctExplanation": "The price equation $P_t = E_t[P_{t+1} + D_{t+1}]/(1+r)$ is solved by $F_t + B_t$ only if $E_t[B_{t+1}] = (1+r)B_t$: the bubble must grow at the rate $r$ in expectation, an explosive process.",
                "incorrectExplanation": "A constant, a vanishing or a shrinking expected bubble does not satisfy the pricing equation: an investor holds an overvalued asset only if the overvaluation is expected to grow at the rate $r$, i.e. $E_t[B_{t+1}] = (1+r)B_t$."
            },
            "ro": {
                "title": "Creșterea unei bule raționale",
                "text": "Cu evaluare fără arbitraj și rata de actualizare $r$, cum trebuie să evolueze în medie componenta de bulă rațională $B_t = P_t - F_t$?",
                "options": [
                    "$E_t[B_{t+1}] = (1+r)B_t$",
                    "$E_t[B_{t+1}] = B_t$",
                    "$E_t[B_{t+1}] = 0$",
                    "$E_t[B_{t+1}] = (1-r)B_t$"
                ],
                "correctExplanation": "Ecuația de preț $P_t = E_t[P_{t+1} + D_{t+1}]/(1+r)$ are soluția $F_t + B_t$ doar dacă $E_t[B_{t+1}] = (1+r)B_t$: bula trebuie să crească în medie cu rata $r$, un proces exploziv.",
                "incorrectExplanation": "O bulă constantă, care dispare sau care scade în medie nu verifică ecuația de preț: un investitor deține un activ supraevaluat doar dacă se așteaptă ca supraevaluarea să crească cu rata $r$, adică $E_t[B_{t+1}] = (1+r)B_t$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "A bubble that can burst",
                "text": "In a Blanchard–Watson bubble with $r = 2\\%$ and survival probability $\\pi = 0.98$ per period, by how much does the bubble grow in a period in which it survives?",
                "options": [
                    "By exactly 2%",
                    "By about 4.1%",
                    "By 98%",
                    "It does not grow; it only bursts"
                ],
                "correctExplanation": "While it survives, the bubble grows to $\\frac{1+r}{\\pi}B_t$: $1.02/0.98 - 1 \\approx 4.1\\%$. The extra growth above $r$ compensates for the risk of a burst, so that the expected growth is still $r$.",
                "incorrectExplanation": "A growth of exactly $r$ would make the expected growth lower than $r$ once bursts are counted; 98% confuses the probability with the growth; and a bubble that never grows cannot exist. The survival growth is $(1+r)/\\pi - 1 \\approx 4.1\\%$."
            },
            "ro": {
                "title": "O bulă care se poate sparge",
                "text": "Într-o bulă Blanchard–Watson cu $r = 2\\%$ și probabilitatea de supraviețuire $\\pi = 0{,}98$ pe perioadă, cu cît crește bula într-o perioadă în care supraviețuiește?",
                "options": [
                    "Cu exact 2%",
                    "Cu aproximativ 4,1%",
                    "Cu 98%",
                    "Nu crește; doar se sparge"
                ],
                "correctExplanation": "Cît supraviețuiește, bula crește la $\\frac{1+r}{\\pi}B_t$: $1{,}02/0{,}98 - 1 \\approx 4{,}1\\%$. Creșterea peste $r$ compensează riscul spargerii, astfel încît creșterea medie rămîne $r$.",
                "incorrectExplanation": "O creștere de exact $r$ ar face creșterea medie mai mică decît $r$ după ce se iau în calcul spargerile; 98% confundă probabilitatea cu creșterea; iar o bulă care nu crește niciodată nu poate exista. Creșterea la supraviețuire este $(1+r)/\\pi - 1 \\approx 4{,}1\\%$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Right-tailed unit-root test",
                "text": "In the regression $\\Delta y_t = a + \\delta y_{t-1} + \\varepsilon_t$ on a log price, which hypotheses does a test for an explosive bubble use?",
                "options": [
                    "$H_0$: $\\delta < 0$ against $H_1$: $\\delta = 0$",
                    "$H_0$: $\\delta = 0$ against $H_1$: $\\delta < 0$",
                    "$H_0$: $\\delta = 0$ against $H_1$: $\\delta > 0$",
                    "$H_0$: $\\delta > 0$ against $H_1$: $\\delta = 0$"
                ],
                "correctExplanation": "The null is a unit root ($\\delta = 0$, $\\rho = 1$) and the alternative an explosive root ($\\delta > 0$, $\\rho > 1$): we reject for large positive $t$-statistics, in the right tail.",
                "incorrectExplanation": "The left-tailed test of Chapter 3 ($H_1$: $\\delta < 0$) looks for stationarity, not for bubbles, and a hypothesis of explosiveness is not the null. Bubble tests use $H_0$: $\\delta = 0$ against $H_1$: $\\delta > 0$."
            },
            "ro": {
                "title": "Testul de rădăcină unitară la dreapta",
                "text": "În regresia $\\Delta y_t = a + \\delta y_{t-1} + \\varepsilon_t$ pe un preț logaritmic, ce ipoteze folosește un test pentru o bulă explozivă?",
                "options": [
                    "$H_0$: $\\delta < 0$ față de $H_1$: $\\delta = 0$",
                    "$H_0$: $\\delta = 0$ față de $H_1$: $\\delta < 0$",
                    "$H_0$: $\\delta = 0$ față de $H_1$: $\\delta > 0$",
                    "$H_0$: $\\delta > 0$ față de $H_1$: $\\delta = 0$"
                ],
                "correctExplanation": "Ipoteza nulă este o rădăcină unitară ($\\delta = 0$, $\\rho = 1$), iar alternativa o rădăcină explozivă ($\\delta > 0$, $\\rho > 1$): respingem pentru statistici $t$ pozitive mari, în coada din dreapta.",
                "incorrectExplanation": "Testul la stînga din Capitolul 3 ($H_1$: $\\delta < 0$) caută staționaritatea, nu bulele, iar explozivitatea nu este ipoteza nulă. Testele de bule folosesc $H_0$: $\\delta = 0$ față de $H_1$: $\\delta > 0$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "One test on the whole sample",
                "text": "Why does a single right-tailed ADF test on the whole Nasdaq 100 sample 1990–2004 fail to find the dot-com bubble?",
                "options": [
                    "Because weekly data cannot contain bubbles",
                    "Because the Nasdaq 100 never had an explosive phase",
                    "Because the ADF test needs at least 50 years of data",
                    "Because the crash after the bubble looks like mean reversion and masks the explosive phase"
                ],
                "correctExplanation": "A bubble occupies only part of the sample; the collapse that follows pulls the full-sample estimate of $\\delta$ down. This is why SADF and GSADF compute the statistic on many windows.",
                "incorrectExplanation": "The frequency of the data and the length of the sample are not the issue, and the windowed tests do find an explosive phase in the Nasdaq 100. The crash after the bubble masks it in a single full-sample test."
            },
            "ro": {
                "title": "Un singur test pe tot eșantionul",
                "text": "De ce un singur test ADF la dreapta pe tot eșantionul Nasdaq 100 1990–2004 nu găsește bula dot-com?",
                "options": [
                    "Pentru că datele săptămînale nu pot conține bule",
                    "Pentru că Nasdaq 100 nu a avut niciodată o fază explozivă",
                    "Pentru că testul ADF are nevoie de cel puțin 50 de ani de date",
                    "Pentru că prăbușirea de după bulă seamănă cu revenirea la medie și ascunde faza explozivă"
                ],
                "correctExplanation": "O bulă ocupă doar o parte din eșantion; prăbușirea care urmează trage în jos estimarea lui $\\delta$ pe tot eșantionul. De aceea SADF și GSADF calculează statistica pe multe ferestre.",
                "incorrectExplanation": "Frecvența datelor și lungimea eșantionului nu sînt problema, iar testele pe ferestre găsesc o fază explozivă în Nasdaq 100. Prăbușirea de după bulă o ascunde într-un test unic pe tot eșantionul."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "SADF and GSADF",
                "text": "What distinguishes the GSADF test of Phillips, Shi and Yu from the SADF test of Phillips, Wu and Yu?",
                "options": [
                    "GSADF lets both the start and the end of the window move; SADF fixes the start at the first observation",
                    "GSADF uses the left tail; SADF uses the right tail",
                    "GSADF needs no critical values",
                    "GSADF works only on returns, SADF only on prices"
                ],
                "correctExplanation": "SADF takes the supremum over forward-expanding windows that all start at 0; GSADF also moves the start point, so it can detect several bubbles in one sample.",
                "incorrectExplanation": "Both tests are right-tailed, both need simulated critical values and both are applied to log prices. The difference is the set of windows: GSADF also moves the start point."
            },
            "ro": {
                "title": "SADF și GSADF",
                "text": "Ce deosebește testul GSADF al lui Phillips, Shi și Yu de testul SADF al lui Phillips, Wu și Yu?",
                "options": [
                    "GSADF lasă să se miște și începutul, și sfîrșitul ferestrei; SADF fixează începutul la prima observație",
                    "GSADF folosește coada din stînga; SADF coada din dreapta",
                    "GSADF nu are nevoie de valori critice",
                    "GSADF se aplică doar randamentelor, SADF doar prețurilor"
                ],
                "correctExplanation": "SADF ia supremumul pe ferestre care se extind și încep toate la 0; GSADF mută și punctul de început, deci poate detecta mai multe bule într-un eșantion.",
                "incorrectExplanation": "Ambele teste sînt la dreapta, ambele au nevoie de valori critice simulate și ambele se aplică prețurilor logaritmice. Diferența este mulțimea ferestrelor: GSADF mută și punctul de început."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "The smallest window",
                "text": "With the rule $r_0 = 0.01 + 1.8/\\sqrt{T}$ of Phillips, Shi and Yu and $T = 400$ weekly observations, what is the smallest window?",
                "options": [
                    "4 observations",
                    "40 observations",
                    "180 observations",
                    "400 observations"
                ],
                "correctExplanation": "$r_0 = 0.01 + 1.8/20 = 0.10$, so the smallest window is $0.10 \\times 400 = 40$ observations.",
                "incorrectExplanation": "The value 4 forgets to multiply by $T$, 180 uses $1.8$ without dividing by $\\sqrt{T}$, and 400 is the whole sample. The rule gives $r_0 = 0.10$, i.e. 40 observations."
            },
            "ro": {
                "title": "Fereastra minimă",
                "text": "Cu regula $r_0 = 0{,}01 + 1{,}8/\\sqrt{T}$ a lui Phillips, Shi și Yu și $T = 400$ de observații săptămînale, cît este fereastra minimă?",
                "options": [
                    "4 observații",
                    "40 de observații",
                    "180 de observații",
                    "400 de observații"
                ],
                "correctExplanation": "$r_0 = 0{,}01 + 1{,}8/20 = 0{,}10$, deci fereastra minimă are $0{,}10 \\times 400 = 40$ de observații.",
                "incorrectExplanation": "Valoarea 4 uită înmulțirea cu $T$, 180 folosește $1{,}8$ fără împărțirea la $\\sqrt{T}$, iar 400 este tot eșantionul. Regula dă $r_0 = 0{,}10$, adică 40 de observații."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Date-stamping with BSADF",
                "text": "How does the BSADF procedure date the start of an explosive episode?",
                "options": [
                    "At the date of the highest price in the sample",
                    "At the first date with a negative return",
                    "At the first date where the BSADF statistic exceeds its critical value, if it stays above for a minimum duration",
                    "At the date of the crash"
                ],
                "correctExplanation": "BSADF is computed for each end date with data up to that date only; an episode starts when it crosses its critical value and lasts at least about $\\log T$ observations.",
                "incorrectExplanation": "The highest price and the crash are known only afterwards, and one negative return says nothing about explosiveness. The rule is the first crossing of the critical value, with a minimum duration."
            },
            "ro": {
                "title": "Datarea cu BSADF",
                "text": "Cum datează procedura BSADF începutul unui episod exploziv?",
                "options": [
                    "La data celui mai mare preț din eșantion",
                    "La prima dată cu un randament negativ",
                    "La prima dată la care statistica BSADF depășește valoarea critică, dacă rămîne peste ea o durată minimă",
                    "La data crahului"
                ],
                "correctExplanation": "BSADF se calculează pentru fiecare dată de sfîrșit, doar cu datele pînă la acea dată; un episod începe cînd statistica depășește valoarea critică și durează cel puțin aproximativ $\\log T$ observații.",
                "incorrectExplanation": "Cel mai mare preț și crahul se cunosc doar după aceea, iar un randament negativ nu spune nimic despre explozivitate. Regula este prima depășire a valorii critice, cu o durată minimă."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Explosive price, bubble?",
                "text": "A GSADF test finds an explosive episode in a stock index. What else is needed before calling it a bubble?",
                "options": [
                    "Nothing: an explosive price is always a bubble",
                    "A left-tailed ADF test on the same window",
                    "A larger sample of the same index",
                    "Evidence that the fundamentals (for example dividends) were not explosive, e.g. a test on the price–dividend ratio"
                ],
                "correctExplanation": "A price can be explosive because the fundamentals are explosive. Testing the price–dividend ratio, as Phillips, Shi and Yu do for the S&P 500, separates the two.",
                "incorrectExplanation": "An explosive price is not enough, a left-tailed test answers another question and more data of the same price do not remove the problem. The fundamentals must be checked, e.g. with the price–dividend ratio."
            },
            "ro": {
                "title": "Preț exploziv, deci bulă?",
                "text": "Un test GSADF găsește un episod exploziv într-un indice bursier. Ce mai este necesar înainte de a-l numi bulă?",
                "options": [
                    "Nimic: un preț exploziv este întotdeauna o bulă",
                    "Un test ADF la stînga pe aceeași fereastră",
                    "Un eșantion mai lung al aceluiași indice",
                    "Dovezi că fundamentele (de exemplu dividendele) nu au fost explozive, de exemplu un test pe raportul preț–dividend"
                ],
                "correctExplanation": "Un preț poate fi exploziv pentru că fundamentele sînt explozive. Testul pe raportul preț–dividend, ca la Phillips, Shi și Yu pentru S&P 500, separă cele două cazuri.",
                "incorrectExplanation": "Un preț exploziv nu este suficient, un test la stînga răspunde la altă întrebare, iar mai multe date ale aceluiași preț nu elimină problema. Trebuie verificate fundamentele, de exemplu prin raportul preț–dividend."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "A flag during a collapse",
                "text": "BSADF flags the BET between September 2008 and April 2009, during the collapse. How do we read this?",
                "options": [
                    "An accelerating fall also gives $\\hat\\delta > 0$: the flag marks the collapse, not a bubble",
                    "The BET had a speculative bubble during the 2008 crisis",
                    "The test has a programming error",
                    "The critical values were computed for the wrong sample size"
                ],
                "correctExplanation": "In a fall that accelerates, lower prices go with larger falls, so the slope $\\delta$ on $y_{t-1}$ is positive. Dates must always be read against the price chart.",
                "incorrectExplanation": "Nothing in the price chart shows a bubble in 2008–2009, and the flag is not a coding or critical-value problem. An accelerating fall also produces $\\hat\\delta > 0$."
            },
            "ro": {
                "title": "Un semnal în timpul unei prăbușiri",
                "text": "BSADF semnalează BET între septembrie 2008 și aprilie 2009, în timpul prăbușirii. Cum citim acest semnal?",
                "options": [
                    "O scădere care se accelerează dă tot $\\hat\\delta > 0$: semnalul marchează prăbușirea, nu o bulă",
                    "BET a avut o bulă speculativă în timpul crizei din 2008",
                    "Testul are o eroare de programare",
                    "Valorile critice au fost calculate pentru o altă mărime a eșantionului"
                ],
                "correctExplanation": "Într-o scădere care se accelerează, prețurile mai mici merg cu scăderi mai mari, deci panta $\\delta$ pe $y_{t-1}$ este pozitivă. Datele trebuie citite întotdeauna alături de graficul prețului.",
                "incorrectExplanation": "Graficul prețului nu arată nicio bulă în 2008–2009, iar semnalul nu provine dintr-o eroare de cod sau de valori critice. O scădere care se accelerează produce și ea $\\hat\\delta > 0$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Super-exponential growth",
                "text": "What does super-exponential growth of a price mean?",
                "options": [
                    "The price grows at a constant rate",
                    "The growth rate of the price itself increases over time",
                    "The price grows linearly",
                    "The volatility of the price is constant"
                ],
                "correctExplanation": "Exponential growth has a constant growth rate (a straight line on a log scale). Super-exponential growth has a rising growth rate: the log price bends upward, as in $A + B(t_c - t)^m$ with $B < 0$, $0 < m < 1$.",
                "incorrectExplanation": "A constant growth rate is ordinary exponential growth and linear growth is slower still; volatility is a different property. Super-exponential means an increasing growth rate."
            },
            "ro": {
                "title": "Creșterea superexponențială",
                "text": "Ce înseamnă creșterea superexponențială a unui preț?",
                "options": [
                    "Prețul crește cu un ritm constant",
                    "Chiar ritmul de creștere al prețului crește în timp",
                    "Prețul crește liniar",
                    "Volatilitatea prețului este constantă"
                ],
                "correctExplanation": "Creșterea exponențială are un ritm constant (o dreaptă pe scară logaritmică). Creșterea superexponențială are un ritm în creștere: prețul logaritmic se îndoaie în sus, ca în $A + B(t_c - t)^m$ cu $B < 0$, $0 < m < 1$.",
                "incorrectExplanation": "Un ritm constant înseamnă creștere exponențială obișnuită, iar creșterea liniară este și mai lentă; volatilitatea este o altă proprietate. Superexponențial înseamnă un ritm de creștere în creștere."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The exponent m",
                "text": "Why must the LPPL exponent satisfy $0 < m < 1$?",
                "options": [
                    "So that the price falls before $t_c$",
                    "So that the oscillations disappear",
                    "So that the growth rate diverges at $t_c$ while the log price stays finite",
                    "So that the model becomes linear"
                ],
                "correctExplanation": "With $\\ln P = A + B(t_c - t)^m$, the growth rate is $-Bm(t_c - t)^{m-1}$: it diverges for $m < 1$, while $\\ln P(t_c) = A$ is finite for $m > 0$.",
                "incorrectExplanation": "The price rises, not falls, when $B < 0$; the oscillations are governed by $C$ and $\\omega$; and the model stays nonlinear in $m$. The range $0 < m < 1$ gives a diverging growth rate with a finite price."
            },
            "ro": {
                "title": "Exponentul m",
                "text": "De ce trebuie ca exponentul LPPL să verifice $0 < m < 1$?",
                "options": [
                    "Pentru ca prețul să scadă înainte de $t_c$",
                    "Pentru ca oscilațiile să dispară",
                    "Pentru ca ritmul de creștere să tindă la infinit la $t_c$, în timp ce prețul logaritmic rămîne finit",
                    "Pentru ca modelul să devină liniar"
                ],
                "correctExplanation": "Cu $\\ln P = A + B(t_c - t)^m$, ritmul de creștere este $-Bm(t_c - t)^{m-1}$: tinde la infinit pentru $m < 1$, în timp ce $\\ln P(t_c) = A$ este finit pentru $m > 0$.",
                "incorrectExplanation": "Prețul crește, nu scade, cînd $B < 0$; oscilațiile depind de $C$ și $\\omega$; iar modelul rămîne neliniar în $m$. Intervalul $0 < m < 1$ dă un ritm de creștere care tinde la infinit, cu preț finit."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The sign of B",
                "text": "Why does a positive (rising) bubble require $B < 0$ in the LPPL equation?",
                "options": [
                    "Because $B$ is the phase of the oscillations",
                    "Because $B$ must cancel $A$",
                    "Because $B < 0$ makes the oscillations faster",
                    "Because $(t_c - t)^m$ falls as $t \\to t_c$, so $B(t_c - t)^m$ increases only if $B < 0$"
                ],
                "correctExplanation": "The term $(t_c - t)^m$ shrinks towards 0 as $t$ approaches $t_c$; multiplied by a negative $B$ it rises towards 0, so the log price rises towards $A$.",
                "incorrectExplanation": "$B$ is not the phase, it does not have to cancel $A$, and the speed of the oscillations is set by $\\omega$. $B < 0$ is what makes the power-law term rise towards $t_c$."
            },
            "ro": {
                "title": "Semnul lui B",
                "text": "De ce o bulă pozitivă (în creștere) cere $B < 0$ în ecuația LPPL?",
                "options": [
                    "Pentru că $B$ este faza oscilațiilor",
                    "Pentru că $B$ trebuie să-l anuleze pe $A$",
                    "Pentru că $B < 0$ face oscilațiile mai rapide",
                    "Pentru că $(t_c - t)^m$ scade cînd $t \\to t_c$, deci $B(t_c - t)^m$ crește doar dacă $B < 0$"
                ],
                "correctExplanation": "Termenul $(t_c - t)^m$ scade spre 0 cînd $t$ se apropie de $t_c$; înmulțit cu un $B$ negativ, el crește spre 0, deci prețul logaritmic crește spre $A$.",
                "incorrectExplanation": "$B$ nu este faza, nu trebuie să-l anuleze pe $A$, iar viteza oscilațiilor este dată de $\\omega$. $B < 0$ face ca termenul de tip lege de putere să crească spre $t_c$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The scaling ratio",
                "text": "An LPPL fit gives $\\omega = 8$. By what factor does the time left to $t_c$ shrink from one oscillation peak to the next?",
                "options": [
                    "$\\lambda = e^{2\\pi/8} \\approx 2.19$",
                    "$\\lambda = 8$",
                    "$\\lambda = 2\\pi \\approx 6.28$",
                    "$\\lambda = 8/2\\pi \\approx 1.27$"
                ],
                "correctExplanation": "Successive peaks satisfy $(t_c - t_n)/(t_c - t_{n+1}) = e^{2\\pi/\\omega}$; with $\\omega = 8$ this is about 2.19.",
                "incorrectExplanation": "The factor is not $\\omega$ itself, nor $2\\pi$, nor their ratio: the oscillation is periodic in $\\ln(t_c - t)$ with period $2\\pi/\\omega$, so the factor is $e^{2\\pi/\\omega} \\approx 2.19$."
            },
            "ro": {
                "title": "Raportul de scală",
                "text": "O ajustare LPPL dă $\\omega = 8$. Cu ce factor se micșorează timpul rămas pînă la $t_c$ de la un maxim al oscilației la următorul?",
                "options": [
                    "$\\lambda = e^{2\\pi/8} \\approx 2{,}19$",
                    "$\\lambda = 8$",
                    "$\\lambda = 2\\pi \\approx 6{,}28$",
                    "$\\lambda = 8/2\\pi \\approx 1{,}27$"
                ],
                "correctExplanation": "Maximele succesive verifică $(t_c - t_n)/(t_c - t_{n+1}) = e^{2\\pi/\\omega}$; pentru $\\omega = 8$ factorul este aproximativ 2,19.",
                "incorrectExplanation": "Factorul nu este $\\omega$, nici $2\\pi$, nici raportul lor: oscilația este periodică în $\\ln(t_c - t)$ cu perioada $2\\pi/\\omega$, deci factorul este $e^{2\\pi/\\omega} \\approx 2{,}19$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Log-periodicity",
                "text": "In the term $\\cos(\\omega \\ln(t_c - t) - \\phi)$, in which variable are the oscillations periodic?",
                "options": [
                    "In calendar time $t$",
                    "In $\\ln(t_c - t)$",
                    "In the price level",
                    "In the volatility"
                ],
                "correctExplanation": "The cosine repeats each time $\\ln(t_c - t)$ changes by $2\\pi/\\omega$. In calendar time the cycles therefore become shorter and shorter as $t_c$ approaches.",
                "incorrectExplanation": "The cycles are not regular in calendar time, and they do not depend on the price level or on volatility. They are periodic in $\\ln(t_c - t)$, hence the name log-periodic."
            },
            "ro": {
                "title": "Log-periodicitatea",
                "text": "În termenul $\\cos(\\omega \\ln(t_c - t) - \\phi)$, în ce variabilă sînt periodice oscilațiile?",
                "options": [
                    "În timpul calendaristic $t$",
                    "În $\\ln(t_c - t)$",
                    "În nivelul prețului",
                    "În volatilitate"
                ],
                "correctExplanation": "Cosinusul se repetă de fiecare dată cînd $\\ln(t_c - t)$ se schimbă cu $2\\pi/\\omega$. În timp calendaristic ciclurile devin deci tot mai scurte pe măsură ce se apropie $t_c$.",
                "incorrectExplanation": "Ciclurile nu sînt regulate în timp calendaristic și nu depind de nivelul prețului sau de volatilitate. Ele sînt periodice în $\\ln(t_c - t)$, de unde numele de log-periodice."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The two-step method",
                "text": "What is the key step of the Filimonov–Sornette calibration of LPPL?",
                "options": [
                    "Estimate all seven parameters by one OLS regression",
                    "Fix $t_c$ at the highest observed price",
                    "For given $(t_c, m, \\omega)$, estimate $A$, $B$, $C_1$, $C_2$ by OLS, then search only over $(t_c, m, \\omega)$",
                    "Replace the cosine by a straight line"
                ],
                "correctExplanation": "Writing $C\\cos(\\omega\\ln\\tau - \\phi) = C_1\\cos(\\omega\\ln\\tau) + C_2\\sin(\\omega\\ln\\tau)$ makes four parameters linear; they are solved by OLS for each $(t_c, m, \\omega)$, which reduces the search to three nonlinear parameters.",
                "incorrectExplanation": "Three parameters enter nonlinearly, so one OLS cannot estimate all seven; fixing $t_c$ at the maximum uses the future; and the cosine is not removed. The trick is to concentrate out the four linear parameters."
            },
            "ro": {
                "title": "Metoda în doi pași",
                "text": "Care este pasul-cheie al calibrării Filimonov–Sornette a modelului LPPL?",
                "options": [
                    "Estimarea tuturor celor șapte parametri printr-o singură regresie OLS",
                    "Fixarea lui $t_c$ la cel mai mare preț observat",
                    "Pentru $(t_c, m, \\omega)$ dați, estimarea lui $A$, $B$, $C_1$, $C_2$ prin OLS, apoi căutarea doar după $(t_c, m, \\omega)$",
                    "Înlocuirea cosinusului cu o dreaptă"
                ],
                "correctExplanation": "Scrierea $C\\cos(\\omega\\ln\\tau - \\phi) = C_1\\cos(\\omega\\ln\\tau) + C_2\\sin(\\omega\\ln\\tau)$ face liniari patru parametri; ei se obțin prin OLS pentru fiecare $(t_c, m, \\omega)$, ceea ce reduce căutarea la trei parametri neliniari.",
                "incorrectExplanation": "Trei parametri intră neliniar, deci o singură regresie OLS nu îi poate estima pe toți șapte; fixarea lui $t_c$ la maximum folosește viitorul; iar cosinusul nu se elimină. Ideea este concentrarea celor patru parametri liniari."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Amplitude of the oscillations",
                "text": "After the linear step, $\\hat C_1 = 0.03$ and $\\hat C_2 = 0.04$. What is the amplitude $\\hat C$?",
                "options": [
                    "0.07",
                    "0.01",
                    "0.0012",
                    "0.05"
                ],
                "correctExplanation": "$C = \\sqrt{C_1^2 + C_2^2} = \\sqrt{0.0009 + 0.0016} = 0.05$.",
                "incorrectExplanation": "Adding, subtracting or multiplying the two coefficients does not give the amplitude of $C_1\\cos(x) + C_2\\sin(x)$. The amplitude is $\\sqrt{C_1^2 + C_2^2} = 0.05$."
            },
            "ro": {
                "title": "Amplitudinea oscilațiilor",
                "text": "După pasul liniar, $\\hat C_1 = 0{,}03$ și $\\hat C_2 = 0{,}04$. Cît este amplitudinea $\\hat C$?",
                "options": [
                    "0,07",
                    "0,01",
                    "0,0012",
                    "0,05"
                ],
                "correctExplanation": "$C = \\sqrt{C_1^2 + C_2^2} = \\sqrt{0{,}0009 + 0{,}0016} = 0{,}05$.",
                "incorrectExplanation": "Suma, diferența sau produsul celor doi coeficienți nu dau amplitudinea lui $C_1\\cos(x) + C_2\\sin(x)$. Amplitudinea este $\\sqrt{C_1^2 + C_2^2} = 0{,}05$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Crash hazard and growth",
                "text": "In the JLS model, the no-arbitrage condition gives $\\mu(t) = \\kappa h(t)$. What does it imply?",
                "options": [
                    "The higher the crash hazard, the faster the price must rise to compensate investors",
                    "The price falls when the crash hazard rises",
                    "The hazard rate is constant",
                    "The crash happens exactly at $t_c$"
                ],
                "correctExplanation": "Before the crash, the expected return must be zero net of the expected crash loss: the drift $\\mu(t)$ equals the size of the crash $\\kappa$ times its hazard rate $h(t)$. Risk and growth rise together.",
                "incorrectExplanation": "A higher hazard does not lower the price before the crash, the hazard is not constant (it grows towards $t_c$), and the crash is only most likely near $t_c$, not certain. The drift rises with the hazard."
            },
            "ro": {
                "title": "Riscul de crah și creșterea",
                "text": "În modelul JLS, condiția de lipsă a arbitrajului dă $\\mu(t) = \\kappa h(t)$. Ce implică aceasta?",
                "options": [
                    "Cu cît riscul de crah este mai mare, cu atît prețul trebuie să crească mai repede pentru a-i compensa pe investitori",
                    "Prețul scade cînd riscul de crah crește",
                    "Rata de hazard este constantă",
                    "Crahul are loc exact la $t_c$"
                ],
                "correctExplanation": "Înainte de crah, randamentul așteptat trebuie să fie zero după scăderea pierderii așteptate din crah: driftul $\\mu(t)$ este egal cu mărimea crahului $\\kappa$ înmulțită cu rata de hazard $h(t)$. Riscul și creșterea cresc împreună.",
                "incorrectExplanation": "Un hazard mai mare nu scade prețul înainte de crah, hazardul nu este constant (crește spre $t_c$), iar crahul este doar cel mai probabil în apropierea lui $t_c$, nu sigur. Driftul crește odată cu hazardul."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "A fit at the boundary",
                "text": "An LPPL fit gives $\\hat m = 1.00$, at the edge of the search space. Is it a qualified bubble fit?",
                "options": [
                    "Yes, because the fit error is small",
                    "No: the filter requires $m \\le 0.99$; $m = 1$ means no acceleration of growth",
                    "Yes, if $\\omega$ is between 2 and 25",
                    "Only if $B > 0$"
                ],
                "correctExplanation": "A parameter at the edge of the search space signals that the data do not support the model. With $m = 1$ the power-law term is linear in $t$: there is no super-exponential growth.",
                "incorrectExplanation": "A small error does not rescue a boundary estimate, the other conditions do not compensate, and $B > 0$ would describe a falling price. The fit fails the condition $0.01 \\le m \\le 0.99$."
            },
            "ro": {
                "title": "O ajustare la margine",
                "text": "O ajustare LPPL dă $\\hat m = 1{,}00$, la marginea spațiului de căutare. Este o ajustare calificată de bulă?",
                "options": [
                    "Da, pentru că eroarea ajustării este mică",
                    "Nu: filtrul cere $m \\le 0{,}99$; $m = 1$ înseamnă că ritmul de creștere nu se accelerează",
                    "Da, dacă $\\omega$ este între 2 și 25",
                    "Doar dacă $B > 0$"
                ],
                "correctExplanation": "Un parametru la marginea spațiului de căutare arată că datele nu susțin modelul. Cu $m = 1$ termenul de tip lege de putere este liniar în $t$: nu există creștere superexponențială.",
                "incorrectExplanation": "O eroare mică nu salvează o estimare de la margine, celelalte condiții nu compensează, iar $B > 0$ ar descrie un preț în scădere. Ajustarea nu trece de condiția $0{,}01 \\le m \\le 0{,}99$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "The damping condition",
                "text": "What does the damping condition $m|B|/(\\omega|C|) \\ge 1$ guarantee?",
                "options": [
                    "That the fit has at least 2.5 oscillations",
                    "That $t_c$ lies after the last observation",
                    "That the implied crash hazard rate stays positive",
                    "That the residuals are stationary"
                ],
                "correctExplanation": "The hazard rate is proportional to the growth rate of the log price; if the oscillations are too strong relative to the trend, it would become negative, which is impossible for a probability rate.",
                "incorrectExplanation": "The number of oscillations, the position of $t_c$ and the stationarity of the residuals are separate conditions of the filter. Damping keeps the hazard rate positive."
            },
            "ro": {
                "title": "Condiția de amortizare",
                "text": "Ce garantează condiția de amortizare $m|B|/(\\omega|C|) \\ge 1$?",
                "options": [
                    "Că ajustarea are cel puțin 2,5 oscilații",
                    "Că $t_c$ se află după ultima observație",
                    "Că rata de hazard a crahului rămîne pozitivă",
                    "Că reziduurile sînt staționare"
                ],
                "correctExplanation": "Rata de hazard este proporțională cu ritmul de creștere al prețului logaritmic; dacă oscilațiile sînt prea puternice față de trend, ea ar deveni negativă, ceea ce este imposibil pentru o rată de probabilitate.",
                "incorrectExplanation": "Numărul de oscilații, poziția lui $t_c$ și staționaritatea reziduurilor sînt condiții separate ale filtrului. Amortizarea menține rata de hazard pozitivă."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The LPPLS confidence indicator",
                "text": "How is the LPPLS confidence indicator at a date $t_2$ defined?",
                "options": [
                    "The value of $\\hat m$ in the longest window",
                    "The probability that a crash happens tomorrow",
                    "The $R^2$ of one LPPL fit",
                    "The share of the windows ending at $t_2$ whose LPPL fit passes the filter conditions"
                ],
                "correctExplanation": "Many windows $[t_2 - L, t_2]$ are fitted; the indicator is the fraction of qualified fits. It uses only data up to $t_2$.",
                "incorrectExplanation": "It is neither one parameter, nor a crash probability, nor the fit quality of one window. It is the share of qualified fits over many windows ending at $t_2$."
            },
            "ro": {
                "title": "Indicatorul de încredere LPPLS",
                "text": "Cum se definește indicatorul de încredere LPPLS la o dată $t_2$?",
                "options": [
                    "Valoarea lui $\\hat m$ în cea mai lungă fereastră",
                    "Probabilitatea ca un crah să aibă loc mîine",
                    "$R^2$ al unei singure ajustări LPPL",
                    "Ponderea ferestrelor care se încheie la $t_2$ și au o ajustare LPPL care trece de condițiile de filtrare"
                ],
                "correctExplanation": "Se ajustează multe ferestre $[t_2 - L, t_2]$; indicatorul este fracțiunea ajustărilor calificate. Folosește doar datele pînă la $t_2$.",
                "incorrectExplanation": "Nu este nici un parametru, nici o probabilitate de crah, nici calitatea ajustării unei singure ferestre. Este ponderea ajustărilor calificate pe multe ferestre care se încheie la $t_2$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Why many windows",
                "text": "Why is LPPL fitted on many windows instead of one?",
                "options": [
                    "Because the start of a bubble is unknown in real time and $\\hat t_c$ depends on the window; the spread over windows measures the uncertainty",
                    "Because one window always gives $m = 1$",
                    "Because OLS needs at least 29 regressions",
                    "Because the Lomb test requires it"
                ],
                "correctExplanation": "Choosing the start at the low of the run-up uses information available only after the bubble. Many windows avoid this choice and show how much $\\hat t_c$ varies.",
                "incorrectExplanation": "One window does not always fail, and neither OLS nor the Lomb test requires several windows. The reason is the unknown start of the bubble and the instability of $\\hat t_c$."
            },
            "ro": {
                "title": "De ce multe ferestre",
                "text": "De ce se ajustează LPPL pe multe ferestre în loc de una singură?",
                "options": [
                    "Pentru că începutul unei bule nu este cunoscut în timp real, iar $\\hat t_c$ depinde de fereastră; dispersia pe ferestre măsoară incertitudinea",
                    "Pentru că o singură fereastră dă mereu $m = 1$",
                    "Pentru că OLS are nevoie de cel puțin 29 de regresii",
                    "Pentru că testul Lomb o cere"
                ],
                "correctExplanation": "Alegerea începutului la minimul creșterii folosește informație disponibilă doar după bulă. Multe ferestre evită această alegere și arată cît variază $\\hat t_c$.",
                "incorrectExplanation": "O singură fereastră nu eșuează mereu, iar nici OLS, nici testul Lomb nu cer mai multe ferestre. Motivul este începutul necunoscut al bulei și instabilitatea lui $\\hat t_c$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Look-ahead bias",
                "text": "Which of the following is an example of look-ahead bias in a study of crash prediction?",
                "options": [
                    "Using only data up to $t_2$ for the fit at $t_2$",
                    "Choosing, after the crash, the window whose $\\hat t_c$ was closest to the actual peak",
                    "Fixing the alarm threshold before looking at the results",
                    "Reporting all the false alarms"
                ],
                "correctExplanation": "Selecting the window, the start date or the filter after seeing the outcome uses information that was not available at $t_2$; the resulting “prediction” is a description of the past.",
                "incorrectExplanation": "Using only past data, fixing the threshold in advance and reporting false alarms are the remedies, not the bias. Choosing the best window after the crash is look-ahead bias."
            },
            "ro": {
                "title": "Look-ahead bias",
                "text": "Care dintre următoarele este un exemplu de look-ahead bias într-un studiu de prognoză a crahurilor?",
                "options": [
                    "Folosirea doar a datelor pînă la $t_2$ pentru ajustarea de la $t_2$",
                    "Alegerea, după crah, a ferestrei al cărei $\\hat t_c$ a fost cel mai aproape de maximul real",
                    "Fixarea pragului de alarmă înainte de a vedea rezultatele",
                    "Raportarea tuturor alarmelor false"
                ],
                "correctExplanation": "Alegerea ferestrei, a datei de început sau a filtrului după ce se cunoaște rezultatul folosește informație care nu era disponibilă la $t_2$; „predicția” obținută este o descriere a trecutului.",
                "incorrectExplanation": "Folosirea doar a datelor trecute, fixarea dinainte a pragului și raportarea alarmelor false sînt remediile, nu eroarea. Alegerea celei mai bune ferestre după crah este look-ahead bias."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Judging an alarm",
                "text": "An indicator gave an alarm before 8 of the last 10 crashes. What must you also know to judge it?",
                "options": [
                    "Nothing: 80% is a good hit rate",
                    "Only the date of the last crash",
                    "The share of alarm dates followed by a crash, compared with the unconditional frequency of crashes (the base rate)",
                    "The value of $\\omega$ in the last fit"
                ],
                "correctExplanation": "If alarms are on almost all the time, catching 8 crashes is easy. An alarm is useful only if $P(\\text{crash} \\mid \\text{alarm})$ is clearly above $P(\\text{crash})$, counting all the false alarms.",
                "incorrectExplanation": "A hit rate on crashes alone ignores the false alarms, and one date or one parameter says nothing about usefulness. The comparison with the base rate is what matters."
            },
            "ro": {
                "title": "Judecarea unei alarme",
                "text": "Un indicator a dat o alarmă înainte de 8 dintre ultimele 10 crahuri. Ce mai trebuie să știți pentru a-l judeca?",
                "options": [
                    "Nimic: 80% este o rată de reușită bună",
                    "Doar data ultimului crah",
                    "Ponderea datelor cu alarmă urmate de un crah, comparată cu frecvența necondiționată a crahurilor (frecvența de bază)",
                    "Valoarea lui $\\omega$ din ultima ajustare"
                ],
                "correctExplanation": "Dacă alarmele apar aproape tot timpul, este ușor să prinzi 8 crahuri. O alarmă este utilă doar dacă $P(\\text{crah} \\mid \\text{alarmă})$ este clar peste $P(\\text{crah})$, numărînd toate alarmele false.",
                "incorrectExplanation": "O rată de reușită calculată doar pe crahuri ignoră alarmele false, iar o dată sau un parametru nu spun nimic despre utilitate. Contează comparația cu frecvența de bază."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "The S&P 500 evaluation",
                "text": "On the S&P 500 since 1993, a fall of 20% within six months follows about 7% of all dates but none of the alarm dates of the indicator. What is the right conclusion?",
                "options": [
                    "The indicator predicts crashes perfectly",
                    "The S&P 500 never had a crash",
                    "The threshold of 0.2 is too low and should be lowered further",
                    "The alarms do not raise the short-term crash probability: the indicator describes bubble-like regimes, it does not time crashes"
                ],
                "correctExplanation": "The alarms came in calm, steady uptrends; the large falls came later or without an alarm. The indicator is a description of the regime, not a timing device.",
                "incorrectExplanation": "A hit rate of zero is the opposite of perfect prediction, the S&P 500 had several falls of 20% or more, and lowering the threshold adds more false alarms. The alarms do not time crashes."
            },
            "ro": {
                "title": "Evaluarea pe S&P 500",
                "text": "Pe S&P 500 din 1993, o scădere de 20% în șase luni urmează după aproximativ 7% dintre toate datele, dar după niciuna dintre datele cu alarmă ale indicatorului. Care este concluzia corectă?",
                "options": [
                    "Indicatorul prezice perfect crahurile",
                    "S&P 500 nu a avut niciodată un crah",
                    "Pragul de 0,2 este prea mic și trebuie coborît și mai mult",
                    "Alarmele nu cresc probabilitatea unui crah pe termen scurt: indicatorul descrie regimuri asemănătoare bulelor, nu datează crahurile"
                ],
                "correctExplanation": "Alarmele au apărut în creșteri calme și constante; scăderile mari au venit mai tîrziu sau fără alarmă. Indicatorul descrie regimul, nu este un instrument de datare.",
                "incorrectExplanation": "O rată de reușită zero este opusul unei predicții perfecte, S&P 500 a avut mai multe scăderi de 20% sau mai mult, iar coborîrea pragului adaugă alte alarme false. Alarmele nu datează crahurile."
            }
        }
    ]
};
