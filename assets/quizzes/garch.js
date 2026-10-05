// ============================================================
// Chapter 5 quiz bank: Conditional volatility: ARCH and GARCH (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['garch'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 0,
            "en": {
                "title": "Volatility clustering",
                "text": "What does \"volatility clustering\" mean?",
                "options": [
                    "Large changes tend to be followed by large changes, of either sign, and small changes by small changes",
                    "Returns are positively autocorrelated, so a rise is followed by a rise",
                    "Volatility is the same in every year of the sample",
                    "Returns follow the Normal distribution"
                ],
                "correctExplanation": "Mandelbrot (1963): \"large changes tend to be followed by large changes, of either sign\". The size of the moves is persistent, not their direction; this is what ARCH and GARCH models describe.",
                "incorrectExplanation": "Clustering concerns the size of the returns (their squares or absolute values), not their sign: daily returns are almost uncorrelated. It also rules out a constant volatility and does not require Normality."
            },
            "ro": {
                "title": "Volatility clustering",
                "text": "Ce înseamnă fenomenul de „volatility clustering”?",
                "options": [
                    "Variațiile mari tind să fie urmate de variații mari, de orice semn, iar cele mici de variații mici",
                    "Randamentele sînt pozitiv autocorelate, deci o creștere este urmată de o creștere",
                    "Volatilitatea este aceeași în fiecare an al eșantionului",
                    "Randamentele au distribuția Normală"
                ],
                "correctExplanation": "Mandelbrot (1963): „variațiile mari tind să fie urmate de variații mari, de orice semn”. Mărimea variațiilor este persistentă, nu direcția lor; acest fenomen este descris de modelele ARCH și GARCH.",
                "incorrectExplanation": "Fenomenul privește mărimea randamentelor (pătratele sau valorile absolute), nu semnul lor: randamentele zilnice sînt aproape necorelate. El exclude o volatilitate constantă și nu cere distribuția Normală."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "ACF of returns and squared returns",
                "text": "For daily S&P 500 returns since 2000, what do the sample ACFs of $r_t$ and $r_t^2$ typically show?",
                "options": [
                    "Both ACFs are close to zero at every lag",
                    "The ACF of $r_t$ is close to zero, while the ACF of $r_t^2$ is positive and decays slowly",
                    "The ACF of $r_t$ decays slowly, while the ACF of $r_t^2$ is close to zero",
                    "Both ACFs are large and negative at lag 1"
                ],
                "correctExplanation": "Returns are close to white noise (Chapter 1), so the direction of tomorrow's move is almost unpredictable; their squares have a positive, slowly decaying ACF (about 0.3 at lag 1 and still positive at lag 50): the size of the move is predictable.",
                "incorrectExplanation": "The sign of daily returns is almost unpredictable, so the ACF of $r_t$ stays near zero; the squares are strongly and persistently autocorrelated. Uncorrelated does not mean independent."
            },
            "ro": {
                "title": "ACF a randamentelor și a pătratelor lor",
                "text": "Pentru randamentele zilnice S&P 500 din 2000, ce arată de obicei ACF de selecție a lui $r_t$ și a lui $r_t^2$?",
                "options": [
                    "Ambele ACF sînt apropiate de zero la toate decalajele",
                    "ACF a lui $r_t$ este apropiată de zero, iar ACF a lui $r_t^2$ este pozitivă și scade lent",
                    "ACF a lui $r_t$ scade lent, iar ACF a lui $r_t^2$ este apropiată de zero",
                    "Ambele ACF sînt mari și negative la decalajul 1"
                ],
                "correctExplanation": "Randamentele sînt apropiate de zgomotul alb (Capitolul 1), deci direcția variației de mîine este aproape imprevizibilă; pătratele lor au o ACF pozitivă care scade lent (circa 0,3 la decalajul 1 și încă pozitivă la decalajul 50): mărimea variației este previzibilă.",
                "incorrectExplanation": "Semnul randamentelor zilnice este aproape imprevizibil, deci ACF a lui $r_t$ rămîne aproape de zero; pătratele sînt autocorelate puternic și persistent. Necorelat nu înseamnă independent."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "ARCH-LM test",
                "text": "An ARCH-LM regression of $\\hat\\varepsilon_t^2$ on a constant and 5 lags uses $n = 500$ observations and gives $R^2 = 0.05$. What do you conclude at 5% ($\\chi^2_{0.95}(5) = 11.07$)?",
                "options": [
                    "$\\mathrm{LM} = 0.05$: no ARCH effects",
                    "$\\mathrm{LM} = 2.5$: no ARCH effects",
                    "$\\mathrm{LM} = 25$: reject $H_0$, there are ARCH effects",
                    "The test cannot be computed without the coefficients"
                ],
                "correctExplanation": "$\\mathrm{LM} = nR^2 = 500 \\times 0.05 = 25 > 11.07$: the squared residuals are predictable from their own past, so the conditional variance is not constant.",
                "incorrectExplanation": "The statistic is $nR^2$, compared with $\\chi^2(q)$; here $500 \\times 0.05 = 25$, well above 11.07. The coefficients are not needed, only $R^2$ and $n$."
            },
            "ro": {
                "title": "Testul ARCH-LM",
                "text": "O regresie ARCH-LM a lui $\\hat\\varepsilon_t^2$ pe o constantă și 5 decalaje folosește $n = 500$ de observații și dă $R^2 = 0,05$. Ce concluzie trageți la 5% ($\\chi^2_{0,95}(5) = 11,07$)?",
                "options": [
                    "$\\mathrm{LM} = 0,05$: nu există efecte ARCH",
                    "$\\mathrm{LM} = 2,5$: nu există efecte ARCH",
                    "$\\mathrm{LM} = 25$: respingem $H_0$, există efecte ARCH",
                    "Testul nu se poate calcula fără coeficienți"
                ],
                "correctExplanation": "$\\mathrm{LM} = nR^2 = 500 \\times 0,05 = 25 > 11,07$: pătratele reziduurilor sînt previzibile din propriul trecut, deci varianța condiționată nu este constantă.",
                "incorrectExplanation": "Statistica este $nR^2$, comparată cu $\\chi^2(q)$; aici $500 \\times 0,05 = 25$, mult peste 11,07. Coeficienții nu sînt necesari, doar $R^2$ și $n$."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Conditional and unconditional variance",
                "text": "Which statement about $\\sigma_t^2 = \\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1})$ is correct?",
                "options": [
                    "It is the same number for every day of the sample",
                    "It can only be computed after day $t$ has ended",
                    "It equals $r_t^2$ exactly",
                    "It is known at $t-1$, changes over time, and its average is the unconditional variance"
                ],
                "correctExplanation": "The conditional variance uses only the information up to $t-1$, so it is known one day ahead and moves with the news; by the law of total variance, its average (with a constant mean) is the unconditional variance.",
                "incorrectExplanation": "The unconditional variance is one number; the conditional variance moves from day to day and is known one day in advance. $r_t^2$ is only a noisy proxy of $\\sigma_t^2$, since $E_{t-1}[r_t^2] = \\sigma_t^2$ when the mean is zero."
            },
            "ro": {
                "title": "Varianța condiționată și varianța necondiționată",
                "text": "Care afirmație despre $\\sigma_t^2 = \\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1})$ este corectă?",
                "options": [
                    "Este același număr pentru fiecare zi a eșantionului",
                    "Poate fi calculată doar după încheierea zilei $t$",
                    "Este exact egală cu $r_t^2$",
                    "Este cunoscută la $t-1$, se schimbă în timp, iar media ei este varianța necondiționată"
                ],
                "correctExplanation": "Varianța condiționată folosește doar informația pînă la $t-1$, deci este cunoscută cu o zi înainte și se schimbă odată cu știrile; din legea varianței totale, media ei (cu medie constantă) este varianța necondiționată.",
                "incorrectExplanation": "Varianța necondiționată este un singur număr; varianța condiționată se schimbă de la o zi la alta și este cunoscută cu o zi înainte. $r_t^2$ este doar o aproximare zgomotoasă a lui $\\sigma_t^2$, deoarece $E_{t-1}[r_t^2] = \\sigma_t^2$ cînd media este zero."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "ARCH(1)",
                "text": "An ARCH(1) model has $\\omega = 0.2$ and $\\alpha = 0.6$. What is its unconditional variance?",
                "options": [
                    "$0.5$",
                    "$0.2$",
                    "$0.12$",
                    "$0.8$"
                ],
                "correctExplanation": "Taking expectations in $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2$ gives $\\bar\\sigma^2 = \\omega + \\alpha\\bar\\sigma^2$, so $\\bar\\sigma^2 = \\omega/(1 - \\alpha) = 0.2/0.4 = 0.5$.",
                "incorrectExplanation": "The unconditional variance solves $\\bar\\sigma^2 = \\omega + \\alpha\\bar\\sigma^2$, which gives $\\omega/(1 - \\alpha) = 0.5$; $\\omega$ alone is the variance after a zero shock, not the average variance."
            },
            "ro": {
                "title": "ARCH(1)",
                "text": "Un model ARCH(1) are $\\omega = 0,2$ și $\\alpha = 0,6$. Cît este varianța lui necondiționată?",
                "options": [
                    "$0,5$",
                    "$0,2$",
                    "$0,12$",
                    "$0,8$"
                ],
                "correctExplanation": "Aplicînd media în $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2$ obținem $\\bar\\sigma^2 = \\omega + \\alpha\\bar\\sigma^2$, deci $\\bar\\sigma^2 = \\omega/(1 - \\alpha) = 0,2/0,4 = 0,5$.",
                "incorrectExplanation": "Varianța necondiționată rezolvă ecuația $\\bar\\sigma^2 = \\omega + \\alpha\\bar\\sigma^2$, de unde $\\omega/(1 - \\alpha) = 0,5$; $\\omega$ singur este varianța după un șoc nul, nu varianța medie."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "GARCH and ARCH",
                "text": "What is the main advantage of GARCH(1,1) over ARCH($q$)?",
                "options": [
                    "It removes the need to estimate any parameter",
                    "With three parameters it reproduces a long memory of the variance that ARCH needs many lags for",
                    "It makes the returns Normally distributed",
                    "It models the conditional mean instead of the variance"
                ],
                "correctExplanation": "GARCH(1,1) is an ARCH($\\infty$) with geometrically decaying weights $\\alpha\\beta^j$: one extra parameter replaces a long ARCH($q$). On the S&P 500, GARCH(1,1) beats ARCH(10) by AIC and BIC.",
                "incorrectExplanation": "GARCH still has parameters, still models the variance, and does not make returns Normal; its advantage is parsimony: $\\beta\\sigma_{t-1}^2$ carries the effect of all past shocks."
            },
            "ro": {
                "title": "GARCH și ARCH",
                "text": "Care este principalul avantaj al GARCH(1,1) față de ARCH($q$)?",
                "options": [
                    "Nu mai este nevoie de estimarea vreunui parametru",
                    "Cu trei parametri reproduce o memorie lungă a varianței pentru care ARCH are nevoie de multe decalaje",
                    "Face ca randamentele să aibă distribuția Normală",
                    "Modelează media condiționată în locul varianței"
                ],
                "correctExplanation": "GARCH(1,1) este un ARCH($\\infty$) cu ponderi care scad geometric, $\\alpha\\beta^j$: un singur parametru în plus înlocuiește un ARCH($q$) lung. Pe S&P 500, GARCH(1,1) este mai bun decît ARCH(10) după AIC și BIC.",
                "incorrectExplanation": "GARCH are tot parametri, modelează tot varianța și nu face randamentele Normale; avantajul lui este parcimonia: $\\beta\\sigma_{t-1}^2$ transmite efectul tuturor șocurilor trecute."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "GARCH parameters",
                "text": "In $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, what does $\\alpha$ measure?",
                "options": [
                    "The long-run level of the variance",
                    "The memory of the variance",
                    "The reaction of the variance to yesterday's shock",
                    "The mean return"
                ],
                "correctExplanation": "$\\alpha$ multiplies the squared shock of yesterday: the larger $\\alpha$, the larger the jump of the variance after news. $\\beta$ is the memory and $\\omega$ fixes, together with $\\alpha + \\beta$, the long-run level.",
                "incorrectExplanation": "The memory of the variance is $\\beta$; the long-run level is $\\omega/(1 - \\alpha - \\beta)$; the mean return is $\\mu$. $\\alpha$ is the reaction to the latest squared shock."
            },
            "ro": {
                "title": "Parametrii GARCH",
                "text": "În $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, ce măsoară $\\alpha$?",
                "options": [
                    "Nivelul de lungă durată al varianței",
                    "Memoria varianței",
                    "Reacția varianței la șocul de ieri",
                    "Randamentul mediu"
                ],
                "correctExplanation": "$\\alpha$ înmulțește pătratul șocului de ieri: cu cît $\\alpha$ este mai mare, cu atît varianța sare mai mult după o știre. $\\beta$ este memoria, iar $\\omega$ fixează, împreună cu $\\alpha + \\beta$, nivelul de lungă durată.",
                "incorrectExplanation": "Memoria varianței este $\\beta$; nivelul de lungă durată este $\\omega/(1 - \\alpha - \\beta)$; randamentul mediu este $\\mu$. $\\alpha$ este reacția la ultimul șoc la pătrat."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Stationarity condition",
                "text": "With $\\omega > 0$, $\\alpha \\ge 0$ and $\\beta \\ge 0$, when is a GARCH(1,1) covariance stationary?",
                "options": [
                    "When $\\beta < \\alpha$",
                    "When $\\omega < 1$",
                    "When $\\alpha + \\beta > 1$",
                    "When $\\alpha + \\beta < 1$"
                ],
                "correctExplanation": "GARCH(1,1) is an ARMA(1,1) for $\\varepsilon_t^2$ with AR coefficient $\\alpha + \\beta$; the variance has a finite mean $\\omega/(1 - \\alpha - \\beta)$ only if $\\alpha + \\beta < 1$.",
                "incorrectExplanation": "The condition concerns the persistence $\\alpha + \\beta$, the AR coefficient of $\\varepsilon_t^2$; $\\omega$ only scales the level, and $\\alpha + \\beta \\ge 1$ gives an IGARCH or an explosive variance."
            },
            "ro": {
                "title": "Condiția de staționaritate",
                "text": "Cu $\\omega > 0$, $\\alpha \\ge 0$ și $\\beta \\ge 0$, cînd este un GARCH(1,1) staționar în covarianță?",
                "options": [
                    "Cînd $\\beta < \\alpha$",
                    "Cînd $\\omega < 1$",
                    "Cînd $\\alpha + \\beta > 1$",
                    "Cînd $\\alpha + \\beta < 1$"
                ],
                "correctExplanation": "GARCH(1,1) este un ARMA(1,1) pentru $\\varepsilon_t^2$, cu coeficientul AR $\\alpha + \\beta$; varianța are o medie finită, $\\omega/(1 - \\alpha - \\beta)$, doar dacă $\\alpha + \\beta < 1$.",
                "incorrectExplanation": "Condiția privește persistența $\\alpha + \\beta$, coeficientul AR al lui $\\varepsilon_t^2$; $\\omega$ doar scalează nivelul, iar $\\alpha + \\beta \\ge 1$ dă un IGARCH sau o varianță explozivă."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Long-run variance",
                "text": "A GARCH(1,1) for daily returns in % has $\\omega = 0.03$, $\\alpha = 0.07$, $\\beta = 0.90$. What is its long-run variance?",
                "options": [
                    "$1.0$",
                    "$0.03$",
                    "$0.3$",
                    "$3.0$"
                ],
                "correctExplanation": "$\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta) = 0.03/0.03 = 1.0$, a daily volatility of 1%, about 15.9% per year with 252 trading days.",
                "incorrectExplanation": "The long-run variance divides $\\omega$ by $1 - \\alpha - \\beta = 0.03$; dividing by $1 - \\alpha$ or ignoring $\\beta$ gives wrong values."
            },
            "ro": {
                "title": "Varianța de lungă durată",
                "text": "Un GARCH(1,1) pentru randamente zilnice în % are $\\omega = 0,03$, $\\alpha = 0,07$, $\\beta = 0,90$. Cît este varianța lui de lungă durată?",
                "options": [
                    "$1,0$",
                    "$0,03$",
                    "$0,3$",
                    "$3,0$"
                ],
                "correctExplanation": "$\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta) = 0,03/0,03 = 1,0$, o volatilitate zilnică de 1%, circa 15,9% pe an, cu 252 de zile de tranzacționare.",
                "incorrectExplanation": "Varianța de lungă durată se obține împărțind $\\omega$ la $1 - \\alpha - \\beta = 0,03$; împărțirea la $1 - \\alpha$ sau ignorarea lui $\\beta$ dau valori greșite."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Half-life",
                "text": "A GARCH(1,1) has $\\alpha + \\beta = 0.95$. After how many days has half of a variance shock disappeared?",
                "options": [
                    "About 2 days",
                    "About 13.5 days",
                    "About 50 days",
                    "Never"
                ],
                "correctExplanation": "The deviation from the long-run variance shrinks by the factor $\\alpha + \\beta$ each day, so $h_{1/2} = \\ln 0.5/\\ln 0.95 = 13.5$ days.",
                "incorrectExplanation": "The half-life is $\\ln 0.5/\\ln(\\alpha + \\beta)$; with 0.95 it is 13.5 days. A shock never halves only when $\\alpha + \\beta = 1$ (IGARCH)."
            },
            "ro": {
                "title": "Timpul de înjumătățire",
                "text": "Un GARCH(1,1) are $\\alpha + \\beta = 0,95$. După cîte zile a dispărut jumătate dintr-un șoc de varianță?",
                "options": [
                    "Circa 2 zile",
                    "Circa 13,5 zile",
                    "Circa 50 de zile",
                    "Niciodată"
                ],
                "correctExplanation": "Abaterea de la varianța de lungă durată se micșorează cu factorul $\\alpha + \\beta$ în fiecare zi, deci $h_{1/2} = \\ln 0,5/\\ln 0,95 = 13,5$ zile.",
                "incorrectExplanation": "Timpul de înjumătățire este $\\ln 0,5/\\ln(\\alpha + \\beta)$; pentru 0,95 obținem 13,5 zile. Un șoc nu se înjumătățește niciodată doar cînd $\\alpha + \\beta = 1$ (IGARCH)."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "GARCH as an ARMA model",
                "text": "Writing $\\varepsilon_t^2 = \\sigma_t^2 + v_t$, which model does GARCH(1,1) imply for $\\varepsilon_t^2$?",
                "options": [
                    "A white noise",
                    "An MA(1) with coefficient $\\alpha$",
                    "An ARMA(1,1) with AR coefficient $\\alpha + \\beta$ and MA coefficient $-\\beta$",
                    "A random walk"
                ],
                "correctExplanation": "Substituting $\\sigma_t^2 = \\varepsilon_t^2 - v_t$ gives $\\varepsilon_t^2 = \\omega + (\\alpha + \\beta)\\varepsilon_{t-1}^2 + v_t - \\beta v_{t-1}$: an ARMA(1,1) (Chapter 2), whose ACF decays like $(\\alpha + \\beta)^k$.",
                "incorrectExplanation": "The squared shocks are autocorrelated, so they are not white noise; the AR part is $\\alpha + \\beta$ and the MA part is $-\\beta$. A random walk would need $\\alpha + \\beta = 1$ and no MA term."
            },
            "ro": {
                "title": "GARCH ca model ARMA",
                "text": "Scriind $\\varepsilon_t^2 = \\sigma_t^2 + v_t$, ce model implică GARCH(1,1) pentru $\\varepsilon_t^2$?",
                "options": [
                    "Un zgomot alb",
                    "Un MA(1) cu coeficientul $\\alpha$",
                    "Un ARMA(1,1) cu coeficientul AR $\\alpha + \\beta$ și coeficientul MA $-\\beta$",
                    "Un mers aleator"
                ],
                "correctExplanation": "Înlocuind $\\sigma_t^2 = \\varepsilon_t^2 - v_t$ obținem $\\varepsilon_t^2 = \\omega + (\\alpha + \\beta)\\varepsilon_{t-1}^2 + v_t - \\beta v_{t-1}$: un ARMA(1,1) (Capitolul 2), a cărui ACF scade ca $(\\alpha + \\beta)^k$.",
                "incorrectExplanation": "Pătratele șocurilor sînt autocorelate, deci nu sînt zgomot alb; partea AR este $\\alpha + \\beta$, iar partea MA este $-\\beta$. Un mers aleator ar cere $\\alpha + \\beta = 1$ și niciun termen MA."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "EWMA",
                "text": "Which GARCH(1,1) is the EWMA (RiskMetrics) estimator $\\sigma_t^2 = 0.94\\,\\sigma_{t-1}^2 + 0.06\\,r_{t-1}^2$?",
                "options": [
                    "$\\omega = 0.06$, $\\alpha = 0.94$, $\\beta = 0$",
                    "$\\omega = 0.94$, $\\alpha = 0$, $\\beta = 0.06$",
                    "A stationary GARCH with $\\alpha + \\beta = 0.94$",
                    "$\\omega = 0$, $\\alpha = 0.06$, $\\beta = 0.94$: an IGARCH"
                ],
                "correctExplanation": "EWMA has no constant and $\\alpha + \\beta = 1$: an IGARCH. It is simple exponential smoothing (Chapter 0) applied to the squared returns.",
                "incorrectExplanation": "The weight 0.06 is on yesterday's squared return ($\\alpha$) and 0.94 on yesterday's variance ($\\beta$); there is no $\\omega$, and the persistence is exactly 1."
            },
            "ro": {
                "title": "EWMA",
                "text": "Ce GARCH(1,1) este estimatorul EWMA (RiskMetrics) $\\sigma_t^2 = 0,94\\,\\sigma_{t-1}^2 + 0,06\\,r_{t-1}^2$?",
                "options": [
                    "$\\omega = 0,06$, $\\alpha = 0,94$, $\\beta = 0$",
                    "$\\omega = 0,94$, $\\alpha = 0$, $\\beta = 0,06$",
                    "Un GARCH staționar cu $\\alpha + \\beta = 0,94$",
                    "$\\omega = 0$, $\\alpha = 0,06$, $\\beta = 0,94$: un IGARCH"
                ],
                "correctExplanation": "EWMA nu are constantă și are $\\alpha + \\beta = 1$: un IGARCH. Este netezirea exponențială simplă (Capitolul 0) aplicată pătratelor randamentelor.",
                "incorrectExplanation": "Ponderea 0,06 este pe pătratul randamentului de ieri ($\\alpha$), iar 0,94 pe varianța de ieri ($\\beta$); nu există $\\omega$, iar persistența este exact 1."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "IGARCH",
                "text": "What is true for an IGARCH model ($\\alpha + \\beta = 1$)?",
                "options": [
                    "Variance shocks never die out: there is no half-life and no finite long-run variance",
                    "The variance returns to its long-run level within a week",
                    "The returns become Normal",
                    "The model cannot be estimated"
                ],
                "correctExplanation": "With $\\alpha + \\beta = 1$ the deviation from any level is never reduced, so forecasts for every horizon equal tomorrow's variance. EUR/RON and Bitcoin are close to IGARCH.",
                "incorrectExplanation": "Mean reversion needs $\\alpha + \\beta < 1$; IGARCH can be estimated (it is found on the boundary) and says nothing about Normality."
            },
            "ro": {
                "title": "IGARCH",
                "text": "Ce este adevărat pentru un model IGARCH ($\\alpha + \\beta = 1$)?",
                "options": [
                    "Șocurile varianței nu se sting niciodată: nu există timp de înjumătățire și nici varianță de lungă durată finită",
                    "Varianța revine la nivelul de lungă durată într-o săptămînă",
                    "Randamentele devin Normale",
                    "Modelul nu poate fi estimat"
                ],
                "correctExplanation": "Cu $\\alpha + \\beta = 1$, abaterea de la orice nivel nu se reduce niciodată, deci prognozele pentru orice orizont sînt egale cu varianța de mîine. EUR/RON și Bitcoin sînt apropiate de IGARCH.",
                "incorrectExplanation": "Revenirea la medie cere $\\alpha + \\beta < 1$; IGARCH poate fi estimat (estimarea ajunge pe frontieră) și nu spune nimic despre distribuția Normală."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Estimating GARCH models",
                "text": "How are the parameters of a GARCH model estimated?",
                "options": [
                    "By OLS of $r_t^2$ on $\\sigma_{t-1}^2$",
                    "By maximising the conditional log-likelihood numerically, with $\\sigma_t^2$ computed recursively",
                    "By the method of moments from the sample variance only",
                    "By reading them from the ACF of $r_t$"
                ],
                "correctExplanation": "$\\sigma_t^2$ is not observed, so OLS is impossible; for each candidate $\\theta$ the variances are computed recursively and $\\ell(\\theta) = -\\frac12\\sum[\\ln 2\\pi + \\ln\\sigma_t^2 + (r_t - \\mu)^2/\\sigma_t^2]$ is maximised numerically.",
                "incorrectExplanation": "The regressor $\\sigma_{t-1}^2$ is not observed, the sample variance identifies only the long-run level, and the ACF of $r_t$ is close to zero: maximum likelihood is the standard method."
            },
            "ro": {
                "title": "Estimarea modelelor GARCH",
                "text": "Cum se estimează parametrii unui model GARCH?",
                "options": [
                    "Prin OLS a lui $r_t^2$ pe $\\sigma_{t-1}^2$",
                    "Prin maximizarea numerică a log-verosimilității condiționate, cu $\\sigma_t^2$ calculat recursiv",
                    "Prin metoda momentelor, doar din varianța de selecție",
                    "Prin citirea lor din ACF a lui $r_t$"
                ],
                "correctExplanation": "$\\sigma_t^2$ nu este observat, deci OLS este imposibil; pentru fiecare $\\theta$ candidat varianțele se calculează recursiv, iar $\\ell(\\theta) = -\\frac12\\sum[\\ln 2\\pi + \\ln\\sigma_t^2 + (r_t - \\mu)^2/\\sigma_t^2]$ se maximizează numeric.",
                "incorrectExplanation": "Regresorul $\\sigma_{t-1}^2$ nu este observat, varianța de selecție identifică doar nivelul de lungă durată, iar ACF a lui $r_t$ este aproape zero: verosimilitatea maximă este metoda standard."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Robust standard errors",
                "text": "Why do we report robust (Bollerslev–Wooldridge) standard errors after a Gaussian GARCH fit?",
                "options": [
                    "They are always smaller than the classic ones",
                    "They make the estimates unbiased",
                    "They stay valid when the innovations are not Normal, while the classic ones then overstate the precision",
                    "They are needed only for ARCH(1)"
                ],
                "correctExplanation": "Maximising the Normal likelihood when $z_t$ is heavy-tailed is quasi-maximum likelihood: the estimates are still consistent, but only the \"sandwich\" standard errors are valid; on the S&P 500 they are about 1.4 times the classic ones.",
                "incorrectExplanation": "Robust standard errors do not change the estimates and are usually larger, not smaller; they matter for every GARCH model fitted with a likelihood that may be misspecified."
            },
            "ro": {
                "title": "Erori standard robuste",
                "text": "De ce raportăm erori standard robuste (Bollerslev–Wooldridge) după o estimare GARCH Gaussiană?",
                "options": [
                    "Sînt întotdeauna mai mici decît cele clasice",
                    "Fac estimările nedeplasate",
                    "Rămîn valide cînd inovațiile nu sînt Normale, în timp ce cele clasice supraestimează atunci precizia",
                    "Sînt necesare doar pentru ARCH(1)"
                ],
                "correctExplanation": "Maximizarea verosimilității Normale cînd $z_t$ are cozi groase este cvasi-verosimilitate maximă: estimările rămîn consistente, dar doar erorile standard de tip „sandwich” sînt valide; pe S&P 500 ele sînt de circa 1,4 ori mai mari decît cele clasice.",
                "incorrectExplanation": "Erorile standard robuste nu schimbă estimările și sînt de obicei mai mari, nu mai mici; ele contează pentru orice model GARCH estimat cu o verosimilitate posibil greșit specificată."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Student-t innovations",
                "text": "Why are Student-t innovations used in GARCH models of daily returns?",
                "options": [
                    "To make the variance constant",
                    "To remove the autocorrelation of $r_t$",
                    "Because the t distribution is symmetric and the Normal is not",
                    "Because the standardised residuals of a Normal GARCH still have kurtosis well above 3"
                ],
                "correctExplanation": "GARCH explains part of the kurtosis of returns, but the standardised residuals of the S&P 500 Normal GARCH still have kurtosis about 4.7; the standardised t with $\\nu \\approx 6$ captures these remaining heavy tails, which matter for VaR.",
                "incorrectExplanation": "Both distributions are symmetric; the innovation distribution does not change the variance dynamics or the mean autocorrelation. Its job is the tails of $z_t$."
            },
            "ro": {
                "title": "Inovații Student-t",
                "text": "De ce se folosesc inovații Student-t în modelele GARCH pentru randamente zilnice?",
                "options": [
                    "Pentru ca varianța să fie constantă",
                    "Pentru a elimina autocorelația lui $r_t$",
                    "Pentru că distribuția t este simetrică, iar cea Normală nu",
                    "Pentru că reziduurile standardizate ale unui GARCH Normal au încă un coeficient de boltire mult peste 3"
                ],
                "correctExplanation": "GARCH explică o parte din boltirea randamentelor, dar reziduurile standardizate ale GARCH Normal pe S&P 500 au încă boltire de circa 4,7; distribuția t standardizată cu $\\nu \\approx 6$ surprinde aceste cozi groase rămase, importante pentru VaR.",
                "incorrectExplanation": "Ambele distribuții sînt simetrice; distribuția inovațiilor nu schimbă dinamica varianței și nici autocorelația mediei. Rolul ei privește cozile lui $z_t$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "ARMA-GARCH",
                "text": "Why estimate an AR(1)-GARCH(1,1) jointly instead of reporting the OLS $t$ statistic of the AR(1) coefficient?",
                "options": [
                    "With GARCH errors the usual OLS standard errors are wrong, and the joint ML estimate weights calm and stormy days correctly",
                    "Because OLS cannot estimate an AR(1)",
                    "Because the AR coefficient is always zero for returns",
                    "Because GARCH removes the need for a mean model"
                ],
                "correctExplanation": "OLS assumes a constant variance; under conditional heteroskedasticity its standard errors are invalid. In the joint model the AR coefficient of the S&P 500 shrinks from about $-0.10$ to $-0.05$ once the storms get less weight.",
                "incorrectExplanation": "OLS can estimate an AR(1), but its classic $t$ tests assume homoskedastic errors; the AR coefficient is not always zero (BET: about 0.09), and GARCH models the variance, not the mean."
            },
            "ro": {
                "title": "ARMA-GARCH",
                "text": "De ce estimăm simultan un AR(1)-GARCH(1,1) în loc să raportăm statistica $t$ OLS a coeficientului AR(1)?",
                "options": [
                    "Cu erori GARCH, erorile standard OLS obișnuite sînt greșite, iar estimarea ML comună ponderează corect zilele liniștite și cele agitate",
                    "Pentru că OLS nu poate estima un AR(1)",
                    "Pentru că pentru randamente coeficientul AR este întotdeauna zero",
                    "Pentru că GARCH face inutil un model pentru medie"
                ],
                "correctExplanation": "OLS presupune o varianță constantă; cu heteroscedasticitate condiționată, erorile lui standard nu sînt valide. În modelul comun, coeficientul AR pentru S&P 500 scade de la circa $-0,10$ la $-0,05$ cînd furtunile primesc o pondere mai mică.",
                "incorrectExplanation": "OLS poate estima un AR(1), dar testele $t$ clasice presupun erori homoscedastice; coeficientul AR nu este întotdeauna zero (BET: circa 0,09), iar GARCH modelează varianța, nu media."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "GJR-GARCH",
                "text": "In GJR-GARCH, $\\sigma_t^2 = \\omega + (\\alpha + \\gamma I_{t-1})\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$ with $I_{t-1} = 1$ if $\\varepsilon_{t-1} < 0$. What does $\\gamma > 0$ mean?",
                "options": [
                    "Positive shocks raise the variance more than negative ones",
                    "Negative shocks raise the variance more than positive shocks of the same size",
                    "The variance is not persistent",
                    "The innovations are skewed to the right"
                ],
                "correctExplanation": "Bad news has the slope $\\alpha + \\gamma$, good news only $\\alpha$: the leverage effect. For the S&P 500, $\\hat\\alpha \\approx 0$ and $\\hat\\gamma \\approx 0.2$: only falls raise the variance.",
                "incorrectExplanation": "$\\gamma$ is added only for negative shocks; it is about the news impact, not about the persistence ($\\alpha + \\beta + \\gamma/2$) or the shape of the innovation distribution."
            },
            "ro": {
                "title": "GJR-GARCH",
                "text": "În GJR-GARCH, $\\sigma_t^2 = \\omega + (\\alpha + \\gamma I_{t-1})\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, cu $I_{t-1} = 1$ dacă $\\varepsilon_{t-1} < 0$. Ce înseamnă $\\gamma > 0$?",
                "options": [
                    "Șocurile pozitive cresc varianța mai mult decît cele negative",
                    "Șocurile negative cresc varianța mai mult decît șocurile pozitive de aceeași mărime",
                    "Varianța nu este persistentă",
                    "Inovațiile sînt asimetrice la dreapta"
                ],
                "correctExplanation": "Știrile proaste au panta $\\alpha + \\gamma$, cele bune doar $\\alpha$: efectul de levier. Pentru S&P 500, $\\hat\\alpha \\approx 0$ și $\\hat\\gamma \\approx 0,2$: doar scăderile cresc varianța.",
                "incorrectExplanation": "$\\gamma$ se adaugă doar pentru șocurile negative; privește impactul știrilor, nu persistența ($\\alpha + \\beta + \\gamma/2$) sau forma distribuției inovațiilor."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "EGARCH",
                "text": "What is an advantage of EGARCH, $\\ln\\sigma_t^2 = \\omega + \\alpha(|z_{t-1}| - E|z_{t-1}|) + \\gamma z_{t-1} + \\beta\\ln\\sigma_{t-1}^2$?",
                "options": [
                    "It has fewer parameters than ARCH(1)",
                    "It needs no innovation distribution",
                    "The logarithm keeps the variance positive without sign restrictions, and $\\gamma$ captures asymmetry",
                    "It always forecasts better than GARCH"
                ],
                "correctExplanation": "Since the model is written for $\\ln\\sigma_t^2$, any parameter values give a positive variance; a negative $\\gamma$ means that bad news raises volatility more. Its persistence is $\\beta$.",
                "incorrectExplanation": "EGARCH has more parameters than ARCH(1), still needs a distribution for $z_t$, and is not always better out of sample; its advantages are positivity without constraints and a sign term."
            },
            "ro": {
                "title": "EGARCH",
                "text": "Care este un avantaj al EGARCH, $\\ln\\sigma_t^2 = \\omega + \\alpha(|z_{t-1}| - E|z_{t-1}|) + \\gamma z_{t-1} + \\beta\\ln\\sigma_{t-1}^2$?",
                "options": [
                    "Are mai puțini parametri decît ARCH(1)",
                    "Nu are nevoie de o distribuție a inovațiilor",
                    "Logaritmul păstrează varianța pozitivă fără restricții de semn, iar $\\gamma$ surprinde asimetria",
                    "Prognozează întotdeauna mai bine decît GARCH"
                ],
                "correctExplanation": "Deoarece modelul este scris pentru $\\ln\\sigma_t^2$, orice valori ale parametrilor dau o varianță pozitivă; un $\\gamma$ negativ înseamnă că știrile proaste cresc volatilitatea mai mult. Persistența lui este $\\beta$.",
                "incorrectExplanation": "EGARCH are mai mulți parametri decît ARCH(1), are nevoie tot de o distribuție pentru $z_t$ și nu este întotdeauna mai bun în afara eșantionului; avantajele lui sînt pozitivitatea fără restricții și termenul de semn."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Diagnostics",
                "text": "After fitting a GARCH model, what should the Ljung–Box test on the squared standardised residuals $\\hat z_t^2$ show?",
                "options": [
                    "A rejection, since squared returns are always autocorrelated",
                    "A value equal to the test on $r_t^2$",
                    "A negative statistic",
                    "No rejection: the model has absorbed the ARCH effects"
                ],
                "correctExplanation": "If the model is right, $\\hat z_t = (r_t - \\hat\\mu_t)/\\hat\\sigma_t$ is close to i.i.d., so $\\hat z_t^2$ is not autocorrelated; on the S&P 500, $Q(10)$ falls from about 6100 for $r_t^2$ to about 14 for $\\hat z_t^2$. A pass does not prove the model right.",
                "incorrectExplanation": "The test on $\\hat z_t^2$ checks what is left after the model; a rejection would mean remaining ARCH effects. A Ljung–Box statistic cannot be negative."
            },
            "ro": {
                "title": "Diagnosticare",
                "text": "După estimarea unui model GARCH, ce ar trebui să arate testul Ljung–Box pe pătratele reziduurilor standardizate $\\hat z_t^2$?",
                "options": [
                    "O respingere, deoarece pătratele randamentelor sînt întotdeauna autocorelate",
                    "O valoare egală cu cea a testului pe $r_t^2$",
                    "O statistică negativă",
                    "Nicio respingere: modelul a absorbit efectele ARCH"
                ],
                "correctExplanation": "Dacă modelul este corect, $\\hat z_t = (r_t - \\hat\\mu_t)/\\hat\\sigma_t$ este aproape i.i.d., deci $\\hat z_t^2$ nu este autocorelat; pe S&P 500, $Q(10)$ scade de la circa 6100 pentru $r_t^2$ la circa 14 pentru $\\hat z_t^2$. Un test trecut nu dovedește că modelul este corect.",
                "incorrectExplanation": "Testul pe $\\hat z_t^2$ verifică ce a rămas după model; o respingere ar însemna efecte ARCH rămase. O statistică Ljung–Box nu poate fi negativă."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Volatility forecasts",
                "text": "In a covariance-stationary GARCH(1,1), what happens to the forecast $E_t[\\sigma_{t+h}^2]$ as $h$ grows?",
                "options": [
                    "It converges to the long-run variance $\\omega/(1 - \\alpha - \\beta)$ at the speed $(\\alpha + \\beta)^{h-1}$",
                    "It stays equal to $\\sigma_{t+1}^2$",
                    "It grows without bound",
                    "It converges to zero"
                ],
                "correctExplanation": "$E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$: like an AR(1) forecast converging to its mean (Chapter 2); the term structure slopes down in a storm and up in a calm period.",
                "incorrectExplanation": "A flat forecast is the IGARCH or EWMA case; a stationary GARCH reverts to its long-run level, neither to zero nor to infinity."
            },
            "ro": {
                "title": "Prognoza volatilității",
                "text": "Într-un GARCH(1,1) staționar în covarianță, ce se întîmplă cu prognoza $E_t[\\sigma_{t+h}^2]$ cînd $h$ crește?",
                "options": [
                    "Converge spre varianța de lungă durată $\\omega/(1 - \\alpha - \\beta)$ cu viteza $(\\alpha + \\beta)^{h-1}$",
                    "Rămîne egală cu $\\sigma_{t+1}^2$",
                    "Crește nelimitat",
                    "Converge spre zero"
                ],
                "correctExplanation": "$E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$: ca prognoza unui AR(1) care converge spre medie (Capitolul 2); structura la termen coboară într-o furtună și urcă într-o perioadă liniștită.",
                "incorrectExplanation": "O prognoză constantă corespunde cazului IGARCH sau EWMA; un GARCH staționar revine la nivelul de lungă durată, nici la zero, nici la infinit."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Multi-day risk",
                "text": "How is the variance of the 10-day return computed from a GARCH(1,1)?",
                "options": [
                    "As $10\\,\\sigma_{t+1}^2$, the square-root-of-time rule",
                    "As the sum of the ten daily variance forecasts $E_t[\\sigma_{t+1}^2] + \\dots + E_t[\\sigma_{t+10}^2]$",
                    "As $\\sigma_{t+10}^2$ only",
                    "As the sample variance times 10"
                ],
                "correctExplanation": "Daily returns are uncorrelated, so the variance of their sum is the sum of the daily variances; with GARCH these change with the horizon, so $10\\,\\sigma_{t+1}^2$ is too high in a crisis and too low in a calm period.",
                "incorrectExplanation": "The square-root-of-time rule assumes a constant variance; the 10-day variance needs all ten daily forecasts, not only the last one or the sample average."
            },
            "ro": {
                "title": "Riscul pe mai multe zile",
                "text": "Cum se calculează varianța randamentului pe 10 zile dintr-un GARCH(1,1)?",
                "options": [
                    "Ca $10\\,\\sigma_{t+1}^2$, regula rădăcinii pătrate a timpului",
                    "Ca suma celor zece prognoze zilnice ale varianței $E_t[\\sigma_{t+1}^2] + \\dots + E_t[\\sigma_{t+10}^2]$",
                    "Doar ca $\\sigma_{t+10}^2$",
                    "Ca varianța de selecție înmulțită cu 10"
                ],
                "correctExplanation": "Randamentele zilnice sînt necorelate, deci varianța sumei lor este suma varianțelor zilnice; cu GARCH acestea se schimbă cu orizontul, așa că $10\\,\\sigma_{t+1}^2$ este prea mare în criză și prea mic într-o perioadă liniștită.",
                "incorrectExplanation": "Regula rădăcinii pătrate a timpului presupune o varianță constantă; varianța pe 10 zile are nevoie de toate cele zece prognoze zilnice, nu doar de ultima sau de media de selecție."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Evaluating volatility forecasts",
                "text": "Two variance forecasts are compared out of sample with the QLIKE loss $r_t^2/h_t + \\ln h_t$. Which statement is correct?",
                "options": [
                    "A forecast that is too high is penalised more than one that is too low",
                    "QLIKE needs the true variance $\\sigma_t^2$, which is observed",
                    "QLIKE ranks forecasts correctly with the noisy proxy $r_t^2$, and the Diebold–Mariano test checks whether the mean loss difference is zero",
                    "The model with the higher QLIKE is better"
                ],
                "correctExplanation": "Patton (2011): QLIKE gives the right ranking even with a noisy proxy; it punishes forecasts that are too low. The DM test (with HAC standard errors) guards against conclusions driven by a few days, as for EUR/RON on 6 May 2025.",
                "incorrectExplanation": "The true variance is never observed; lower QLIKE is better; and QLIKE penalises under-prediction more than over-prediction, since $r_t^2/h_t$ explodes when $h_t$ is small."
            },
            "ro": {
                "title": "Evaluarea prognozelor de volatilitate",
                "text": "Două prognoze ale varianței se compară în afara eșantionului cu pierderea QLIKE $r_t^2/h_t + \\ln h_t$. Care afirmație este corectă?",
                "options": [
                    "O prognoză prea mare este penalizată mai mult decît una prea mică",
                    "QLIKE are nevoie de varianța adevărată $\\sigma_t^2$, care este observată",
                    "QLIKE ordonează corect prognozele cu aproximarea zgomotoasă $r_t^2$, iar testul Diebold–Mariano verifică dacă diferența medie a pierderilor este zero",
                    "Modelul cu QLIKE mai mare este mai bun"
                ],
                "correctExplanation": "Patton (2011): QLIKE dă ordinea corectă chiar și cu o aproximare zgomotoasă; penalizează prognozele prea mici. Testul DM (cu erori standard HAC) ne ferește de concluzii determinate de cîteva zile, ca la EUR/RON pe 6 mai 2025.",
                "incorrectExplanation": "Varianța adevărată nu este observată niciodată; un QLIKE mai mic este mai bun; iar QLIKE penalizează subestimarea mai mult decît supraestimarea, deoarece $r_t^2/h_t$ explodează cînd $h_t$ este mic."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "VaR 1% from GARCH",
                "text": "A GARCH(1,1)-t gives $\\mu = 0$, $\\sigma_{t+1} = 1\\%$ and $\\nu = 5$. Compared with Normal innovations, the one-day VaR 1% is:",
                "options": [
                    "Smaller, because the t distribution has more mass near zero",
                    "The same, because the variance is the same",
                    "Undefined, because VaR needs Normal innovations",
                    "Larger: $-q_{0.01}(z) = 3.365\\sqrt{3/5} = 2.61\\%$ against $2.33\\%$"
                ],
                "correctExplanation": "$\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}q_{0.01}(z))$; for the standardised t with 5 degrees of freedom the 1% quantile is $-2.61$, further from zero than the Normal $-2.326$: heavy tails raise the VaR 1% at the same variance.",
                "incorrectExplanation": "With the same variance, a heavy-tailed distribution puts more probability in the far tail, so its 1% quantile is further from zero; VaR 1% is defined for any distribution of $z_t$."
            },
            "ro": {
                "title": "VaR 1% dintr-un GARCH",
                "text": "Un GARCH(1,1)-t dă $\\mu = 0$, $\\sigma_{t+1} = 1\\%$ și $\\nu = 5$. Față de inovații Normale, VaR 1% pe o zi este:",
                "options": [
                    "Mai mic, deoarece distribuția t are mai multă masă în jurul lui zero",
                    "Același, deoarece varianța este aceeași",
                    "Nedefinit, deoarece VaR cere inovații Normale",
                    "Mai mare: $-q_{0,01}(z) = 3,365\\sqrt{3/5} = 2,61\\%$, față de $2,33\\%$"
                ],
                "correctExplanation": "$\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}q_{0,01}(z))$; pentru t standardizată cu 5 grade de libertate cuantila de 1% este $-2,61$, mai departe de zero decît valoarea Normală $-2,326$: cozile groase cresc VaR 1% la aceeași varianță.",
                "incorrectExplanation": "La aceeași varianță, o distribuție cu cozi groase pune mai multă probabilitate în coada îndepărtată, deci cuantila ei de 1% este mai departe de zero; VaR 1% este definit pentru orice distribuție a lui $z_t$."
            }
        }
    ]
};
