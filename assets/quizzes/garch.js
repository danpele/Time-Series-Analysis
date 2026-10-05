// ============================================================
// Chapter 5 quiz bank: Conditional volatility: ARCH and GARCH (EN + RO)
// 20 questions ported from the 2025/2026 site; 20 drawn per attempt.
// The bank grows to 24 questions when the chapter is rebuilt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['garch'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Volatility clustering",
                "text": "What does \"volatility clustering\" mean?",
                "options": [
                    "Volatility is constant over time",
                    "Periods of high volatility tend to be followed by further periods of high volatility",
                    "Returns are autocorrelated over time",
                    "Returns follow the Normal distribution"
                ],
                "correctExplanation": "Volatility clustering (Mandelbrot, 1963): turbulent periods tend to be followed by turbulent periods and calm periods by calm ones. This persistence of volatility motivates ARCH/GARCH models.",
                "incorrectExplanation": "Clustering concerns the magnitude of returns (squared or absolute returns are autocorrelated), not the returns themselves, which are close to uncorrelated; and it is incompatible with constant volatility. In Mandelbrot's words, \"large changes tend to be followed by large changes\"."
            },
            "ro": {
                "title": "Volatility clustering",
                "text": "Ce reprezintă fenomenul de „volatility clustering”?",
                "options": [
                    "Volatilitatea este constantă în timp",
                    "Perioadele de volatilitate ridicată tind să fie urmate de alte perioade de volatilitate ridicată",
                    "Randamentele sînt autocorelate în timp",
                    "Randamentele au distribuția Normală"
                ],
                "correctExplanation": "Volatility clustering (Mandelbrot, 1963): perioadele agitate tind să fie urmate de perioade agitate, iar cele calme de perioade calme. Această persistență a volatilității motivează modelele ARCH/GARCH.",
                "incorrectExplanation": "Fenomenul privește mărimea randamentelor (randamentele la pătrat sau în valoare absolută sînt autocorelate), nu randamentele propriu-zise, care sînt aproape necorelate; în plus, este incompatibil cu o volatilitate constantă. În formularea lui Mandelbrot, „large changes tend to be followed by large changes”."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "GARCH parameters",
                "text": "In the GARCH(1,1) model σₜ² = ω + α·εₜ₋₁² + β·σₜ₋₁², what does α represent?",
                "options": [
                    "The persistence of volatility",
                    "The baseline level of volatility",
                    "The reaction to recent shocks (news coefficient)",
                    "The unconditional variance"
                ],
                "correctExplanation": "α is the ARCH coefficient: it measures how strongly volatility reacts to the latest squared shock εₜ₋₁². β measures persistence (the memory of volatility) and ω sets the baseline level.",
                "incorrectExplanation": "Persistence is governed by β (and overall by α + β), the baseline by ω, and the unconditional variance is ω/(1 − α − β), a combination of all three. A large α means volatility reacts strongly to recent news."
            },
            "ro": {
                "title": "Parametrii GARCH",
                "text": "În modelul GARCH(1,1) σₜ² = ω + α·εₜ₋₁² + β·σₜ₋₁², ce reprezintă α?",
                "options": [
                    "Persistența volatilității",
                    "Nivelul de bază al volatilității",
                    "Reacția la șocurile recente (news coefficient)",
                    "Varianța necondiționată"
                ],
                "correctExplanation": "α este coeficientul ARCH: măsoară cît de puternic reacționează volatilitatea la ultimul șoc la pătrat, εₜ₋₁². β măsoară persistența (memoria volatilității), iar ω fixează nivelul de bază.",
                "incorrectExplanation": "Persistența este dată de β (și, global, de α + β), nivelul de bază de ω, iar varianța necondiționată este ω/(1 − α − β), o combinație a tuturor celor trei. Un α mare înseamnă că volatilitatea reacționează puternic la veștile recente."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Stationarity condition",
                "text": "With ω > 0, α ≥ 0 and β ≥ 0, what is the condition for covariance stationarity of a GARCH(1,1)?",
                "options": [
                    "ω > 0",
                    "α + β = 1",
                    "α + β < 1",
                    "α > β"
                ],
                "correctExplanation": "α + β < 1 ensures mean reversion: volatility reverts to the unconditional level σ̄² = ω/(1 − α − β). If α + β = 1 the model is IGARCH (shocks to volatility persist indefinitely).",
                "incorrectExplanation": "ω > 0 only guarantees a positive variance; α + β = 1 is the IGARCH boundary, where the unconditional variance is infinite; the relative size of α and β is irrelevant for stationarity. The condition α + β < 1 guarantees a finite unconditional variance."
            },
            "ro": {
                "title": "Condiția de staționaritate",
                "text": "Cu ω > 0, α ≥ 0 și β ≥ 0, care este condiția de staționaritate în covarianță pentru un GARCH(1,1)?",
                "options": [
                    "ω > 0",
                    "α + β = 1",
                    "α + β < 1",
                    "α > β"
                ],
                "correctExplanation": "Condiția α + β < 1 asigură revenirea la medie: volatilitatea revine la nivelul necondiționat σ̄² = ω/(1 − α − β). Dacă α + β = 1, modelul devine IGARCH (șocurile de volatilitate persistă la nesfîrșit).",
                "incorrectExplanation": "ω > 0 garantează doar o varianță pozitivă; α + β = 1 este limita IGARCH, unde varianța necondiționată este infinită; mărimea relativă a lui α și β nu contează pentru staționaritate. Condiția α + β < 1 garantează o varianță necondiționată finită."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Unconditional variance",
                "text": "What is the unconditional variance of a covariance-stationary GARCH(1,1)?",
                "options": [
                    "σ̄² = ω",
                    "σ̄² = ω / (1 − α)",
                    "σ̄² = ω / (1 − α − β)",
                    "σ̄² = ω / (α + β)"
                ],
                "correctExplanation": "From E[σₜ²] = ω + (α + β)·E[σₜ²] we get σ̄² = ω/(1 − α − β). Example with daily returns: ω = 0.00001, α = 0.05, β = 0.93 give σ̄² = 0.0005, i.e. a daily volatility of about 2.24%, roughly 35% annualised (×√252).",
                "incorrectExplanation": "ω alone ignores the feedback from past shocks and past variance; ω/(1 − α) is the ARCH(1) formula, which omits β; ω/(α + β) has no derivation. The formula ω/(1 − α − β) requires α + β < 1."
            },
            "ro": {
                "title": "Varianța necondiționată",
                "text": "Care este varianța necondiționată a unui GARCH(1,1) staționar în covarianță?",
                "options": [
                    "σ̄² = ω",
                    "σ̄² = ω / (1 − α)",
                    "σ̄² = ω / (1 − α − β)",
                    "σ̄² = ω / (α + β)"
                ],
                "correctExplanation": "Din E[σₜ²] = ω + (α + β)·E[σₜ²] rezultă σ̄² = ω/(1 − α − β). Exemplu pentru randamente zilnice: ω = 0,00001, α = 0,05 și β = 0,93 dau σ̄² = 0,0005, adică o volatilitate zilnică de aproximativ 2,24%, circa 35% anualizat (×√252).",
                "incorrectExplanation": "ω singur ignoră efectul șocurilor și al varianței din trecut; ω/(1 − α) este formula pentru ARCH(1), care omite β; ω/(α + β) nu are nicio justificare. Formula ω/(1 − α − β) cere α + β < 1."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Leverage effect",
                "text": "What is the \"leverage effect\"?",
                "options": [
                    "Positive shocks raise volatility more than negative ones",
                    "Negative shocks raise volatility more than positive shocks of the same size",
                    "Volatility does not depend on the sign of shocks",
                    "Returns have an asymmetric distribution"
                ],
                "correctExplanation": "Leverage effect (Black, 1976): a fall in the share price raises the debt-to-equity ratio, the firm becomes riskier and volatility rises. A standard GARCH cannot capture this, because it depends on ε² and is symmetric in the sign of the shock.",
                "incorrectExplanation": "The asymmetry runs from bad news to higher volatility, not the other way round; sign independence is exactly what the standard GARCH assumes; skewness of returns is a different property of the unconditional distribution. Bad news amplifies volatility more than good news."
            },
            "ro": {
                "title": "Leverage effect",
                "text": "Ce este „leverage effect”?",
                "options": [
                    "Șocurile pozitive cresc volatilitatea mai mult decît cele negative",
                    "Șocurile negative cresc volatilitatea mai mult decît șocurile pozitive de aceeași mărime",
                    "Volatilitatea nu depinde de semnul șocurilor",
                    "Randamentele au o distribuție asimetrică"
                ],
                "correctExplanation": "Leverage effect (Black, 1976): o scădere a prețului acțiunii crește raportul dintre datorii și capitalul propriu, firma devine mai riscantă, iar volatilitatea crește. Un GARCH standard nu poate surprinde acest efect, deoarece depinde de ε² și este simetric în raport cu semnul șocului.",
                "incorrectExplanation": "Asimetria merge de la veștile proaste spre volatilitate mai mare, nu invers; independența de semn este exact ipoteza GARCH standard; asimetria randamentelor este o altă proprietate, a distribuției necondiționate. Veștile proaste amplifică volatilitatea mai mult decît veștile bune."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "EGARCH asymmetry parameter",
                "text": "In the EGARCH model ln(σₜ²) = ω + α|zₜ₋₁| + γzₜ₋₁ + β·ln(σₜ₋₁²), a negative γ indicates:",
                "options": [
                    "No leverage effect",
                    "The presence of a leverage effect",
                    "Constant volatility",
                    "A nonstationary model"
                ],
                "correctExplanation": "EGARCH (Nelson, 1991): with γ < 0 a negative standardised shock (zₜ₋₁ < 0) raises ln(σₜ²) by α|z| − γ|z| = (α + |γ|)|z|, more than a positive shock of the same size, i.e. a leverage effect.",
                "incorrectExplanation": "γ = 0 would mean symmetry (no leverage effect); volatility is constant only if α = γ = β = 0; stationarity depends on |β| < 1, not on the sign of γ. A negative γ confirms that negative returns raise volatility more."
            },
            "ro": {
                "title": "Parametrul de asimetrie din EGARCH",
                "text": "În modelul EGARCH ln(σₜ²) = ω + α|zₜ₋₁| + γzₜ₋₁ + β·ln(σₜ₋₁²), un parametru γ negativ indică:",
                "options": [
                    "Absența leverage effect",
                    "Prezența leverage effect",
                    "Volatilitate constantă",
                    "Un model nestaționar"
                ],
                "correctExplanation": "EGARCH (Nelson, 1991): cu γ < 0, un șoc standardizat negativ (zₜ₋₁ < 0) crește ln(σₜ²) cu α|z| − γ|z| = (α + |γ|)|z|, mai mult decît un șoc pozitiv de aceeași mărime, adică apare leverage effect.",
                "incorrectExplanation": "γ = 0 ar însemna simetrie (fără leverage effect); volatilitatea este constantă doar dacă α = γ = β = 0; staționaritatea depinde de |β| < 1, nu de semnul lui γ. Un γ negativ confirmă că randamentele negative cresc volatilitatea mai mult."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Advantage of EGARCH",
                "text": "What is the main advantage of EGARCH over GARCH?",
                "options": [
                    "It is faster to estimate",
                    "It needs no non-negativity restrictions on the parameters",
                    "It has fewer parameters",
                    "It is easier to interpret"
                ],
                "correctExplanation": "EGARCH models ln(σ²) rather than σ². Any value of ln(σ²) gives σ² > 0 automatically, without the restrictions ω > 0, α ≥ 0, β ≥ 0 required in GARCH; it also captures asymmetry.",
                "incorrectExplanation": "EGARCH is not faster to estimate and has at least as many parameters as GARCH(1,1) (the extra γ); its log-scale parameters are, if anything, harder to interpret. The key gain is positivity without parameter constraints."
            },
            "ro": {
                "title": "Avantajul EGARCH",
                "text": "Care este principalul avantaj al EGARCH față de GARCH?",
                "options": [
                    "Se estimează mai rapid",
                    "Nu necesită restricții de nenegativitate asupra parametrilor",
                    "Are mai puțini parametri",
                    "Este mai ușor de interpretat"
                ],
                "correctExplanation": "EGARCH modelează ln(σ²), nu σ². Orice valoare a lui ln(σ²) dă automat σ² > 0, fără restricțiile ω > 0, α ≥ 0, β ≥ 0 necesare în GARCH; în plus, surprinde asimetria.",
                "incorrectExplanation": "EGARCH nu se estimează mai rapid și are cel puțin tot atîția parametri cît GARCH(1,1) (în plus, γ); parametrii pe scară logaritmică sînt, mai degrabă, mai greu de interpretat. Avantajul esențial este pozitivitatea fără restricții asupra parametrilor."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "ARCH-LM test",
                "text": "Which test is used to detect ARCH effects in residuals?",
                "options": [
                    "The Dickey-Fuller test",
                    "The Ljung-Box test on the residuals",
                    "Engle's ARCH-LM test",
                    "The Breusch-Pagan test"
                ],
                "correctExplanation": "ARCH-LM (Engle, 1982): regress ε̂ₜ² on ε̂ₜ₋₁², …, ε̂ₜ₋q² and use T·R² ~ χ²(q). Rejection indicates conditional heteroskedasticity.",
                "incorrectExplanation": "Dickey-Fuller tests for unit roots; Ljung-Box on the residuals themselves detects autocorrelation in the mean (on squared residuals it becomes the McLeod-Li test, a close relative of ARCH-LM); Breusch-Pagan tests heteroskedasticity linked to regressors, not to past shocks."
            },
            "ro": {
                "title": "Testul ARCH-LM",
                "text": "Ce test se folosește pentru a detecta efecte ARCH în reziduuri?",
                "options": [
                    "Testul Dickey-Fuller",
                    "Testul Ljung-Box aplicat reziduurilor",
                    "Testul ARCH-LM al lui Engle",
                    "Testul Breusch-Pagan"
                ],
                "correctExplanation": "ARCH-LM (Engle, 1982): se regresează ε̂ₜ² pe ε̂ₜ₋₁², …, ε̂ₜ₋q² și se folosește T·R² ~ χ²(q). Respingerea ipotezei nule indică heteroscedasticitate condiționată.",
                "incorrectExplanation": "Testul Dickey-Fuller privește rădăcinile unitare; Ljung-Box aplicat reziduurilor propriu-zise detectează autocorelația în medie (aplicat pătratelor reziduurilor devine testul McLeod-Li, înrudit cu ARCH-LM); Breusch-Pagan testează heteroscedasticitatea legată de regresori, nu de șocurile trecute."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Typical persistence",
                "text": "For daily S&P 500 returns, typical estimates of α + β in a GARCH(1,1) are:",
                "options": [
                    "0.50 – 0.70",
                    "0.70 – 0.85",
                    "0.95 – 0.99",
                    "Greater than 1"
                ],
                "correctExplanation": "Typical daily S&P 500 estimates: α ≈ 0.04–0.08 and β ≈ 0.90–0.94, so α + β ≈ 0.97–0.99. Persistence is very high: volatility shocks fade slowly (a half-life of several weeks).",
                "incorrectExplanation": "Values of 0.5–0.85 would imply that volatility shocks vanish within days, which contradicts the observed clustering; values above 1 would imply an explosive, nonstationary variance. Market volatility is highly persistent, with α + β close to but below 1."
            },
            "ro": {
                "title": "Valori tipice ale persistenței",
                "text": "Pentru randamentele zilnice ale indicelui S&P 500, valorile tipice estimate pentru α + β într-un GARCH(1,1) sînt:",
                "options": [
                    "0,50 – 0,70",
                    "0,70 – 0,85",
                    "0,95 – 0,99",
                    "Mai mari decît 1"
                ],
                "correctExplanation": "Estimări tipice pentru S&P 500, date zilnice: α ≈ 0,04–0,08 și β ≈ 0,90–0,94, deci α + β ≈ 0,97–0,99. Persistența este foarte ridicată: șocurile de volatilitate se sting lent (timp de înjumătățire de cîteva săptămîni).",
                "incorrectExplanation": "Valori de 0,5–0,85 ar însemna că șocurile de volatilitate dispar în cîteva zile, ceea ce contrazice volatility clustering observat; valori peste 1 ar implica o varianță explozivă, nestaționară. Volatilitatea piețelor este foarte persistentă, cu α + β apropiat de 1, dar sub 1."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Innovation distribution",
                "text": "Which distribution is most often used for GARCH innovations in order to capture fat tails?",
                "options": [
                    "The Normal distribution",
                    "The uniform distribution",
                    "Student's t distribution",
                    "The exponential distribution"
                ],
                "correctExplanation": "Student's t with ν degrees of freedom (rescaled to unit variance) captures fat tails (leptokurtosis); as ν → ∞ it converges to the Normal distribution. A common alternative is the GED (generalised error distribution).",
                "incorrectExplanation": "Even a GARCH with Normal innovations has fat-tailed unconditional returns, but its standardised residuals are usually still too fat-tailed for the Normal distribution; the uniform distribution has thin, bounded tails; the exponential distribution is one-sided. Student's t fits the residual tails best among these."
            },
            "ro": {
                "title": "Distribuția inovațiilor",
                "text": "Ce distribuție se folosește cel mai des pentru inovațiile GARCH, pentru a surprinde cozile groase?",
                "options": [
                    "Distribuția Normală",
                    "Distribuția uniformă",
                    "Distribuția Student t",
                    "Distribuția exponențială"
                ],
                "correctExplanation": "Distribuția Student t cu ν grade de libertate (rescalată la varianță unitară) surprinde cozile groase (leptocurtoză); cînd ν → ∞, converge către distribuția Normală. O alternativă frecventă este GED (generalised error distribution).",
                "incorrectExplanation": "Chiar și un GARCH cu inovații Normale produce randamente necondiționate cu cozi groase, dar reziduurile standardizate au de obicei cozi încă prea groase pentru distribuția Normală; distribuția uniformă are cozi subțiri și mărginite; distribuția exponențială este unilaterală. Dintre acestea, distribuția Student t descrie cel mai bine cozile reziduurilor."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "News impact curve",
                "text": "What does the news impact curve of a GARCH-type model show?",
                "options": [
                    "The relationship between returns and trading volume",
                    "How shocks (positive and negative) affect next-period volatility",
                    "The forecast of prices",
                    "The correlation between assets"
                ],
                "correctExplanation": "News impact curve (Engle and Ng, 1993): σₜ² plotted against εₜ₋₁, with past variance held at its unconditional level. GARCH gives a symmetric parabola; EGARCH and GJR give an asymmetric curve, steeper for negative shocks.",
                "incorrectExplanation": "The curve does not involve volume, prices or cross-asset correlation; it maps the size and sign of a shock into next-period conditional variance, which is why it is the standard way to visualise asymmetry."
            },
            "ro": {
                "title": "News impact curve",
                "text": "Ce arată news impact curve pentru un model de tip GARCH?",
                "options": [
                    "Relația dintre randamente și volumul tranzacțiilor",
                    "Cum afectează șocurile (pozitive și negative) volatilitatea din perioada următoare",
                    "Prognoza prețurilor",
                    "Corelația dintre active"
                ],
                "correctExplanation": "News impact curve (Engle și Ng, 1993): graficul lui σₜ² în funcție de εₜ₋₁, cu varianța trecută fixată la nivelul ei necondiționat. GARCH dă o parabolă simetrică; EGARCH și GJR dau o curbă asimetrică, mai abruptă pentru șocurile negative.",
                "incorrectExplanation": "Curba nu implică volumul, prețurile sau corelația dintre active; ea transformă mărimea și semnul unui șoc în varianța condiționată din perioada următoare, de aceea este modul standard de a vizualiza asimetria."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "GARCH and ARCH",
                "text": "What is the main advantage of GARCH over ARCH?",
                "options": [
                    "It is faster to estimate",
                    "It models the persistence of volatility with fewer parameters",
                    "It needs no historical data",
                    "It works only with daily data"
                ],
                "correctExplanation": "GARCH(1,1) captures persistent volatility with only 3 parameters (ω, α, β) and is equivalent to an ARCH(∞) with geometrically declining weights. An ARCH(q) would need q + 1 parameters, with q large, for the same persistence.",
                "incorrectExplanation": "Estimation speed is similar, both models need historical data, and both can be applied at any frequency. The gain comes from the term β·σₜ₋₁², which captures the memory of volatility parsimoniously."
            },
            "ro": {
                "title": "GARCH față de ARCH",
                "text": "Care este principalul avantaj al GARCH față de ARCH?",
                "options": [
                    "Se estimează mai rapid",
                    "Modelează persistența volatilității cu mai puțini parametri",
                    "Nu necesită date istorice",
                    "Funcționează doar cu date zilnice"
                ],
                "correctExplanation": "GARCH(1,1) surprinde persistența volatilității cu doar 3 parametri (ω, α, β) și este echivalent cu un ARCH(∞) cu ponderi descrescătoare geometric. Un ARCH(q) ar avea nevoie de q + 1 parametri, cu q mare, pentru aceeași persistență.",
                "incorrectExplanation": "Viteza de estimare este similară, ambele modele necesită date istorice și ambele se pot aplica la orice frecvență. Avantajul provine din termenul β·σₜ₋₁², care surprinde parcimonios memoria volatilității."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Value at Risk",
                "text": "The one-day VaR 1% of a position represents:",
                "options": [
                    "The maximum gain achieved with probability 1%",
                    "The loss that is exceeded with a probability of only 1%",
                    "The mean return",
                    "The average volatility"
                ],
                "correctExplanation": "VaR 1% is minus the 1% quantile of the return distribution: VaR₀.₀₁ = −q₀.₀₁. With GARCH, VaRₜ = −(μ + z₀.₀₁·σₜ), where z₀.₀₁ = −2.326 for the Normal distribution and is larger in absolute value for a standardised Student t.",
                "incorrectExplanation": "VaR is a loss measure, not a gain, a mean or a volatility; it is a quantile, so it says nothing about how large the losses beyond it are (that is what ES measures). With daily data, VaR 1% is exceeded on about 1% of days, roughly 2.5 days a year."
            },
            "ro": {
                "title": "Value at Risk",
                "text": "VaR 1% la orizont de o zi pentru o poziție reprezintă:",
                "options": [
                    "Cîștigul maxim obținut cu probabilitate 1%",
                    "Pierderea care este depășită cu o probabilitate de numai 1%",
                    "Randamentul mediu",
                    "Volatilitatea medie"
                ],
                "correctExplanation": "VaR 1% este cuantila de 1% a distribuției randamentelor, cu semn schimbat: VaR₀,₀₁ = −q₀,₀₁. Cu GARCH, VaRₜ = −(μ + z₀,₀₁·σₜ), unde z₀,₀₁ = −2,326 pentru distribuția Normală și este mai mare în valoare absolută pentru o distribuție Student t standardizată.",
                "incorrectExplanation": "VaR este o măsură a pierderii, nu un cîștig, o medie sau o volatilitate; fiind o cuantilă, nu spune cît de mari sînt pierderile de dincolo de ea (aceasta măsoară ES). Pe date zilnice, VaR 1% este depășit în aproximativ 1% din zile, adică în circa 2,5 zile pe an."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "GJR-GARCH",
                "text": "In the GJR-GARCH model, the term γ·I(εₜ₋₁ < 0)·εₜ₋₁² captures:",
                "options": [
                    "The effect of positive shocks",
                    "The additional impact of negative shocks (leverage effect)",
                    "The constant of the model",
                    "The persistence of volatility"
                ],
                "correctExplanation": "GJR-GARCH (Glosten, Jagannathan and Runkle, 1993): the indicator I(εₜ₋₁ < 0) equals 1 only for negative shocks. With γ > 0, bad news raises next-period variance by (α + γ)·ε² instead of α·ε².",
                "incorrectExplanation": "Positive shocks enter only through α·ε²; the constant is ω and persistence is governed by β (overall α + β + γ/2 for symmetric innovations). The indicator term switches on γ only when the shock is negative."
            },
            "ro": {
                "title": "GJR-GARCH",
                "text": "În modelul GJR-GARCH, termenul γ·I(εₜ₋₁ < 0)·εₜ₋₁² surprinde:",
                "options": [
                    "Efectul șocurilor pozitive",
                    "Impactul suplimentar al șocurilor negative (leverage effect)",
                    "Constanta modelului",
                    "Persistența volatilității"
                ],
                "correctExplanation": "GJR-GARCH (Glosten, Jagannathan și Runkle, 1993): indicatorul I(εₜ₋₁ < 0) este 1 doar pentru șocurile negative. Cu γ > 0, veștile proaste cresc varianța din perioada următoare cu (α + γ)·ε², în loc de α·ε².",
                "incorrectExplanation": "Șocurile pozitive intră doar prin α·ε²; constanta este ω, iar persistența este dată de β (global, α + β + γ/2 pentru inovații simetrice). Termenul cu indicator îl activează pe γ doar cînd șocul este negativ."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "IGARCH",
                "text": "An IGARCH (integrated GARCH) model has the property that:",
                "options": [
                    "Volatility reverts quickly to its mean",
                    "α + β = 1, so volatility does not revert to a finite mean level",
                    "Volatility is constant",
                    "α + β < 0.5"
                ],
                "correctExplanation": "IGARCH: α + β = 1, i.e. unit persistence. Shocks to volatility do not die out in the forecasts, and the unconditional variance ω/(1 − 1) does not exist (it is infinite). The RiskMetrics EWMA is an IGARCH with ω = 0.",
                "incorrectExplanation": "Fast mean reversion corresponds to a small α + β, and constant volatility to α = β = 0; IGARCH is the opposite extreme. It is the volatility analogue of a unit root."
            },
            "ro": {
                "title": "IGARCH",
                "text": "Un model IGARCH (GARCH integrat) are proprietatea că:",
                "options": [
                    "Volatilitatea revine rapid la medie",
                    "α + β = 1, deci volatilitatea nu revine la un nivel mediu finit",
                    "Volatilitatea este constantă",
                    "α + β < 0,5"
                ],
                "correctExplanation": "IGARCH: α + β = 1, adică persistență unitară. Șocurile de volatilitate nu se sting în prognoze, iar varianța necondiționată ω/(1 − 1) nu există (este infinită). Modelul EWMA din RiskMetrics este un IGARCH cu ω = 0.",
                "incorrectExplanation": "Revenirea rapidă la medie corespunde unui α + β mic, iar volatilitatea constantă cazului α = β = 0; IGARCH este extrema opusă. El reprezintă, pentru volatilitate, analogul unei rădăcini unitare."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Standardised residuals",
                "text": "The standardised residuals of a GARCH model are:",
                "options": [
                    "εₜ (the raw residuals)",
                    "εₜ / σₜ (the residuals divided by the conditional volatility)",
                    "σₜ² (the conditional variance)",
                    "εₜ × σₜ"
                ],
                "correctExplanation": "zₜ = εₜ/σₜ should be i.i.d. with mean 0 and variance 1 if the model is correct (Normal or Student t, as assumed). Diagnostics: Ljung-Box on zₜ and zₜ² (no remaining ARCH) and Jarque-Bera or a QQ plot on zₜ.",
                "incorrectExplanation": "Raw residuals still contain the volatility clustering, the conditional variance is a fitted quantity rather than a residual, and multiplying by σₜ amplifies heteroskedasticity instead of removing it. Standardisation divides by the estimated σₜ."
            },
            "ro": {
                "title": "Reziduuri standardizate",
                "text": "Reziduurile standardizate ale unui model GARCH sînt:",
                "options": [
                    "εₜ (reziduurile brute)",
                    "εₜ / σₜ (reziduurile împărțite la volatilitatea condiționată)",
                    "σₜ² (varianța condiționată)",
                    "εₜ × σₜ"
                ],
                "correctExplanation": "Dacă modelul este corect, zₜ = εₜ/σₜ trebuie să fie i.i.d., cu media 0 și varianța 1 (Normale sau Student t, după ipoteza făcută). Diagnosticare: Ljung-Box pe zₜ și pe zₜ² (fără efecte ARCH rămase) și Jarque-Bera sau QQ plot pe zₜ.",
                "incorrectExplanation": "Reziduurile brute conțin încă volatility clustering, varianța condiționată este o mărime estimată, nu un reziduu, iar înmulțirea cu σₜ amplifică heteroscedasticitatea în loc să o elimine. Standardizarea înseamnă împărțirea la σₜ estimat."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "TGARCH",
                "text": "In its underlying idea, the TGARCH (threshold GARCH) model is closest to:",
                "options": [
                    "The standard GARCH",
                    "GJR-GARCH (both capture asymmetry)",
                    "ARCH(1)",
                    "The random walk"
                ],
                "correctExplanation": "TGARCH (Zakoian, 1994) models σₜ rather than σₜ², while GJR-GARCH models σₜ². Both use a threshold at zero to separate the impact of positive and negative shocks.",
                "incorrectExplanation": "Standard GARCH and ARCH(1) are symmetric in the sign of the shock, and the random walk is a model for the level of a series, not for its volatility. TGARCH and GJR-GARCH are related asymmetric models that capture the leverage effect through indicator functions."
            },
            "ro": {
                "title": "TGARCH",
                "text": "Ca idee de bază, modelul TGARCH (threshold GARCH) este cel mai apropiat de:",
                "options": [
                    "GARCH standard",
                    "GJR-GARCH (ambele surprind asimetria)",
                    "ARCH(1)",
                    "Mersul aleator"
                ],
                "correctExplanation": "TGARCH (Zakoian, 1994) modelează σₜ, nu σₜ², în timp ce GJR-GARCH modelează σₜ². Ambele folosesc un prag în zero pentru a separa impactul șocurilor pozitive de cel al șocurilor negative.",
                "incorrectExplanation": "GARCH standard și ARCH(1) sînt simetrice în raport cu semnul șocului, iar mersul aleator este un model pentru nivelul unei serii, nu pentru volatilitatea ei. TGARCH și GJR-GARCH sînt modele asimetrice înrudite, care surprind leverage effect prin funcții indicator."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Kurtosis of returns",
                "text": "Financial returns usually have:",
                "options": [
                    "Kurtosis = 3 (as for the Normal distribution)",
                    "Kurtosis < 3 (thin tails)",
                    "Kurtosis > 3 (fat tails, leptokurtic)",
                    "Kurtosis = 0"
                ],
                "correctExplanation": "Financial returns are leptokurtic: fat tails (kurtosis > 3, i.e. positive excess kurtosis) and a sharper peak than the Normal distribution. This stylised fact motivates Student t innovations in GARCH models.",
                "incorrectExplanation": "Kurtosis 3 is the Normal benchmark, which returns typically exceed; kurtosis below 3 would mean thin tails; kurtosis cannot be 0 for any non-degenerate distribution (it is at least 1). Excess kurtosis above 0 is one of the stylised facts of returns."
            },
            "ro": {
                "title": "Boltirea randamentelor",
                "text": "Randamentele financiare au de obicei:",
                "options": [
                    "Coeficient de boltire = 3 (ca la distribuția Normală)",
                    "Coeficient de boltire < 3 (cozi subțiri)",
                    "Coeficient de boltire > 3 (cozi groase, leptocurtică)",
                    "Coeficient de boltire = 0"
                ],
                "correctExplanation": "Randamentele financiare sînt leptocurtice: cozi groase (coeficient de boltire > 3, adică exces de boltire pozitiv) și un vîrf mai ascuțit decît la distribuția Normală. Acest fapt stilizat motivează inovațiile Student t în modelele GARCH.",
                "incorrectExplanation": "Valoarea 3 este reperul distribuției Normale, pe care randamentele îl depășesc de regulă; un coeficient sub 3 ar însemna cozi subțiri; coeficientul de boltire nu poate fi 0 pentru nicio distribuție nedegenerată (este cel puțin 1). Excesul de boltire pozitiv este unul dintre faptele stilizate ale randamentelor."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Volatility forecasts",
                "text": "In a covariance-stationary GARCH(1,1), long-horizon variance forecasts converge to:",
                "options": [
                    "Zero",
                    "The unconditional variance ω/(1 − α − β)",
                    "Infinity",
                    "The last observed variance"
                ],
                "correctExplanation": "σ²ₜ₊ₕ|ₜ − σ̄² = (α + β)ʰ⁻¹(σ²ₜ₊₁|ₜ − σ̄²), so σ²ₜ₊ₕ|ₜ → σ̄² = ω/(1 − α − β) as h → ∞. The closer α + β is to 1, the slower the convergence.",
                "incorrectExplanation": "Forecasts tend to zero or infinity only in degenerate or explosive cases, and staying at the last variance is the IGARCH (α + β = 1) behaviour. With α + β < 1, variance forecasts revert to the unconditional variance."
            },
            "ro": {
                "title": "Prognoza volatilității",
                "text": "Într-un GARCH(1,1) staționar în covarianță, prognozele varianței pe orizonturi lungi converg către:",
                "options": [
                    "Zero",
                    "Varianța necondiționată ω/(1 − α − β)",
                    "Infinit",
                    "Ultima varianță observată"
                ],
                "correctExplanation": "σ²ₜ₊ₕ|ₜ − σ̄² = (α + β)ʰ⁻¹(σ²ₜ₊₁|ₜ − σ̄²), deci σ²ₜ₊ₕ|ₜ → σ̄² = ω/(1 − α − β) cînd h → ∞. Cu cît α + β este mai aproape de 1, cu atît convergența este mai lentă.",
                "incorrectExplanation": "Prognozele tind la zero sau la infinit doar în cazuri degenerate sau explozive, iar menținerea ultimei varianțe este comportamentul IGARCH (α + β = 1). Cu α + β < 1, prognozele varianței revin la varianța necondiționată."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Estimating GARCH models",
                "text": "GARCH models are usually estimated by:",
                "options": [
                    "OLS (ordinary least squares)",
                    "Maximum likelihood (MLE)",
                    "Yule-Walker equations",
                    "A simple linear regression of returns on time"
                ],
                "correctExplanation": "MLE maximises the log-likelihood ℓ = −½ Σ[ln(σₜ²) + εₜ²/σₜ²] (Normal case, up to a constant). OLS does not apply because the conditional variance depends nonlinearly on the parameters through the recursion. Quasi-MLE (QMLE) with robust standard errors protects against a misspecified distribution.",
                "incorrectExplanation": "OLS and a regression on time ignore the variance equation, and Yule-Walker equations are a moment method for AR models of the mean. GARCH requires MLE (or QMLE) because of the nonlinear recursion for σₜ²."
            },
            "ro": {
                "title": "Estimarea modelelor GARCH",
                "text": "Modelele GARCH se estimează de obicei prin:",
                "options": [
                    "OLS (metoda celor mai mici pătrate)",
                    "Metoda verosimilității maxime (MLE)",
                    "Ecuațiile Yule-Walker",
                    "O regresie liniară simplă a randamentelor în funcție de timp"
                ],
                "correctExplanation": "MLE maximizează log-verosimilitatea ℓ = −½ Σ[ln(σₜ²) + εₜ²/σₜ²] (cazul Normal, pînă la o constantă). OLS nu se poate aplica, deoarece varianța condiționată depinde neliniar de parametri prin recursivitate. Quasi-MLE (QMLE), cu erori standard robuste, protejează împotriva unei distribuții greșit specificate.",
                "incorrectExplanation": "OLS și regresia în funcție de timp ignoră ecuația varianței, iar ecuațiile Yule-Walker sînt o metodă a momentelor pentru modelele AR ale mediei. GARCH necesită MLE (sau QMLE) din cauza recursivității neliniare a lui σₜ²."
            }
        }
    ]
};
