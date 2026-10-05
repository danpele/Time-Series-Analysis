// ============================================================
// Chapter 12 quiz bank: Spectral analysis (EN + RO)
// 24 questions, 20 drawn per attempt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['spectral'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 2,
            "en": {
                "title": "Period and frequency",
                "text": "Monthly data show a spectral peak at $\\nu = 1/12$ cycles per month. What is the period of the cycle?",
                "options": [
                    "1/12 of a month",
                    "6 months, because only frequencies up to 1/2 are visible",
                    "12 months: an annual cycle",
                    "12 years"
                ],
                "correctExplanation": "The period is the inverse of the frequency: $1/\\nu = 12$ observations, i.e. 12 months.",
                "incorrectExplanation": "The period is $1/\\nu$ measured in observations; with monthly data $1/(1/12) = 12$ months, an annual cycle."
            },
            "ro": {
                "title": "Perioadă și frecvență",
                "text": "Niște date lunare au un vîrf spectral la $\\nu = 1/12$ cicluri pe lună. Care este perioada ciclului?",
                "options": [
                    "1/12 dintr-o lună",
                    "6 luni, deoarece sînt vizibile doar frecvențele pînă la 1/2",
                    "12 luni: un ciclu anual",
                    "12 ani"
                ],
                "correctExplanation": "Perioada este inversul frecvenței: $1/\\nu = 12$ observații, adică 12 luni.",
                "incorrectExplanation": "Perioada este $1/\\nu$, măsurată în observații; pentru date lunare $1/(1/12) = 12$ luni, un ciclu anual."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Fourier frequencies",
                "text": "For a sample of $T = 200$ observations, which frequencies are Fourier frequencies?",
                "options": [
                    "$\\nu_j = j/100$, $j = 0, \\ldots, 200$",
                    "Any frequency between 0 and 1",
                    "$\\nu_j = 2\\pi j$, $j = 1, \\ldots, 200$",
                    "$\\nu_j = j/200$, $j = 0, 1, \\ldots, 100$"
                ],
                "correctExplanation": "The Fourier frequencies are $j/T$, cycles that fit a whole number of times in the sample, up to the Nyquist frequency $1/2$.",
                "incorrectExplanation": "The Fourier frequencies are $\\nu_j = j/T$ for $j$ up to $T/2$: here $j/200$, $j = 0, \\ldots, 100$."
            },
            "ro": {
                "title": "Frecvențele Fourier",
                "text": "Pentru un eșantion de $T = 200$ de observații, care sînt frecvențele Fourier?",
                "options": [
                    "$\\nu_j = j/100$, $j = 0, \\ldots, 200$",
                    "Orice frecvență între 0 și 1",
                    "$\\nu_j = 2\\pi j$, $j = 1, \\ldots, 200$",
                    "$\\nu_j = j/200$, $j = 0, 1, \\ldots, 100$"
                ],
                "correctExplanation": "Frecvențele Fourier sînt $j/T$, ciclurile care încap de un număr întreg de ori în eșantion, pînă la frecvența Nyquist $1/2$.",
                "incorrectExplanation": "Frecvențele Fourier sînt $\\nu_j = j/T$, cu $j$ pînă la $T/2$: aici $j/200$, $j = 0, \\ldots, 100$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Nyquist frequency",
                "text": "What is the highest frequency that can be identified in a series observed at equal intervals?",
                "options": [
                    "$\\nu = 1$ cycle per observation",
                    "$\\nu = 1/T$",
                    "$\\nu = 1/2$ cycle per observation",
                    "There is no upper limit"
                ],
                "correctExplanation": "A cycle needs at least two observations: the Nyquist frequency is $1/2$ cycle per observation (angular frequency $\\pi$).",
                "incorrectExplanation": "With one observation per period, a cycle shorter than two observations cannot be seen; faster cycles are aliased to frequencies below $1/2$."
            },
            "ro": {
                "title": "Frecvența Nyquist",
                "text": "Care este cea mai mare frecvență care poate fi identificată într-o serie observată la intervale egale?",
                "options": [
                    "$\\nu = 1$ ciclu pe observație",
                    "$\\nu = 1/T$",
                    "$\\nu = 1/2$ cicluri pe observație",
                    "Nu există o limită superioară"
                ],
                "correctExplanation": "Un ciclu are nevoie de cel puțin două observații: frecvența Nyquist este $1/2$ cicluri pe observație (frecvența unghiulară $\\pi$).",
                "incorrectExplanation": "Cu o observație pe perioadă, un ciclu mai scurt de două observații nu poate fi văzut; ciclurile mai rapide apar ca aliasuri la frecvențe sub $1/2$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Aliasing",
                "text": "A cycle of 4 months is observed only once a quarter. How does it appear in the quarterly data?",
                "options": [
                    "As a cycle of 4 months",
                    "As a cycle of 12 months (4 quarters)",
                    "It disappears completely",
                    "As a cycle of 2 quarters"
                ],
                "correctExplanation": "Per quarter its frequency is $3/4$, above $1/2$; it folds to $|3/4 - 1| = 1/4$ cycles per quarter, a period of 4 quarters = 12 months.",
                "incorrectExplanation": "The true frequency $3/4$ cycles per quarter exceeds the Nyquist frequency and is folded to $1/4$: a false 12-month cycle."
            },
            "ro": {
                "title": "Aliasing",
                "text": "Un ciclu de 4 luni este observat doar o dată pe trimestru. Cum apare el în datele trimestriale?",
                "options": [
                    "Ca un ciclu de 4 luni",
                    "Ca un ciclu de 12 luni (4 trimestre)",
                    "Dispare complet",
                    "Ca un ciclu de 2 trimestre"
                ],
                "correctExplanation": "Pe trimestru frecvența lui este $3/4$, peste $1/2$; ea se pliază la $|3/4 - 1| = 1/4$ cicluri pe trimestru, o perioadă de 4 trimestre = 12 luni.",
                "incorrectExplanation": "Frecvența adevărată, $3/4$ cicluri pe trimestru, depășește frecvența Nyquist și se pliază la $1/4$: un ciclu fals de 12 luni."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Parseval's identity",
                "text": "What does Parseval's identity say about the periodogram ordinates?",
                "options": [
                    "They add up to 1",
                    "They add up to $T$ times the sample variance: the variance is split by frequency",
                    "They are all equal for a stationary series",
                    "Their maximum equals the variance"
                ],
                "correctExplanation": "$\\sum_t (x_t - \\bar x)^2 = \\sum_j |d(\\nu_j)|^2$: the periodogram decomposes the total sum of squares over the Fourier frequencies.",
                "incorrectExplanation": "Parseval: the sum of the periodogram ordinates equals the sum of squared deviations, i.e. $T$ times the sample variance."
            },
            "ro": {
                "title": "Identitatea lui Parseval",
                "text": "Ce spune identitatea lui Parseval despre ordonatele periodogramei?",
                "options": [
                    "Suma lor este 1",
                    "Suma lor este $T$ înmulțit cu varianța de selecție: varianța se împarte pe frecvențe",
                    "Sînt toate egale pentru o serie staționară",
                    "Maximul lor este egal cu varianța"
                ],
                "correctExplanation": "$\\sum_t (x_t - \\bar x)^2 = \\sum_j |d(\\nu_j)|^2$: periodograma descompune suma totală a pătratelor pe frecvențele Fourier.",
                "incorrectExplanation": "Parseval: suma ordonatelor periodogramei este egală cu suma pătratelor abaterilor, adică $T$ înmulțit cu varianța de selecție."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Spectral density and variance",
                "text": "For a stationary series with spectral density $f(\\nu) = \\sum_h \\gamma(h) e^{-2\\pi i \\nu h}$, what is $\\int_{-1/2}^{1/2} f(\\nu)\\,d\\nu$?",
                "options": [
                    "The mean of the series",
                    "The variance $\\gamma(0)$",
                    "Always 1",
                    "The first autocorrelation $\\rho(1)$"
                ],
                "correctExplanation": "From the inverse relation $\\gamma(h) = \\int f(\\nu) e^{2\\pi i \\nu h} d\\nu$ with $h = 0$: the area under the spectrum is the variance.",
                "incorrectExplanation": "Setting $h = 0$ in $\\gamma(h) = \\int f(\\nu) e^{2\\pi i \\nu h} d\\nu$ gives $\\gamma(0)$: the spectrum decomposes the variance."
            },
            "ro": {
                "title": "Densitatea spectrală și varianța",
                "text": "Pentru o serie staționară cu densitatea spectrală $f(\\nu) = \\sum_h \\gamma(h) e^{-2\\pi i \\nu h}$, cît este $\\int_{-1/2}^{1/2} f(\\nu)\\,d\\nu$?",
                "options": [
                    "Media seriei",
                    "Varianța $\\gamma(0)$",
                    "Întotdeauna 1",
                    "Prima autocorelație $\\rho(1)$"
                ],
                "correctExplanation": "Din relația inversă $\\gamma(h) = \\int f(\\nu) e^{2\\pi i \\nu h} d\\nu$, pentru $h = 0$: aria de sub spectru este varianța.",
                "incorrectExplanation": "Pentru $h = 0$, $\\gamma(h) = \\int f(\\nu) e^{2\\pi i \\nu h} d\\nu$ dă $\\gamma(0)$: spectrul descompune varianța."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Spectrum of white noise",
                "text": "What is the spectral density of white noise with variance $\\sigma^2$?",
                "options": [
                    "Decreasing from $\\nu = 0$",
                    "Constant: $f(\\nu) = \\sigma^2$ at every frequency",
                    "A single peak at $\\nu = 1/2$",
                    "Zero everywhere except at $\\nu = 0$"
                ],
                "correctExplanation": "All autocovariances except $\\gamma(0) = \\sigma^2$ are zero, so $f(\\nu) = \\sigma^2$: a flat spectrum, like white light.",
                "incorrectExplanation": "With $\\gamma(h) = 0$ for $h \\neq 0$, the sum defining $f$ has a single term: $f(\\nu) = \\sigma^2$ at all frequencies."
            },
            "ro": {
                "title": "Spectrul zgomotului alb",
                "text": "Care este densitatea spectrală a unui zgomot alb cu varianța $\\sigma^2$?",
                "options": [
                    "Descrescătoare de la $\\nu = 0$",
                    "Constantă: $f(\\nu) = \\sigma^2$ la orice frecvență",
                    "Un singur vîrf la $\\nu = 1/2$",
                    "Zero peste tot, cu excepția lui $\\nu = 0$"
                ],
                "correctExplanation": "Toate autocovarianțele, în afară de $\\gamma(0) = \\sigma^2$, sînt nule, deci $f(\\nu) = \\sigma^2$: un spectru plat, ca lumina albă.",
                "incorrectExplanation": "Cu $\\gamma(h) = 0$ pentru $h \\neq 0$, suma care definește $f$ are un singur termen: $f(\\nu) = \\sigma^2$ la toate frecvențele."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "AR(1) spectrum",
                "text": "An AR(1) with $\\phi = 0.6$ and $\\sigma^2 = 1$ has $f(\\nu) = 1/(1 - 1.2\\cos(2\\pi\\nu) + 0.36)$. What is $f(0)$?",
                "options": [
                    "6.25",
                    "0.39",
                    "1",
                    "2.5"
                ],
                "correctExplanation": "$f(0) = 1/(1 - 0.6)^2 = 1/0.16 = 6.25$; at $\\nu = 1/2$ it is only $1/1.6^2 = 0.39$: power at low frequencies.",
                "incorrectExplanation": "At $\\nu = 0$, $\\cos 0 = 1$ and the denominator is $(1 - \\phi)^2 = 0.16$, so $f(0) = 6.25$; 0.39 is the value at $\\nu = 1/2$."
            },
            "ro": {
                "title": "Spectrul AR(1)",
                "text": "Un AR(1) cu $\\phi = 0{,}6$ și $\\sigma^2 = 1$ are $f(\\nu) = 1/(1 - 1{,}2\\cos(2\\pi\\nu) + 0{,}36)$. Cît este $f(0)$?",
                "options": [
                    "6,25",
                    "0,39",
                    "1",
                    "2,5"
                ],
                "correctExplanation": "$f(0) = 1/(1 - 0{,}6)^2 = 1/0{,}16 = 6{,}25$; la $\\nu = 1/2$ este doar $1/1{,}6^2 = 0{,}39$: puterea este la frecvențele joase.",
                "incorrectExplanation": "La $\\nu = 0$, $\\cos 0 = 1$, iar numitorul este $(1 - \\phi)^2 = 0{,}16$, deci $f(0) = 6{,}25$; 0,39 este valoarea la $\\nu = 1/2$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Negative autocorrelation",
                "text": "Where is the power of an AR(1) process with $\\phi = -0.6$?",
                "options": [
                    "At low frequencies, near $\\nu = 0$",
                    "Spread evenly over all frequencies",
                    "At high frequencies, near $\\nu = 1/2$: the series zig-zags",
                    "In a peak inside $(0, 1/2)$"
                ],
                "correctExplanation": "With $\\phi < 0$ the denominator $1 - 2\\phi\\cos(2\\pi\\nu) + \\phi^2$ is smallest at $\\nu = 1/2$: consecutive values alternate in sign.",
                "incorrectExplanation": "A negative $\\phi$ mirrors the AR(1) spectrum: $f(1/2) = 1/(1 + \\phi)^2$ is the maximum, the signature of alternating values."
            },
            "ro": {
                "title": "Autocorelația negativă",
                "text": "Unde se află puterea unui proces AR(1) cu $\\phi = -0{,}6$?",
                "options": [
                    "La frecvențele joase, lîngă $\\nu = 0$",
                    "Distribuită uniform pe toate frecvențele",
                    "La frecvențele înalte, lîngă $\\nu = 1/2$: seria evoluează în zigzag",
                    "Într-un vîrf din interiorul intervalului $(0, 1/2)$"
                ],
                "correctExplanation": "Pentru $\\phi < 0$ numitorul $1 - 2\\phi\\cos(2\\pi\\nu) + \\phi^2$ este minim la $\\nu = 1/2$: valorile consecutive alternează ca semn.",
                "incorrectExplanation": "Un $\\phi$ negativ oglindește spectrul AR(1): $f(1/2) = 1/(1 + \\phi)^2$ este maximul, semnul valorilor alternante."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Pseudo-cycles",
                "text": "Which model can produce a spectral peak strictly inside $(0, 1/2)$, i.e. a pseudo-cycle?",
                "options": [
                    "An AR(1) with $\\phi > 0$",
                    "An MA(1)",
                    "White noise",
                    "An AR(2) with complex roots"
                ],
                "correctExplanation": "Complex roots give a damped oscillating ACF and a spectral peak at $\\cos(2\\pi\\nu^*) = \\phi_1(\\phi_2 - 1)/(4\\phi_2)$; Yule (1927) used this for sunspots.",
                "incorrectExplanation": "AR(1) and MA(1) spectra are monotone in $\\nu$ and white noise is flat; only an AR(2) with complex roots (or higher order) gives an interior peak."
            },
            "ro": {
                "title": "Pseudo-ciclurile",
                "text": "Care model poate produce un vîrf spectral strict în interiorul intervalului $(0, 1/2)$, adică un pseudo-ciclu?",
                "options": [
                    "Un AR(1) cu $\\phi > 0$",
                    "Un MA(1)",
                    "Zgomotul alb",
                    "Un AR(2) cu rădăcini complexe"
                ],
                "correctExplanation": "Rădăcinile complexe dau o ACF oscilantă amortizată și un vîrf spectral la $\\cos(2\\pi\\nu^*) = \\phi_1(\\phi_2 - 1)/(4\\phi_2)$; Yule (1927) a folosit acest model pentru petele solare.",
                "incorrectExplanation": "Spectrele AR(1) și MA(1) sînt monotone în $\\nu$, iar zgomotul alb are spectrul plat; doar un AR(2) cu rădăcini complexe (sau de ordin mai mare) dă un vîrf interior."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Periodogram by hand",
                "text": "The deviations from the mean of a series are $(2, 0, -2, 0)$. Where is all the periodogram power?",
                "options": [
                    "At $\\nu = 1/2$, a period of 2 observations",
                    "At $\\nu = 1/4$, a period of 4 observations",
                    "At $\\nu = 0$",
                    "Spread equally over $\\nu = 1/4$ and $\\nu = 1/2$"
                ],
                "correctExplanation": "$|d(1/4)|^2 = 4$ and $|d(1/2)|^2 = 0$: the series is the cycle $2\\cos(2\\pi t/4)$ with period 4.",
                "incorrectExplanation": "At $\\nu = 1/2$ the sum $\\sum_t y_t (-1)^t$ is zero; the whole sum of squares, 8, sits at $\\nu = 1/4$ (and its mirror $3/4$)."
            },
            "ro": {
                "title": "Periodograma calculată de mînă",
                "text": "Abaterile de la medie ale unei serii sînt $(2, 0, -2, 0)$. Unde se află toată puterea periodogramei?",
                "options": [
                    "La $\\nu = 1/2$, o perioadă de 2 observații",
                    "La $\\nu = 1/4$, o perioadă de 4 observații",
                    "La $\\nu = 0$",
                    "Împărțită în mod egal între $\\nu = 1/4$ și $\\nu = 1/2$"
                ],
                "correctExplanation": "$|d(1/4)|^2 = 4$ și $|d(1/2)|^2 = 0$: seria este ciclul $2\\cos(2\\pi t/4)$, cu perioada 4.",
                "incorrectExplanation": "La $\\nu = 1/2$ suma $\\sum_t y_t (-1)^t$ este zero; întreaga sumă a pătratelor, 8, se află la $\\nu = 1/4$ (și la oglinda ei, $3/4$)."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Inconsistency of the periodogram",
                "text": "Why is the raw periodogram not a consistent estimator of the spectral density?",
                "options": [
                    "It is biased even in large samples",
                    "It is only defined for white noise",
                    "Its variance stays about $f(\\nu)^2$ however large $T$ is",
                    "It needs the true autocovariances"
                ],
                "correctExplanation": "$2I(\\nu_j)/f(\\nu_j) \\approx \\chi^2_2$: unbiased, but with variance $f^2$ that does not shrink; more data only add ordinates.",
                "incorrectExplanation": "The periodogram is asymptotically unbiased; the problem is its variance, about $f(\\nu)^2$, which does not decrease with $T$."
            },
            "ro": {
                "title": "Inconsistența periodogramei",
                "text": "De ce periodograma brută nu este un estimator consistent al densității spectrale?",
                "options": [
                    "Este deplasată chiar și în eșantioane mari",
                    "Este definită doar pentru zgomotul alb",
                    "Varianța ei rămîne aproximativ $f(\\nu)^2$, oricît de mare ar fi $T$",
                    "Are nevoie de autocovarianțele adevărate"
                ],
                "correctExplanation": "$2I(\\nu_j)/f(\\nu_j) \\approx \\chi^2_2$: nedeplasată, dar cu varianța $f^2$, care nu scade; mai multe date adaugă doar ordonate.",
                "incorrectExplanation": "Periodograma este asimptotic nedeplasată; problema este varianța ei, aproximativ $f(\\nu)^2$, care nu scade cu $T$."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Fisher's g test",
                "text": "What does Fisher's $g$ statistic compare?",
                "options": [
                    "The periodogram with an AR(1) spectrum",
                    "Two periodograms of different series",
                    "The largest periodogram ordinate with the sum of all ordinates",
                    "The mean with the variance of the series"
                ],
                "correctExplanation": "$g = \\max_j I(\\nu_j) / \\sum_j I(\\nu_j)$; under white noise each ordinate is about $1/m$ of the total, so a large $g$ signals a hidden cycle.",
                "incorrectExplanation": "Fisher (1929): $g$ is the share of the largest ordinate in the total; its null hypothesis is Gaussian white noise."
            },
            "ro": {
                "title": "Testul g al lui Fisher",
                "text": "Ce compară statistica $g$ a lui Fisher?",
                "options": [
                    "Periodograma cu spectrul unui AR(1)",
                    "Două periodograme ale unor serii diferite",
                    "Cea mai mare ordonată a periodogramei cu suma tuturor ordonatelor",
                    "Media cu varianța seriei"
                ],
                "correctExplanation": "$g = \\max_j I(\\nu_j) / \\sum_j I(\\nu_j)$; sub ipoteza de zgomot alb fiecare ordonată reprezintă aproximativ $1/m$ din total, deci un $g$ mare semnalează un ciclu ascuns.",
                "incorrectExplanation": "Fisher (1929): $g$ este ponderea celei mai mari ordonate în total; ipoteza nulă este zgomotul alb gaussian."
            }
        },
        {
            "correct": 2,
            "en": {
                "title": "Spectral leakage",
                "text": "What causes spectral leakage, and what reduces it?",
                "options": [
                    "Too many observations; shortening the sample reduces it",
                    "Smoothing the periodogram; using the raw periodogram reduces it",
                    "A cycle that does not fit a whole number of times in the sample; a taper (e.g. Hann window) reduces it",
                    "Removing the mean; keeping the mean reduces it"
                ],
                "correctExplanation": "The cut at both ends creates a jump; its power spreads to all frequencies. A window that goes smoothly to 0 at the ends suppresses it.",
                "incorrectExplanation": "Leakage comes from truncating a cycle that is not at a Fourier frequency; tapering the data before the DFT reduces it at the cost of a slightly wider peak."
            },
            "ro": {
                "title": "Scurgerea spectrală",
                "text": "Ce cauzează scurgerea spectrală și ce o reduce?",
                "options": [
                    "Prea multe observații; scurtarea eșantionului o reduce",
                    "Netezirea periodogramei; folosirea periodogramei brute o reduce",
                    "Un ciclu care nu încape de un număr întreg de ori în eșantion; o fereastră de atenuare (de exemplu Hann) o reduce",
                    "Eliminarea mediei; păstrarea mediei o reduce"
                ],
                "correctExplanation": "Tăierea la cele două capete creează un salt; puterea lui se împrăștie la toate frecvențele. O fereastră care scade lin spre 0 la capete o atenuează.",
                "incorrectExplanation": "Scurgerea provine din trunchierea unui ciclu care nu se află la o frecvență Fourier; atenuarea datelor înainte de DFT o reduce, cu prețul unui vîrf puțin mai lat."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Daniell smoothing",
                "text": "The Daniell smoother averages $L = 2m + 1$ neighbouring ordinates. What happens when $L$ increases?",
                "options": [
                    "Both the variance and the bias fall",
                    "The variance rises and the bias falls",
                    "Nothing changes, because the ordinates are independent",
                    "The variance falls (about $f^2/L$) but narrow peaks are flattened"
                ],
                "correctExplanation": "Averaging $L$ nearly independent ordinates divides the variance by about $L$; a wider band also mixes in neighbouring frequencies (bias).",
                "incorrectExplanation": "This is the bias-variance trade-off of the bandwidth $B = L/T$: less noise, but peaks become lower and wider."
            },
            "ro": {
                "title": "Netezirea Daniell",
                "text": "Netezirea Daniell face media a $L = 2m + 1$ ordonate vecine. Ce se întîmplă cînd $L$ crește?",
                "options": [
                    "Scad atît varianța, cît și deplasarea",
                    "Varianța crește, iar deplasarea scade",
                    "Nimic nu se schimbă, deoarece ordonatele sînt independente",
                    "Varianța scade (aproximativ $f^2/L$), dar vîrfurile înguste sînt aplatizate"
                ],
                "correctExplanation": "Media a $L$ ordonate aproape independente împarte varianța la aproximativ $L$; o bandă mai largă amestecă și frecvențele vecine (deplasare).",
                "incorrectExplanation": "Acesta este compromisul deplasare–varianță al lățimii de bandă $B = L/T$: mai puțin zgomot, dar vîrfuri mai joase și mai late."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Welch's method",
                "text": "How does Welch's method estimate the spectrum?",
                "options": [
                    "It fits an AR model by maximum likelihood",
                    "It takes the periodogram of the whole series without changes",
                    "It averages the series over segments and takes one periodogram",
                    "It averages the periodograms of tapered, overlapping segments of the series"
                ],
                "correctExplanation": "Welch (1967): split into $K$ segments, taper each, compute their periodograms and average them; the segment length sets the resolution.",
                "incorrectExplanation": "Welch averages segment periodograms, not the data: the variance falls roughly by the number of segments."
            },
            "ro": {
                "title": "Metoda Welch",
                "text": "Cum estimează metoda Welch spectrul?",
                "options": [
                    "Estimează un model AR prin verosimilitate maximă",
                    "Calculează periodograma întregii serii, fără modificări",
                    "Face media seriei pe segmente și calculează o singură periodogramă",
                    "Face media periodogramelor unor segmente ale seriei, atenuate și suprapuse"
                ],
                "correctExplanation": "Welch (1967): împărțirea în $K$ segmente, atenuarea fiecăruia, calculul periodogramelor și media lor; lungimea segmentului fixează rezoluția.",
                "incorrectExplanation": "Welch face media periodogramelor segmentelor, nu a datelor: varianța scade aproximativ de atîtea ori cîte segmente există."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Confidence band",
                "text": "A Daniell estimate uses $L = 5$ ordinates ($df = 10$), with $\\chi^2_{10}(0.025) = 3.25$ and $\\chi^2_{10}(0.975) = 20.48$. What is the 95% interval for $f(\\nu)$?",
                "options": [
                    "From $0.49\\,\\hat f$ to $3.08\\,\\hat f$",
                    "From $0.95\\,\\hat f$ to $1.05\\,\\hat f$",
                    "$\\hat f \\pm 1.96\\,\\hat f/\\sqrt{5}$",
                    "From $3.25\\,\\hat f$ to $20.48\\,\\hat f$"
                ],
                "correctExplanation": "$[df\\,\\hat f/\\chi^2_{df}(0.975),\\ df\\,\\hat f/\\chi^2_{df}(0.025)] = [10/20.48,\\ 10/3.25]\\,\\hat f = [0.49,\\ 3.08]\\,\\hat f$: asymmetric.",
                "incorrectExplanation": "Since $10\\,\\hat f/f \\approx \\chi^2_{10}$, the bounds are $10/20.48 = 0.49$ and $10/3.25 = 3.08$ times $\\hat f$."
            },
            "ro": {
                "title": "Banda de încredere",
                "text": "O estimare Daniell folosește $L = 5$ ordonate ($df = 10$), cu $\\chi^2_{10}(0{,}025) = 3{,}25$ și $\\chi^2_{10}(0{,}975) = 20{,}48$. Care este intervalul de 95% pentru $f(\\nu)$?",
                "options": [
                    "De la $0{,}49\\,\\hat f$ la $3{,}08\\,\\hat f$",
                    "De la $0{,}95\\,\\hat f$ la $1{,}05\\,\\hat f$",
                    "$\\hat f \\pm 1{,}96\\,\\hat f/\\sqrt{5}$",
                    "De la $3{,}25\\,\\hat f$ la $20{,}48\\,\\hat f$"
                ],
                "correctExplanation": "$[df\\,\\hat f/\\chi^2_{df}(0{,}975);\\ df\\,\\hat f/\\chi^2_{df}(0{,}025)] = [10/20{,}48;\\ 10/3{,}25]\\,\\hat f = [0{,}49;\\ 3{,}08]\\,\\hat f$: asimetric.",
                "incorrectExplanation": "Deoarece $10\\,\\hat f/f \\approx \\chi^2_{10}$, limitele sînt $10/20{,}48 = 0{,}49$ și $10/3{,}25 = 3{,}08$ ori $\\hat f$."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Typical spectral shape",
                "text": "Granger (1966) described the 'typical spectral shape' of economic levels such as GDP. What is it?",
                "options": [
                    "Power falling steeply from frequency zero: trends dominate",
                    "A flat spectrum, like white noise",
                    "A single sharp peak at the business-cycle frequency",
                    "Power concentrated at the highest frequencies"
                ],
                "correctExplanation": "Trends and near unit roots put almost all the variance at the lowest frequencies; cycles become visible only after differencing or detrending.",
                "incorrectExplanation": "Levels of economic series are dominated by low frequencies; that is why spectra are computed on growth rates or detrended cycles."
            },
            "ro": {
                "title": "Forma spectrală tipică",
                "text": "Granger (1966) a descris „forma spectrală tipică” a nivelurilor economice, precum PIB-ul. Care este aceasta?",
                "options": [
                    "Puterea scade abrupt de la frecvența zero: trendurile domină",
                    "Un spectru plat, ca al zgomotului alb",
                    "Un singur vîrf ascuțit la frecvența ciclului economic",
                    "Puterea concentrată la frecvențele cele mai înalte"
                ],
                "correctExplanation": "Trendurile și rădăcinile aproape unitare pun aproape toată varianța la frecvențele cele mai joase; ciclurile devin vizibile doar după diferențiere sau eliminarea trendului.",
                "incorrectExplanation": "Nivelurile seriilor economice sînt dominate de frecvențele joase; de aceea spectrele se calculează pe rate de creștere sau pe cicluri fără trend."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Business-cycle band",
                "text": "For quarterly data, which frequency band corresponds to business cycles of 1.5 to 8 years?",
                "options": [
                    "$1/32 \\le \\nu \\le 1/6$ cycles per quarter",
                    "$1/8 \\le \\nu \\le 1/1.5$ cycles per quarter",
                    "$6 \\le \\nu \\le 32$ cycles per quarter",
                    "$0 \\le \\nu \\le 1/2$ cycles per quarter"
                ],
                "correctExplanation": "1.5 to 8 years are 6 to 32 quarters; frequency is the inverse of the period, so $1/32 \\le \\nu \\le 1/6$ (Baxter and King, 1999).",
                "incorrectExplanation": "Convert the periods to quarters (6 and 32) and invert them: the band runs from $1/32$ to $1/6$ cycles per quarter."
            },
            "ro": {
                "title": "Banda ciclului economic",
                "text": "Pentru date trimestriale, ce bandă de frecvențe corespunde ciclurilor economice de 1,5 pînă la 8 ani?",
                "options": [
                    "$1/32 \\le \\nu \\le 1/6$ cicluri pe trimestru",
                    "$1/8 \\le \\nu \\le 1/1{,}5$ cicluri pe trimestru",
                    "$6 \\le \\nu \\le 32$ cicluri pe trimestru",
                    "$0 \\le \\nu \\le 1/2$ cicluri pe trimestru"
                ],
                "correctExplanation": "1,5 pînă la 8 ani înseamnă 6 pînă la 32 de trimestre; frecvența este inversul perioadei, deci $1/32 \\le \\nu \\le 1/6$ (Baxter și King, 1999).",
                "incorrectExplanation": "Transformați perioadele în trimestre (6 și 32) și inversați-le: banda este de la $1/32$ la $1/6$ cicluri pe trimestru."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Electricity load",
                "text": "The spectrum of hourly electricity load shows sharp peaks at 24, 12 and 8 hours. Why three peaks?",
                "options": [
                    "There are three different cycles caused by three power plants",
                    "Aliasing of the weekly cycle",
                    "Leakage from the annual cycle",
                    "The daily cycle is not a sine wave, so it has harmonics at multiples of its frequency"
                ],
                "correctExplanation": "A non-sinusoidal daily profile (morning and evening peaks) is a sum of cosines at $1/24$, $2/24$, $3/24$, ...: periods 24, 12, 8 hours.",
                "incorrectExplanation": "Periods of 12 and 8 hours are harmonics ($2/24$ and $3/24$) of the daily cycle; the weekly cycle appears separately at 168 hours."
            },
            "ro": {
                "title": "Consumul de energie electrică",
                "text": "Spectrul consumului orar de energie electrică are vîrfuri ascuțite la 24, 12 și 8 ore. De ce trei vîrfuri?",
                "options": [
                    "Există trei cicluri diferite, produse de trei centrale",
                    "Aliasing-ul ciclului săptămînal",
                    "Scurgerea din ciclul anual",
                    "Ciclul zilnic nu este o sinusoidă, deci are armonice la multiplii frecvenței lui"
                ],
                "correctExplanation": "Un profil zilnic nesinusoidal (vîrfuri de dimineață și de seară) este o sumă de cosinusuri la $1/24$, $2/24$, $3/24$, ...: perioade de 24, 12 și 8 ore.",
                "incorrectExplanation": "Perioadele de 12 și 8 ore sînt armonice ($2/24$ și $3/24$) ale ciclului zilnic; ciclul săptămînal apare separat, la 168 de ore."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "The pole at zero",
                "text": "Near frequency zero the log-log periodogram of a series is a straight line with slope $-0.8$. What does it suggest?",
                "options": [
                    "Long memory with $d \\approx 0.4$",
                    "Short memory with $d = -0.8$",
                    "A unit root, $d = 1$",
                    "White noise"
                ],
                "correctExplanation": "Long memory gives $f(\\nu) \\approx c\\,\\nu^{-2d}$, a log-log slope of $-2d$; $-2d = -0.8$ gives $d = 0.4$ (GPH, Chapter 8).",
                "incorrectExplanation": "The slope of the log spectrum against log frequency is $-2d$; white noise and short memory are flat near zero."
            },
            "ro": {
                "title": "Polul de la zero",
                "text": "În apropierea frecvenței zero, periodograma log-log a unei serii este o dreaptă cu panta $-0{,}8$. Ce sugerează aceasta?",
                "options": [
                    "Memorie lungă, cu $d \\approx 0{,}4$",
                    "Memorie scurtă, cu $d = -0{,}8$",
                    "O rădăcină unitară, $d = 1$",
                    "Zgomot alb"
                ],
                "correctExplanation": "Memoria lungă dă $f(\\nu) \\approx c\\,\\nu^{-2d}$, o pantă log-log de $-2d$; $-2d = -0{,}8$ dă $d = 0{,}4$ (GPH, Capitolul 8).",
                "incorrectExplanation": "Panta logaritmului spectrului în funcție de logaritmul frecvenței este $-2d$; zgomotul alb și memoria scurtă au spectrul plat lîngă zero."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Gain of the first difference",
                "text": "The first difference $y_t = x_t - x_{t-1}$ has squared gain $4\\sin^2(\\pi\\nu)$. What does it do?",
                "options": [
                    "It leaves the spectrum unchanged",
                    "It removes power near $\\nu = 0$ and amplifies the highest frequencies",
                    "It amplifies low frequencies",
                    "It removes only the seasonal frequencies"
                ],
                "correctExplanation": "The gain is 0 at $\\nu = 0$ (trend removed) and 4 at $\\nu = 1/2$: differenced series look noisier, and slow cycles are weakened.",
                "incorrectExplanation": "Since $f_y = |A(\\nu)|^2 f_x$, a gain near 0 at low frequencies and 4 at $\\nu = 1/2$ shifts the power towards fast fluctuations."
            },
            "ro": {
                "title": "Cîștigul primei diferențe",
                "text": "Prima diferență $y_t = x_t - x_{t-1}$ are cîștigul pătratic $4\\sin^2(\\pi\\nu)$. Ce efect are?",
                "options": [
                    "Lasă spectrul neschimbat",
                    "Elimină puterea din apropierea lui $\\nu = 0$ și amplifică frecvențele cele mai înalte",
                    "Amplifică frecvențele joase",
                    "Elimină doar frecvențele sezoniere"
                ],
                "correctExplanation": "Cîștigul este 0 la $\\nu = 0$ (trendul este eliminat) și 4 la $\\nu = 1/2$: seriile diferențiate par mai zgomotoase, iar ciclurile lente sînt slăbite.",
                "incorrectExplanation": "Deoarece $f_y = |A(\\nu)|^2 f_x$, un cîștig aproape nul la frecvențele joase și egal cu 4 la $\\nu = 1/2$ mută puterea spre fluctuațiile rapide."
            }
        },
        {
            "correct": 0,
            "en": {
                "title": "Coherence",
                "text": "The squared coherence of two series is 0.9 at 5-year cycles and 0.1 at monthly cycles. What does it mean?",
                "options": [
                    "Their slow cycles move together, their short-run noise does not",
                    "They are uncorrelated",
                    "One series causes the other",
                    "The second series lags the first by 5 years"
                ],
                "correctExplanation": "Coherence is a squared correlation frequency by frequency: strong co-movement in the business-cycle band, almost none at high frequencies.",
                "incorrectExplanation": "Coherence measures linear association at each frequency; it says nothing about causality, and the lag is read from the phase."
            },
            "ro": {
                "title": "Coerența",
                "text": "Coerența pătratică a două serii este 0,9 la ciclurile de 5 ani și 0,1 la ciclurile lunare. Ce înseamnă aceasta?",
                "options": [
                    "Ciclurile lor lente se mișcă împreună, iar zgomotul lor pe termen scurt nu",
                    "Seriile sînt necorelate",
                    "O serie o cauzează pe cealaltă",
                    "A doua serie este întîrziată cu 5 ani față de prima"
                ],
                "correctExplanation": "Coerența este un pătrat al corelației, frecvență cu frecvență: o mișcare comună puternică în banda ciclului economic și aproape niciuna la frecvențele înalte.",
                "incorrectExplanation": "Coerența măsoară legătura liniară la fiecare frecvență; nu spune nimic despre cauzalitate, iar decalajul se citește din fază."
            }
        },
        {
            "correct": 3,
            "en": {
                "title": "Wavelets",
                "text": "What does a wavelet scalogram show that the periodogram does not?",
                "options": [
                    "The exact amplitude of each Fourier frequency",
                    "The autocorrelation function",
                    "The long-run mean of the series",
                    "How the power at each period changes over time"
                ],
                "correctExplanation": "The periodogram averages over the whole sample; the continuous wavelet transform gives power by period and by date (Torrence and Compo, 1998).",
                "incorrectExplanation": "A scalogram is a time-frequency picture: for example, the sunspot cycle is weak around 1800 and strong in the 1950s."
            },
            "ro": {
                "title": "Wavelets",
                "text": "Ce arată scalograma wavelet și nu arată periodograma?",
                "options": [
                    "Amplitudinea exactă a fiecărei frecvențe Fourier",
                    "Funcția de autocorelație",
                    "Media pe termen lung a seriei",
                    "Cum se schimbă în timp puterea de la fiecare perioadă"
                ],
                "correctExplanation": "Periodograma face media pe întregul eșantion; transformata wavelet continuă dă puterea pe perioade și pe date (Torrence și Compo, 1998).",
                "incorrectExplanation": "Scalograma este o imagine timp–frecvență: de exemplu, ciclul petelor solare este slab în jurul anului 1800 și puternic în anii 1950."
            }
        }
    ]
};
