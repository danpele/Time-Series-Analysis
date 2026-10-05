// ============================================================
// Chapter 12 quiz bank: Spectral analysis (EN + RO)
// 10 questions ported from the 2025/2026 site; 10 drawn per attempt.
// The bank grows to 24 questions when the chapter is rebuilt.
// correct = index (0-3) of the right option in the original order.
// incorrectExplanation must not name a letter: the engine prepends
// "The correct answer is X) ..." after shuffling the options.
// ============================================================
window.TSA_DATA.quizzes['spectral'] = {
    "draw": 20,
    "questions": [
        {
            "correct": 1,
            "en": {
                "title": "Discrete Fourier transform",
                "text": "What does the discrete Fourier transform (DFT) of a time series describe?",
                "options": [
                    "The linear trend of the series",
                    "The decomposition of the series into sinusoidal components at different frequencies",
                    "The number of missing observations",
                    "The correlation between different series"
                ],
                "correctExplanation": "The DFT $d(\\omega_j) = \\frac{1}{\\sqrt{T}} \\sum_{t=1}^{T} x_t e^{-2\\pi i \\omega_j t}$, evaluated at the Fourier frequencies $\\omega_j = j/T$, writes the series as a sum of sinusoids and gives the amplitude and phase at each frequency.",
                "incorrectExplanation": "A trend is estimated by regression or filtering, missing values are a data issue, and co-movement between two series is measured by the cross-spectrum or coherence. The DFT of a single series decomposes it into sinusoids."
            },
            "ro": {
                "title": "Transformata Fourier discretă",
                "text": "Ce descrie transformata Fourier discretă (DFT) a unei serii de timp?",
                "options": [
                    "Trendul liniar al seriei",
                    "Descompunerea seriei în componente sinusoidale de frecvențe diferite",
                    "Numărul de observații lipsă",
                    "Corelația dintre serii diferite"
                ],
                "correctExplanation": "DFT $d(\\omega_j) = \\frac{1}{\\sqrt{T}} \\sum_{t=1}^{T} x_t e^{-2\\pi i \\omega_j t}$, calculată la frecvențele Fourier $\\omega_j = j/T$, scrie seria ca sumă de sinusoide și dă amplitudinea și faza la fiecare frecvență.",
                "incorrectExplanation": "Trendul se estimează prin regresie sau filtrare, valorile lipsă sînt o problemă a datelor, iar co-mișcarea a două serii se măsoară prin spectrul încrucișat sau prin coerență. DFT a unei singure serii o descompune în sinusoide."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Nyquist frequency",
                "text": "What happens when a signal contains frequencies above the Nyquist limit ($f_N = 1/2$ cycle per observation)?",
                "options": [
                    "The frequencies are detected correctly",
                    "Aliasing occurs: high frequencies appear as spurious low frequencies",
                    "The signal is filtered automatically",
                    "The amplitude becomes zero"
                ],
                "correctExplanation": "With one observation per unit of time, oscillations faster than half a cycle per observation cannot be distinguished from slower ones. They fold back below $f_N$ and appear as spurious low-frequency components.",
                "incorrectExplanation": "Frequencies above $f_N$ are not detected correctly, sampling does not filter them out, and their energy does not vanish: it is attributed to the wrong, lower frequency. This is aliasing."
            },
            "ro": {
                "title": "Frecvența Nyquist",
                "text": "Ce se întîmplă cînd un semnal conține frecvențe peste limita Nyquist ($f_N = 1/2$ cicluri pe observație)?",
                "options": [
                    "Frecvențele sînt detectate corect",
                    "Apare aliasing: frecvențele înalte apar ca frecvențe joase false",
                    "Semnalul este filtrat automat",
                    "Amplitudinea devine zero"
                ],
                "correctExplanation": "Cu o observație pe unitatea de timp, oscilațiile mai rapide de o jumătate de ciclu pe observație nu pot fi deosebite de oscilații mai lente. Ele se pliază sub $f_N$ și apar ca componente false de frecvență joasă.",
                "incorrectExplanation": "Frecvențele peste $f_N$ nu sînt detectate corect, eșantionarea nu le filtrează, iar energia lor nu dispare, ci este atribuită unei frecvențe greșite, mai joase. Acesta este fenomenul de aliasing."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Inconsistency of the periodogram",
                "text": "Why is the periodogram an inconsistent estimator of the spectral density?",
                "options": [
                    "Its bias increases with the sample size",
                    "Its variance does NOT decrease as $T \\to \\infty$",
                    "It cannot detect low frequencies",
                    "It can be computed only for series with an even number of observations"
                ],
                "correctExplanation": "For $0 < \\omega_j < 1/2$, $I(\\omega_j) \\xrightarrow{d} f(\\omega_j) \\cdot \\chi^2_2/2$, so its variance stays close to $f(\\omega_j)^2$ whatever the sample size. This motivates smoothed estimators (Daniell kernels, Welch, multitaper).",
                "incorrectExplanation": "The periodogram is asymptotically unbiased, it does estimate low frequencies, and it can be computed for any sample size. Its problem is a variance that does not shrink as $T$ grows."
            },
            "ro": {
                "title": "Inconsistența periodogramei",
                "text": "De ce este periodograma un estimator inconsistent al densității spectrale?",
                "options": [
                    "Distorsiunea (bias) crește odată cu volumul eșantionului",
                    "Varianța ei NU scade cînd $T \\to \\infty$",
                    "Nu poate detecta frecvențele joase",
                    "Poate fi calculată doar pentru serii cu un număr par de observații"
                ],
                "correctExplanation": "Pentru $0 < \\omega_j < 1/2$, $I(\\omega_j) \\xrightarrow{d} f(\\omega_j) \\cdot \\chi^2_2/2$, deci varianța ei rămîne apropiată de $f(\\omega_j)^2$ oricare ar fi volumul eșantionului. De aici nevoia de estimatori netezi (nuclee Daniell, Welch, multitaper).",
                "incorrectExplanation": "Periodograma este asimptotic nedeplasată, estimează și frecvențele joase și poate fi calculată pentru orice volum al eșantionului. Problema ei este o varianță care nu scade pe măsură ce $T$ crește."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Welch's method",
                "text": "How does Welch's method reduce the variance of spectral estimates?",
                "options": [
                    "It uses only the central half of the data",
                    "It averages periodograms over overlapping windowed segments",
                    "It differences the series",
                    "It removes frequencies above a threshold"
                ],
                "correctExplanation": "Welch's method splits the series into overlapping segments, applies a taper (for example Hann) to each, computes their periodograms and averages them. Averaging reduces the variance at the cost of frequency resolution.",
                "incorrectExplanation": "Discarding half the data would increase the variance, differencing changes the spectrum rather than smoothing its estimate, and removing frequencies is filtering. Welch's method averages periodograms of overlapping tapered segments."
            },
            "ro": {
                "title": "Metoda Welch",
                "text": "Cum reduce metoda Welch varianța estimărilor spectrale?",
                "options": [
                    "Folosește doar jumătatea centrală a datelor",
                    "Mediază periodogramele unor segmente suprapuse, ponderate cu o fereastră",
                    "Diferențiază seria",
                    "Elimină frecvențele de peste un prag"
                ],
                "correctExplanation": "Metoda Welch împarte seria în segmente suprapuse, aplică fiecăruia o fereastră de ponderare (de exemplu Hann), calculează periodogramele și le mediază. Medierea reduce varianța, cu prețul unei rezoluții mai slabe în frecvență.",
                "incorrectExplanation": "Renunțarea la jumătate din date ar crește varianța, diferențierea modifică spectrul în loc să-i netezească estimarea, iar eliminarea unor frecvențe înseamnă filtrare. Metoda Welch mediază periodogramele unor segmente suprapuse și ponderate."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Long memory in the frequency domain",
                "text": "How does long memory show up in the frequency domain?",
                "options": [
                    "The spectral density is constant (flat)",
                    "The spectral density diverges at frequency zero: $f(\\omega) \\propto |\\omega|^{-2d}$",
                    "The spectrum has a single peak at the Nyquist frequency",
                    "All autocorrelations are zero"
                ],
                "correctExplanation": "For ARFIMA with $0 < d < 0.5$, $f(\\omega) \\sim c\\,|\\omega|^{-2d}$ as $\\omega \\to 0$: spectral power rises without bound at low frequencies, the frequency-domain counterpart of hyperbolically decaying autocorrelations. The GPH estimator of $d$ is based on this slope.",
                "incorrectExplanation": "A flat spectrum and zero autocorrelations describe white noise, and a peak at the Nyquist frequency corresponds to rapid sign alternation. Long memory concentrates power near frequency zero, with $f(\\omega) \\propto |\\omega|^{-2d}$."
            },
            "ro": {
                "title": "Memoria lungă în domeniul frecvenței",
                "text": "Cum se manifestă memoria lungă în domeniul frecvenței?",
                "options": [
                    "Densitatea spectrală este constantă (plată)",
                    "Densitatea spectrală diverge la frecvența zero: $f(\\omega) \\propto |\\omega|^{-2d}$",
                    "Spectrul are un singur vîrf la frecvența Nyquist",
                    "Toate autocorelațiile sînt zero"
                ],
                "correctExplanation": "Pentru ARFIMA cu $0 < d < 0{,}5$, $f(\\omega) \\sim c\\,|\\omega|^{-2d}$ cînd $\\omega \\to 0$: puterea spectrală crește nemărginit la frecvențele joase, ceea ce corespunde, în domeniul frecvenței, autocorelațiilor care scad hiperbolic. Estimatorul GPH al lui $d$ folosește tocmai această pantă.",
                "incorrectExplanation": "Un spectru plat și autocorelațiile nule descriu zgomotul alb, iar un vîrf la frecvența Nyquist corespunde alternării rapide a semnului. Memoria lungă concentrează puterea lîngă frecvența zero, cu $f(\\omega) \\propto |\\omega|^{-2d}$."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Spectral coherence",
                "text": "What does the squared coherence $C^2_{xy}(\\omega)$ between two series measure?",
                "options": [
                    "The overall correlation between the series",
                    "The linear association at each frequency, analogous to a frequency-specific $R^2$",
                    "The difference in amplitude between the series",
                    "The number of common cycles"
                ],
                "correctExplanation": "$C^2_{xy}(\\omega) = |f_{xy}(\\omega)|^2 / [f_x(\\omega) f_y(\\omega)] \\in [0,1]$ is the share of the variance of one series at frequency $\\omega$ that is linearly explained by the other, a frequency-specific $R^2$.",
                "incorrectExplanation": "Coherence is not a single overall correlation, does not compare amplitudes (that is the gain) and does not count cycles. It measures linear association frequency by frequency."
            },
            "ro": {
                "title": "Coerența spectrală",
                "text": "Ce măsoară coerența pătratică $C^2_{xy}(\\omega)$ dintre două serii?",
                "options": [
                    "Corelația globală dintre serii",
                    "Asocierea liniară la fiecare frecvență, analogă unui $R^2$ pe frecvențe",
                    "Diferența de amplitudine dintre serii",
                    "Numărul de cicluri comune"
                ],
                "correctExplanation": "$C^2_{xy}(\\omega) = |f_{xy}(\\omega)|^2 / [f_x(\\omega) f_y(\\omega)] \\in [0,1]$ este proporția din varianța unei serii la frecvența $\\omega$ explicată liniar de cealaltă serie, adică un $R^2$ pentru fiecare frecvență.",
                "incorrectExplanation": "Coerența nu este o corelație globală unică, nu compară amplitudinile (acesta este rolul cîștigului, gain) și nu numără cicluri. Ea măsoară asocierea liniară frecvență cu frecvență."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Hodrick-Prescott filter",
                "text": "What is the main limitation of the Hodrick-Prescott (HP) filter ($\\lambda = 1600$ for quarterly data) as a business-cycle filter?",
                "options": [
                    "It cannot be applied to quarterly data",
                    "Its cycle is only an approximate high-pass filter: it keeps high-frequency noise and has a gradual cut-off",
                    "It requires stationary data",
                    "It works only for series with a linear trend"
                ],
                "correctExplanation": "The HP cycle passes all frequencies above a gradual cut-off, so it retains short-run noise, and its gain changes smoothly around the cut-off instead of isolating a band. Band-pass filters such as Baxter-King and Christiano-Fitzgerald approximate the ideal rectangular gain more closely. Hamilton (2018) also criticises spurious dynamics and end-point problems.",
                "incorrectExplanation": "The HP filter is routinely used on quarterly data (that is what $\\lambda = 1600$ is calibrated for), it can be applied to non-stationary series, including I(2) ones, and its trend is a flexible smooth curve, not a line. Its weakness is that it is not an ideal band-pass filter."
            },
            "ro": {
                "title": "Filtrul Hodrick-Prescott",
                "text": "Care este principala limită a filtrului Hodrick-Prescott (HP) ($\\lambda = 1600$ pentru date trimestriale) ca filtru pentru ciclul economic?",
                "options": [
                    "Nu poate fi aplicat datelor trimestriale",
                    "Componenta ciclică este doar un filtru trece-sus aproximativ: păstrează zgomotul de frecvență înaltă și are o tăiere graduală",
                    "Necesită date staționare",
                    "Funcționează doar pentru serii cu trend liniar"
                ],
                "correctExplanation": "Componenta ciclică HP lasă să treacă toate frecvențele de peste un prag gradual, deci păstrează zgomotul de termen scurt, iar cîștigul ei variază lin în jurul pragului, fără să izoleze o bandă. Filtrele trece-bandă Baxter-King și Christiano-Fitzgerald aproximează mai bine cîștigul dreptunghiular ideal. Hamilton (2018) critică, în plus, dinamica falsă și problemele de la capetele eșantionului.",
                "incorrectExplanation": "Filtrul HP se aplică în mod curent datelor trimestriale (pentru ele este calibrat $\\lambda = 1600$), poate fi folosit pentru serii nestaționare, inclusiv I(2), iar trendul său este o curbă netedă flexibilă, nu o dreaptă. Slăbiciunea lui este că nu este un filtru trece-bandă ideal."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Wavelet and Fourier analysis",
                "text": "What is the main advantage of wavelet analysis over Fourier analysis for non-stationary series?",
                "options": [
                    "Wavelets have better frequency resolution",
                    "Wavelets give a time-frequency representation, showing WHEN frequencies occur",
                    "Wavelets require less data",
                    "Wavelets remove the trend automatically"
                ],
                "correctExplanation": "Fourier analysis shows WHICH frequencies are present but not WHEN. The wavelet scalogram shows both, which matters for series whose cycles appear, disappear or shift over time.",
                "incorrectExplanation": "Wavelets trade some frequency resolution for time localisation, do not need less data and do not remove trends by themselves. Their advantage is localisation in both time and frequency."
            },
            "ro": {
                "title": "Analiza wavelet și analiza Fourier",
                "text": "Care este principalul avantaj al analizei wavelet față de analiza Fourier pentru seriile nestaționare?",
                "options": [
                    "Wavelet-urile au o rezoluție mai bună în frecvență",
                    "Wavelet-urile oferă o reprezentare timp-frecvență, care arată CÎND apar frecvențele",
                    "Wavelet-urile necesită mai puține date",
                    "Wavelet-urile elimină automat trendul"
                ],
                "correctExplanation": "Analiza Fourier arată CE frecvențe sînt prezente, dar nu și CÎND. Scalograma wavelet le arată pe amîndouă, ceea ce contează pentru seriile ale căror cicluri apar, dispar sau se modifică în timp.",
                "incorrectExplanation": "Wavelet-urile renunță la o parte din rezoluția în frecvență în schimbul localizării în timp, nu necesită mai puține date și nu elimină singure trendul. Avantajul lor este localizarea simultană în timp și în frecvență."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Spectral leakage",
                "text": "What causes spectral leakage and how is it reduced?",
                "options": [
                    "The signal frequency is too high; it is reduced by subsampling",
                    "The frequency does not fall exactly on the DFT grid; it is reduced by tapering, for example with a Hann window",
                    "The series has too few observations; it is reduced by zero-padding",
                    "The data contain missing values; it is reduced by interpolation"
                ],
                "correctExplanation": "When the true frequency lies between Fourier frequencies, the finite sample acts as a rectangular window and spreads power into neighbouring frequencies through sidelobes. Tapers such as Hann or DPSS (multitaper) attenuate the sidelobes and concentrate power near the true frequency.",
                "incorrectExplanation": "Subsampling creates aliasing rather than curing leakage, zero-padding only interpolates the spectrum on a finer grid, and interpolation addresses missing data. Leakage comes from the finite window and is reduced by tapering."
            },
            "ro": {
                "title": "Scurgerea spectrală",
                "text": "Ce cauzează scurgerea spectrală (spectral leakage) și cum se reduce?",
                "options": [
                    "Frecvența semnalului este prea mare; se reduce prin subeșantionare",
                    "Frecvența nu cade exact pe grila DFT; se reduce prin ponderare cu o fereastră (tapering), de exemplu Hann",
                    "Seria are prea puține observații; se reduce prin zero-padding",
                    "Datele conțin valori lipsă; se reduce prin interpolare"
                ],
                "correctExplanation": "Cînd frecvența reală se află între două frecvențe Fourier, eșantionul finit acționează ca o fereastră dreptunghiulară și împrăștie puterea spre frecvențele vecine prin lobii laterali. Ferestrele Hann sau DPSS (multitaper) atenuează lobii laterali și concentrează puterea lîngă frecvența reală.",
                "incorrectExplanation": "Subeșantionarea produce aliasing în loc să elimine scurgerea, zero-padding-ul doar interpolează spectrul pe o grilă mai fină, iar interpolarea tratează valorile lipsă. Scurgerea provine din fereastra finită și se reduce prin tapering."
            }
        },
        {
            "correct": 1,
            "en": {
                "title": "Business-cycle band",
                "text": "What is the standard business-cycle band (Burns and Mitchell, used by the Baxter-King filter) for quarterly data?",
                "options": [
                    "2–4 quarters (6 months to 1 year)",
                    "6–32 quarters (1.5 to 8 years)",
                    "40–100 quarters (10 to 25 years)",
                    "1–2 quarters (3 to 6 months)"
                ],
                "correctExplanation": "Following Burns and Mitchell, business cycles are taken to last between 6 and 32 quarters (1.5 to 8 years). Band-pass filters such as Baxter-King and Christiano-Fitzgerald extract fluctuations in this band.",
                "incorrectExplanation": "Fluctuations shorter than 6 quarters are treated as seasonal or irregular noise, and those longer than 32 quarters as trend or long swings. The business-cycle band is 6–32 quarters."
            },
            "ro": {
                "title": "Banda ciclului economic",
                "text": "Care este banda standard a ciclului economic (Burns și Mitchell, folosită de filtrul Baxter-King) pentru date trimestriale?",
                "options": [
                    "2–4 trimestre (de la 6 luni la 1 an)",
                    "6–32 de trimestre (de la 1,5 la 8 ani)",
                    "40–100 de trimestre (de la 10 la 25 de ani)",
                    "1–2 trimestre (de la 3 la 6 luni)"
                ],
                "correctExplanation": "Urmîndu-i pe Burns și Mitchell, ciclurile economice sînt considerate a dura între 6 și 32 de trimestre (de la 1,5 la 8 ani). Filtrele trece-bandă Baxter-King și Christiano-Fitzgerald extrag fluctuațiile din această bandă.",
                "incorrectExplanation": "Fluctuațiile mai scurte de 6 trimestre sînt tratate ca zgomot sezonier sau neregulat, iar cele mai lungi de 32 de trimestre ca trend sau oscilații lungi. Banda ciclului economic este de 6–32 de trimestre."
            }
        }
    ]
};
