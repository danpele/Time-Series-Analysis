// ============================================================
// TSA course data (EN + RO). Rendered by assets/site.js.
// Chapter links: { type, href[, colab][, label] } for an existing file,
// { type, soon: true } for an item still in preparation (shown greyed, no link).
// A chapter with no existing file in a language shows "in preparation" only.
// Chapters 1-15 still link the 2025/2026 decks and notebooks; each chapter switches to the new
// file names (python3 latex/tsa_chapters.py) when it is rebuilt.
// ============================================================
(function () {
    const REPO = 'https://github.com/danpele/Time-Series-Analysis';
    const TREE = REPO + '/tree/main/';
    const COLAB = 'https://colab.research.google.com/github/danpele/Time-Series-Analysis/blob/main/';

    // Quantinar courses, referenced by key from the chapters (URLs checked: HTTP 200)
    const Q = {
        tsaPython: ['Applied Time Series Analysis with Python', 'https://quantinar.com/course/137/applied-time-series-analysis-with-python'],
        sfm: ['Statistics of Financial Markets', 'https://quantinar.com/course/103/statistics-of-financial-markets'],
        statRisk: ['Measuring Statistical Risk', 'https://quantinar.com/course/100080/measuring-statistical-risk'],
        kalman: ['Kalman Filter', 'https://quantinar.com/course/42/methodology'],
        rf: ['Random Forests', 'https://quantinar.com/course/68/RF'],
        mlRisk: ['Machine learning in Financial Risk', 'https://quantinar.com/course/934/machine-learning-in-financial-risk'],
        cryptoEfficiency: ['Efficiency of cryptocurrency markets - A GMM-based analysis', 'https://quantinar.com/course/184/efficiency-of-cryptocurrency-markets-a-gmm-based-analysis'],
        nextWord: ['The Next Word Problem', 'https://quantinar.com/course/100100/the-next-word-problem-full-course'],
        xfg: ['XFG Advanced Methods in Quantitative Finance', 'https://quantinar.com/course/100067/xfg-advanced-methods-in-quantitative-finance']
    };
    const q = (...keys) => keys.map(k => ({ title: Q[k][0], url: Q[k][1] }));

    // helpers for chapter links
    const pdf = (type, href, label) => (label ? { type, href, label } : { type, href });
    const soon = type => ({ type, soon: true });
    const nb = (path, label) => Object.assign({ type: 'notebook', href: REPO + '/blob/main/' + path, colab: COLAB + path }, label ? { label } : {});
    const NB_LECT = { en: 'Lecture notebook', ro: 'Notebook curs' };
    const NB_SEM = { en: 'Seminar notebook', ro: 'Notebook seminar' };
    const ql = path => ({ type: 'quantlets', href: TREE + path });

    // the 2025/2026 materials (old file names), until the chapter is rebuilt
    const OLD = {
        lectEn: f => pdf('slides', 'EN/Courses/' + f + '.pdf'),
        lectRo: f => pdf('slides', 'RO/Courses/' + f + '_ro.pdf'),
        semEn: f => pdf('seminar', 'EN/Seminars/' + f + '.pdf'),
        semRo: f => pdf('seminar', 'RO/Seminars/' + f + '_ro.pdf'),
        nbL: f => nb('EN/Course_Notebooks/' + f + '_lecture_notebook.ipynb', NB_LECT),
        nbS: f => nb('EN/Seminar_Notebooks/' + f + '_seminar_notebook.ipynb', NB_SEM)
    };
    // lecture + seminar + notebooks + Quantlets of an old chapter, same notebooks on both pages (EN only)
    function old(lect, sem, nbBase, qlPath, opts) {
        const o = opts || {};
        const tail = [].concat(nbBase ? [OLD.nbL(nbBase)] : [], nbBase && o.nbSem !== false ? [OLD.nbS(nbBase)] : [],
            o.extraNb || [], qlPath ? [ql(qlPath)] : []);
        const lab = o.label || null;
        return {
            en: [].concat([lab ? pdf('slides', 'EN/Courses/' + lect + '.pdf', lab) : OLD.lectEn(lect)], o.extraEn || [],
                sem ? [OLD.semEn(sem)] : [], o.extraSemEn || [], tail),
            ro: [].concat([lab ? pdf('slides', 'RO/Courses/' + lect + '_ro.pdf', lab) : OLD.lectRo(lect)], o.extraRo || [],
                sem ? [OLD.semRo(sem)] : [], o.extraSemRo || [], tail)
        };
    }

    window.TSA_DATA = {
        repo: REPO,

        // ---------------------------------------------------------------
        // UI strings
        // ---------------------------------------------------------------
        ui: {
            en: {
                pageTitle: 'Time Series Analysis - Course Website',
                courseTitle: 'Time Series Analysis',
                subtitle: "Bachelor's programmes Economic Informatics and Economic Cybernetics, year 3, semester 2 | Faculty of Cybernetics, Statistics and Economic Informatics | Bucharest University of Economic Studies",
                nav: { home: 'Home', chapters: 'Chapters', project: 'Project', quizzes: 'Quizzes', resources: 'Resources', contact: 'Contact' },
                overview: 'Course Overview',
                objectives: 'Learning Objectives',
                heroTag: 'Trend, seasonality, ARIMA, GARCH, VAR and cointegration: modelling and forecasting time series on real data with Python.',
                heroCta1: 'Explore the chapters',
                heroCta2: 'Team project',
                heroCta3: 'Attendance form',
                qrTeachers: 'For instructors: attendance QR code',
                qrLecture: 'lecture',
                qrSeminar: 'seminar',
                teacherBtn: 'Instructor access',
                teacherPrompt: 'Sign in with the instructor Google account to see the attendance QR code.',
                teacherDenied: 'This account has no instructor access.',
                close: 'Close',
                formulas: 'Key Formulas',
                chapters: 'Course Chapters',
                chapter: 'Chapter',
                chapterShort: 'Ch',
                comingSoon: 'In preparation',
                comingSoonShort: 'in preparation',
                chapterSoon: 'The materials of this chapter are in preparation.',
                selfStudy: 'Self-study',
                quantinar: 'Go deeper on Quantinar',
                links: {
                    slides: 'Lecture Slides', slidesExtra: 'Additional Slides', seminar: 'Seminar', seminarExtra: 'Additional Seminar',
                    notebook: 'Notebook', quantlets: 'Quantlets', colab: 'Open in Colab'
                },
                projectTitle: 'Team Project',
                aiTitle: 'Using AI in this course',
                quizzes: 'Self-Assessment Quizzes',
                quizIntro: 'Each attempt draws up to 20 questions at random from the chapter bank and shuffles the answers. An answer is locked once selected.',
                loginPrompt: 'Sign in with your ASE Google account (@ase.ro or @stud.ase.ro) to take the quizzes.',
                loginRequired: 'Sign in above with your ASE Google account to see this quiz.',
                loginWrongDomain: 'Please use your ASE account (@ase.ro or @stud.ase.ro).',
                saving: 'Saving your score...',
                saved: 'Your score has been recorded.',
                saveExpired: 'Your session has expired: sign out, sign in again and recalculate.',
                saveFailed: 'The score could not be saved. Please try again or tell the instructor.',
                loggedAs: 'Logged in as',
                logout: 'Logout',
                quizSoon: 'The quiz for this chapter will be published together with its materials.',
                question: 'Question',
                correct: 'Correct!',
                incorrect: 'Incorrect.',
                correctIs: 'The correct answer is',
                calc: 'Calculate Score',
                reset: 'New attempt',
                score: 'Score',
                unanswered: 'questions unanswered',
                verdicts: ['Keep practising!', 'Keep studying!', 'Good job!', 'Excellent!'],
                detailed: 'Detailed Results',
                colQ: 'Q', colQuestion: 'Question', colCorrect: 'Correct answer', colYours: 'Your answer', colResult: 'Result',
                resources: 'Resources',
                bibliography: 'Bibliography',
                dataSources: 'Data sources',
                contact: 'Contact',
                instructor: 'Lecturer',
                seminarCard: 'Seminar',
                seminarRole: 'Seminar instructor',
                office: 'Office Hours',
                officeText: 'By appointment',
                footer: 'Time Series Analysis | Faculty of Cybernetics, Statistics and Economic Informatics | Bucharest University of Economic Studies'
            },
            ro: {
                pageTitle: 'Serii de timp - Site-ul cursului',
                courseTitle: 'Serii de timp',
                subtitle: 'Programele de licență Informatică economică și Cibernetică economică, anul III, semestrul 2 | Facultatea de Cibernetică, Statistică și Informatică Economică | Academia de Studii Economice din București',
                nav: { home: 'Acasă', chapters: 'Capitole', project: 'Proiect', quizzes: 'Quiz-uri', resources: 'Resurse', contact: 'Contact' },
                overview: 'Prezentarea cursului',
                objectives: 'Obiective de învățare',
                heroTag: 'Trend, sezonalitate, ARIMA, GARCH, VAR și cointegrare: modelarea și prognoza seriilor de timp pe date reale, în Python.',
                heroCta1: 'Explorați capitolele',
                heroCta2: 'Proiect de echipă',
                heroCta3: 'Formular de prezență',
                qrTeachers: 'Pentru cadre didactice: cod QR de prezență',
                qrLecture: 'curs',
                qrSeminar: 'seminar',
                teacherBtn: 'Acces cadre didactice',
                teacherPrompt: 'Autentificați-vă cu contul Google de cadru didactic pentru a vedea codul QR de prezență.',
                teacherDenied: 'Acest cont nu are acces de cadru didactic.',
                close: 'Închide',
                formulas: 'Formule-cheie',
                chapters: 'Capitolele cursului',
                chapter: 'Capitolul',
                chapterShort: 'Cap.',
                comingSoon: 'În pregătire',
                comingSoonShort: 'în pregătire',
                chapterSoon: 'Materialele acestui capitol sînt în pregătire.',
                selfStudy: 'Studiu individual',
                quantinar: 'Aprofundare pe Quantinar',
                links: {
                    slides: 'Slide-urile cursului', slidesExtra: 'Slide-uri suplimentare', seminar: 'Seminar', seminarExtra: 'Seminar suplimentar',
                    notebook: 'Notebook', quantlets: 'Quantlets', colab: 'Deschide în Colab'
                },
                projectTitle: 'Proiect de echipă',
                aiTitle: 'Utilizarea instrumentelor AI',
                quizzes: 'Quiz-uri de autoevaluare',
                quizIntro: 'La fiecare încercare se extrag aleator cel mult 20 de întrebări din banca de întrebări a capitolului, iar ordinea variantelor de răspuns se schimbă. Un răspuns ales nu mai poate fi modificat.',
                loginPrompt: 'Autentificați-vă cu contul Google ASE (@ase.ro sau @stud.ase.ro) pentru a rezolva quiz-urile.',
                loginRequired: 'Autentificați-vă mai sus cu contul Google ASE pentru a vedea acest quiz.',
                loginWrongDomain: 'Folosiți contul ASE (@ase.ro sau @stud.ase.ro).',
                saving: 'Se salvează scorul...',
                saved: 'Scorul a fost înregistrat.',
                saveExpired: 'Sesiunea a expirat: deconectați-vă, autentificați-vă din nou și recalculați scorul.',
                saveFailed: 'Scorul nu a putut fi salvat. Încercați din nou sau anunțați titularul de curs.',
                loggedAs: 'Autentificat ca',
                logout: 'Deconectare',
                quizSoon: 'Quiz-ul acestui capitol va fi publicat odată cu materialele capitolului.',
                question: 'Întrebarea',
                correct: 'Corect!',
                incorrect: 'Greșit.',
                correctIs: 'Răspunsul corect este',
                calc: 'Calculează scorul',
                reset: 'Încercare nouă',
                score: 'Scor',
                unanswered: 'întrebări fără răspuns',
                verdicts: ['Mai exersați!', 'Mai studiați!', 'Bine!', 'Excelent!'],
                detailed: 'Rezultate detaliate',
                colQ: 'Nr.', colQuestion: 'Întrebare', colCorrect: 'Răspuns corect', colYours: 'Răspunsul ales', colResult: 'Rezultat',
                resources: 'Resurse',
                bibliography: 'Bibliografie',
                dataSources: 'Surse de date',
                contact: 'Contact',
                instructor: 'Titular de curs',
                seminarCard: 'Seminar',
                seminarRole: 'Titular de seminar',
                office: 'Program de consultații',
                officeText: 'Pe bază de programare',
                footer: 'Serii de timp | Facultatea de Cibernetică, Statistică și Informatică Economică | Academia de Studii Economice din București'
            }
        },

        // ---------------------------------------------------------------
        // Overview cards and objectives
        // ---------------------------------------------------------------
        overview: {
            en: [
                { h: 'Course', p: ['Time Series Analysis', "Bachelor's programmes Economic Informatics and Economic Cybernetics", 'Year 3, semester 2, academic year 2026/2027', '2 hours of lecture and 1 hour of seminar per week; 4 ECTS'] },
                { h: 'Prerequisites', p: ['Probability and statistics', 'Econometrics (linear regression)', 'Python programming'] },
                { h: 'Assessment', p: ['Written exam: 70%', 'Team project: 20%', 'Attendance: 10%'] },
                { h: 'Main textbook', p: ['Huang &amp; Petukhina, <a href="https://doi.org/10.1007/978-3-031-13584-2" target="_blank" rel="noopener"><em>Applied Time Series Analysis and Forecasting with Python</em></a>, Springer, 2022', 'Free companion: Hyndman &amp; Athanasopoulos, <a href="https://otexts.com/fpp3/" target="_blank" rel="noopener"><em>Forecasting: Principles and Practice</em></a> (3rd ed.)'] },
                { h: 'Tools', p: ['Python (statsmodels, arch, pandas), Jupyter / Google Colab', 'GitHub, Quantlet, Quantinar'] }
            ],
            ro: [
                { h: 'Curs', p: ['Serii de timp', 'Programele de licență Informatică economică și Cibernetică economică', 'Anul III, semestrul 2, anul universitar 2026/2027', '2 ore de curs și 1 oră de seminar pe săptămînă; 4 credite ECTS'] },
                { h: 'Cunoștințe prealabile', p: ['Probabilități și statistică', 'Econometrie (regresia liniară)', 'Programare în Python'] },
                { h: 'Evaluare', p: ['Examen scris: 70%', 'Proiect de echipă: 20%', 'Prezență: 10%'] },
                { h: 'Manual de bază', p: ['Huang și Petukhina, <a href="https://doi.org/10.1007/978-3-031-13584-2" target="_blank" rel="noopener"><em>Applied Time Series Analysis and Forecasting with Python</em></a>, Springer, 2022', 'Manual însoțitor, gratuit: Hyndman și Athanasopoulos, <a href="https://otexts.com/fpp3/" target="_blank" rel="noopener"><em>Forecasting: Principles and Practice</em></a> (ediția a 3-a)'] },
                { h: 'Instrumente', p: ['Python (statsmodels, arch, pandas), Jupyter / Google Colab', 'GitHub, Quantlet, Quantinar'] }
            ]
        },

        objectives: {
            en: [
                'Describe a time series by its components (trend, seasonality, cycle, noise) and forecast it with exponential smoothing, evaluated out of sample',
                'Explain stationarity, autocorrelation and the lag operator, and identify, estimate and check ARMA models',
                'Test for unit roots (ADF, KPSS) and build ARIMA and SARIMA models for real economic and financial series',
                'Model volatility with ARCH and GARCH models and interpret persistence and volatility forecasts',
                'Analyse several series together with VAR models, Granger causality, impulse responses, cointegration and VECM',
                'Deliver a reproducible forecasting analysis in Python, published on GitHub and documented as Quantlets'
            ],
            ro: [
                'Descrierea unei serii de timp prin componentele ei (trend, sezonalitate, ciclu, componenta neregulată) și prognoza ei prin netezire exponențială, evaluată în afara eșantionului',
                'Explicarea staționarității, a autocorelației și a operatorului lag; identificarea, estimarea și validarea modelelor ARMA',
                'Testarea rădăcinii unitare (ADF, KPSS) și construirea modelelor ARIMA și SARIMA pentru serii economice și financiare reale',
                'Modelarea volatilității cu modele ARCH și GARCH și interpretarea persistenței și a prognozelor de volatilitate',
                'Analiza simultană a mai multor serii cu modele VAR, cauzalitate Granger, funcții de răspuns la impuls, cointegrare și VECM',
                'Elaborarea unei analize de prognoză reproductibile în Python, publicate pe GitHub și documentate sub formă de Quantlets'
            ]
        },

        // ---------------------------------------------------------------
        // Key formulas (shown in a collapsible box on the chapter card)
        // ---------------------------------------------------------------
        formulas: [
            { ch: 'intro', en: 'Simple exponential smoothing', ro: 'Netezirea exponențială simplă', tex: '$$\\hat y_{t+1|t} = \\alpha y_t + (1-\\alpha)\\,\\hat y_{t|t-1}, \\qquad 0<\\alpha\\le 1$$' },
            { ch: 'intro', en: 'Mean absolute scaled error', ro: 'Eroarea absolută medie scalată', tex: '$$\\mathrm{MASE} = \\frac{\\frac{1}{h}\\sum_{j=1}^{h}|y_{T+j}-\\hat y_{T+j}|}{\\frac{1}{T-m}\\sum_{t=m+1}^{T}|y_t-y_{t-m}|}$$' },
            { ch: 'stationarity', en: 'Autocovariance and autocorrelation (weak stationarity)', ro: 'Autocovarianța și autocorelația (staționaritate slabă)', tex: '$$\\gamma(h) = \\mathrm{Cov}(X_t, X_{t+h}), \\qquad \\rho(h) = \\gamma(h)/\\gamma(0)$$' },
            { ch: 'stationarity', en: 'Random walk', ro: 'Mersul aleator', tex: '$$X_t = X_{t-1} + \\varepsilon_t, \\qquad \\mathrm{Var}(X_t) = t\\,\\sigma^2$$' },
            { ch: 'arma', en: 'ARMA(p,q) with the lag operator', ro: 'ARMA(p,q) cu operatorul lag', tex: '$$\\phi(L)\\,X_t = \\theta(L)\\,\\varepsilon_t, \\qquad \\phi(L) = 1-\\phi_1 L-\\dots-\\phi_p L^p$$' },
            { ch: 'arma', en: 'AR(1): stationarity and autocorrelation', ro: 'AR(1): staționaritate și autocorelație', tex: '$$X_t = \\phi X_{t-1} + \\varepsilon_t, \\quad |\\phi|<1, \\qquad \\rho(h) = \\phi^{h}$$' },
            { ch: 'arima', en: 'ADF regression', ro: 'Regresia ADF', tex: '$$\\Delta y_t = \\alpha + \\beta t + \\gamma y_{t-1} + \\sum_{i=1}^{p}\\delta_i\\,\\Delta y_{t-i} + \\varepsilon_t, \\qquad H_0: \\gamma = 0$$' },
            { ch: 'seasonal', en: 'SARIMA(p,d,q)(P,D,Q)s', ro: 'SARIMA(p,d,q)(P,D,Q)s', tex: '$$\\Phi(L^s)\\,\\phi(L)\\,(1-L)^d(1-L^s)^D y_t = \\Theta(L^s)\\,\\theta(L)\\,\\varepsilon_t$$' },
            { ch: 'garch', en: 'GARCH(1,1)', ro: 'GARCH(1,1)', tex: '$$\\sigma_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2 + \\beta\\,\\sigma_{t-1}^2, \\qquad \\bar\\sigma^2 = \\frac{\\omega}{1-\\alpha-\\beta}$$' },
            { ch: 'var', en: 'VAR(p)', ro: 'VAR(p)', tex: '$$\\mathbf y_t = \\mathbf c + A_1\\mathbf y_{t-1} + \\dots + A_p\\mathbf y_{t-p} + \\mathbf u_t$$' },
            { ch: 'cointegration', en: 'Vector error correction model', ro: 'Modelul vectorial cu corecția erorii', tex: '$$\\Delta\\mathbf y_t = \\alpha\\beta^{\\top}\\mathbf y_{t-1} + \\sum_{i=1}^{p-1}\\Gamma_i\\,\\Delta\\mathbf y_{t-i} + \\mathbf u_t$$' },
            { ch: 'long-memory', en: 'Fractional differencing and hyperbolic ACF decay', ro: 'Diferențierea fracționară și descreșterea hiperbolică a ACF', tex: '$$(1-L)^d X_t = \\varepsilon_t, \\qquad \\rho(h) \\sim C\\,h^{2d-1}, \\quad 0<d<\\tfrac12$$' },
            { ch: 'state-space', en: 'State space form', ro: 'Forma în spațiul stărilor', tex: '$$y_t = Z\\alpha_t + \\varepsilon_t, \\qquad \\alpha_{t+1} = T\\alpha_t + \\eta_t$$' }
        ],

        // ---------------------------------------------------------------
        // Chapters 0-15. `id` is the stable key (quizzes, anchors, tabs);
        // `num` is only the display order. RO page -> RO files, EN page -> EN files.
        // Chapters 11-14 are self-study (selfStudy: true).
        // ---------------------------------------------------------------
        chapters: [
            {
                id: 'intro', num: 0,
                title: { en: 'Introduction: components and exponential smoothing', ro: 'Introducere: componente și netezire exponențială' },
                topics: {
                    en: ['Course organisation, assessment and tools; time series on real data (Romanian GDP, inflation, EUR/RON, BET, electricity, CO2); a short history from Yule and Slutsky to Box–Jenkins', 'Components (trend, seasonality, cycle, remainder), additive and multiplicative decomposition, moving averages, STL; growth rates and a first look at the ACF', 'Exponential smoothing (SES, Holt, Holt–Winters, ETS) and forecast evaluation: benchmarks, training and test sets, MAE, RMSE, MAPE, MASE, the M4 competition'],
                    ro: ['Organizarea cursului, evaluarea și instrumentele; serii de timp pe date reale (PIB, inflație, EUR/RON, BET, electricitate, CO2); o scurtă istorie, de la Yule și Slutsky la Box–Jenkins', 'Componente (trend, sezonalitate, ciclu, componenta neregulată), descompunere aditivă și multiplicativă, medii mobile, STL; rate de creștere și o primă privire asupra ACF', 'Netezirea exponențială (SES, Holt, Holt–Winters, ETS) și evaluarea prognozei: metode de referință, set de antrenare și set de test, MAE, RMSE, MAPE, MASE, competiția M4']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter0_introduction.pdf'), pdf('seminar', 'EN/Seminars/seminar0_introduction.pdf'),
                         nb('notebooks/EN/chapter0_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter0_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_00')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol0_introducere.pdf'), pdf('seminar', 'RO/Seminarii/seminar0_introducere_ro.pdf'),
                         nb('notebooks/EN/chapter0_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter0_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_00')]
                },
                quantinar: q('tsaPython')
            },
            {
                id: 'stationarity', num: 1,
                title: { en: 'Stochastic processes and stationarity', ro: 'Procese stochastice și staționaritate' },
                topics: {
                    en: ['Stochastic processes, mean, autocovariance and autocorrelation functions; strict and weak stationarity, ergodicity', 'White noise (weak, i.i.d., Gaussian), the random walk, the lag operator, differencing and the Wold decomposition', 'Sample ACF and PACF with confidence bands, Box–Pierce and Ljung–Box tests; log, differencing and Box–Cox transformations on Romanian GDP and inflation, EUR/RON, BET and S&P 500'],
                    ro: ['Procese stochastice, funcțiile de medie, autocovarianță și autocorelație; staționaritate strictă și slabă, ergodicitate', 'Zgomotul alb (slab, i.i.d., gaussian), mersul aleator, operatorul de decalaj, diferențierea și descompunerea Wold', 'ACF și PACF de selecție cu benzi de încredere, testele Box–Pierce și Ljung–Box; transformări (logaritm, diferențiere, Box–Cox) pentru PIB-ul și inflația României, EUR/RON, BET și S&P 500']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter1_stochastic_processes_stationarity.pdf'), pdf('seminar', 'EN/Seminars/seminar1_stochastic_processes_stationarity.pdf'),
                         nb('notebooks/EN/chapter1_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter1_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_01')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol1_procese_stochastice_stationaritate.pdf'), pdf('seminar', 'RO/Seminarii/seminar1_procese_stochastice_stationaritate_ro.pdf'),
                         nb('notebooks/EN/chapter1_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter1_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_01')]
                },
                quantinar: q('tsaPython')
            },
            {
                id: 'arma', num: 2,
                title: { en: 'ARMA models', ro: 'Modele ARMA' },
                topics: {
                    en: ['AR(p), MA(q) and ARMA(p,q): characteristic roots, stationarity, invertibility, ψ weights and impulse responses', 'Identification with the ACF and PACF; estimation by Yule–Walker, conditional least squares and maximum likelihood; AIC and BIC', 'Residual diagnostics (Ljung–Box with m−p−q degrees of freedom, Jarque–Bera), forecasts with intervals, the Box–Jenkins method on Romanian GDP and inflation, BET, EUR/RON and the sunspots'],
                    ro: ['AR(p), MA(q) și ARMA(p,q): rădăcini caracteristice, staționaritate, invertibilitate, ponderi ψ și răspunsuri la impuls', 'Identificarea cu ACF și PACF; estimarea prin Yule–Walker, cele mai mici pătrate condiționate și verosimilitate maximă; AIC și BIC', 'Diagnosticarea reziduurilor (Ljung–Box cu m−p−q grade de libertate, Jarque–Bera), prognoze cu intervale, metoda Box–Jenkins pentru PIB-ul și inflația României, BET, EUR/RON și petele solare']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter2_arma_models.pdf'), pdf('seminar', 'EN/Seminars/seminar2_arma_models.pdf'),
                         nb('notebooks/EN/chapter2_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter2_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_02')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol2_modele_arma.pdf'), pdf('seminar', 'RO/Seminarii/seminar2_modele_arma_ro.pdf'),
                         nb('notebooks/EN/chapter2_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter2_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_02')]
                },
                quantinar: q('tsaPython')
            },
            {
                id: 'arima', num: 3,
                title: { en: 'Unit roots and ARIMA models', ro: 'Rădăcini unitare și modele ARIMA' },
                topics: {
                    en: ['Deterministic and stochastic trends, integrated processes I(d), spurious regression (Yule, Granger–Newbold)', 'Unit-root and stationarity tests: Dickey–Fuller and ADF (deterministic terms, lags, critical values), Phillips–Perron, KPSS and their joint use; structural breaks (Perron, Zivot–Andrews)', 'ARIMA(p,d,q): identification, estimation, diagnostics, forecasts with widening intervals, over-differencing, automatic ARIMA used with care'],
                    ro: ['Trend determinist și trend stochastic, procese integrate I(d), regresia falsă (Yule, Granger–Newbold)', 'Teste de rădăcină unitară și de staționaritate: Dickey–Fuller și ADF (termeni determiniști, decalaje, valori critice), Phillips–Perron, KPSS și folosirea lor împreună; rupturi structurale (Perron, Zivot–Andrews)', 'Modele ARIMA(p,d,q): identificare, estimare, diagnosticare, prognoze cu intervale tot mai largi, supradiferențiere, selecția automată folosită cu grijă']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter3_unit_roots_arima_models.pdf'), pdf('seminar', 'EN/Seminars/seminar3_unit_roots_arima_models.pdf'),
                         nb('notebooks/EN/chapter3_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter3_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_03')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf'), pdf('seminar', 'RO/Seminarii/seminar3_radacini_unitare_modele_arima_ro.pdf'),
                         nb('notebooks/EN/chapter3_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter3_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_03')]
                },
                quantinar: q('tsaPython')
            },
            {
                id: 'seasonal', num: 4,
                title: { en: 'Seasonality and forecasting: SARIMA, TBATS, Prophet', ro: 'Sezonalitate și prognoză: SARIMA, TBATS, Prophet' },
                topics: {
                    en: ['Seasonal differencing and seasonal unit roots (HEGY, Canova–Hansen, OCSB); SARIMA(p,d,q)(P,D,Q)s models, the airline model and Romanian GDP', 'Seasonal adjustment in practice (X-13ARIMA-SEATS, Eurostat NSA and SCA series), Fourier terms and calendar effects (working days, Orthodox Easter)', 'Multiple seasonality in electricity load (MSTL, dynamic harmonic regression, TBATS, Prophet); time-series cross-validation, MASE, the Diebold–Mariano test and forecast combination'],
                    ro: ['Diferențierea sezonieră și rădăcinile unitare sezoniere (HEGY, Canova–Hansen, OCSB); modele SARIMA(p,d,q)(P,D,Q)s, modelul airline și PIB-ul României', 'Ajustarea sezonieră în practică (X-13ARIMA-SEATS, seriile Eurostat NSA și SCA), termeni Fourier și efecte de calendar (zile lucrătoare, Paștele ortodox)', 'Sezonalitatea multiplă a consumului de electricitate (MSTL, regresia armonică dinamică, TBATS, Prophet); validarea încrucișată pentru serii de timp, MASE, testul Diebold–Mariano și combinarea prognozelor']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter4_seasonality_forecasting.pdf'), pdf('seminar', 'EN/Seminars/seminar4_seasonality_forecasting.pdf'),
                         nb('notebooks/EN/chapter4_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter4_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_04')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol4_sezonalitate_prognoza.pdf'), pdf('seminar', 'RO/Seminarii/seminar4_sezonalitate_prognoza_ro.pdf'),
                         nb('notebooks/EN/chapter4_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter4_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_04')]
                },
                quantinar: q('tsaPython')
            },
            {
                id: 'garch', num: 5,
                title: { en: 'Conditional volatility: ARCH and GARCH', ro: 'Volatilitate condiționată: ARCH și GARCH' },
                topics: {
                    en: ['Stylised facts of returns: volatility clustering, heavy tails, the ACF of squared returns and the ARCH-LM test', 'ARCH(q), GARCH(1,1) and ARMA-GARCH: persistence, half-life, long-run variance, IGARCH and EWMA; maximum-likelihood estimation with the arch package and Student-t innovations', 'Asymmetry (GJR-GARCH, EGARCH, news impact curve), diagnostics on standardised residuals, variance forecasts evaluated with QLIKE and Diebold–Mariano, VaR 1%'],
                    ro: ['Faptele stilizate ale randamentelor: volatility clustering, cozi groase, ACF al randamentelor la pătrat și testul ARCH-LM', 'ARCH(q), GARCH(1,1) și ARMA-GARCH: persistență, timp de înjumătățire, varianța pe termen lung, IGARCH și EWMA; estimarea prin verosimilitate maximă cu pachetul arch și inovații Student-t', 'Asimetria (GJR-GARCH, EGARCH, curba de impact a știrilor), diagnosticarea pe reziduurile standardizate, prognoza varianței evaluată cu QLIKE și Diebold–Mariano, VaR 1%']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter5_conditional_volatility_garch.pdf'), pdf('seminar', 'EN/Seminars/seminar5_conditional_volatility_garch.pdf'),
                         nb('notebooks/EN/chapter5_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter5_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_05')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf'), pdf('seminar', 'RO/Seminarii/seminar5_volatilitate_conditionata_garch_ro.pdf'),
                         nb('notebooks/EN/chapter5_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter5_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_05')]
                },
                quantinar: q('tsaPython', 'statRisk')
            },
            {
                id: 'var', num: 6,
                title: { en: 'VAR models and Granger causality', ro: 'Modele VAR și cauzalitate Granger' },
                topics: {
                    en: ['Vector processes and cross-correlations; the VAR(p) model: stability (companion matrix), OLS estimation, lag selection by AIC, BIC and HQ, residual diagnostics', 'Granger causality: F and Wald tests and their pitfalls (omitted variables, instantaneous causality, non-stationarity); Romanian GDP, inflation and ROBOR; S&P 500, DAX and BET', 'Impulse responses (Cholesky ordering, generalised), forecast error variance decomposition, the Diebold–Yilmaz spillover index, VAR forecasts against univariate benchmarks; structural VARs and Stock–Watson (2001)'],
                    ro: ['Procese vectoriale și corelații încrucișate; modelul VAR(p): stabilitatea (matricea companion), estimarea prin OLS, alegerea numărului de decalaje după AIC, BIC și HQ, diagnosticarea reziduurilor', 'Cauzalitatea Granger: testele F și Wald și capcanele lor (variabile omise, cauzalitate instantanee, nestaționaritate); PIB, inflație și ROBOR în România; S&P 500, DAX și BET', 'Funcții de răspuns la impuls (ordinea Cholesky, răspunsuri generalizate), descompunerea varianței erorii de prognoză, indicele de spillover Diebold–Yilmaz, prognoze VAR comparate cu metode univariate; VAR-uri structurale și Stock–Watson (2001)']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter6_var_models_granger_causality.pdf'), pdf('seminar', 'EN/Seminars/seminar6_var_models_granger_causality.pdf'),
                         nb('notebooks/EN/chapter6_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter6_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_06')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol6_modele_var_cauzalitate_granger.pdf'), pdf('seminar', 'RO/Seminarii/seminar6_modele_var_cauzalitate_granger_ro.pdf'),
                         nb('notebooks/EN/chapter6_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter6_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_06')]
                },
                quantinar: q('tsaPython')
            },
            {
                id: 'cointegration', num: 7,
                title: { en: 'Cointegration and VECM', ro: 'Cointegrare și VECM' },
                topics: {
                    en: ['Cointegration as common stochastic trends (the drunk and her dog); the Engle–Granger two-step method with MacKinnon critical values and the Phillips–Ouliaris test; error correction models, speed of adjustment and half-life', 'The VECM and the Granger representation theorem; Johansen trace and maximum-eigenvalue tests and deterministic terms; cointegrating vectors, adjustment coefficients and weak exogeneity; VECM against a VAR in differences out of sample', 'Applications: the US term structure, purchasing power parity for EUR/RON, ROBOR and Euribor, Central European currencies, pairs trading on the Bucharest Stock Exchange out of sample and after costs'],
                    ro: ['Cointegrarea ca trend stochastic comun (exemplul bețivei și al cîinelui ei); metoda Engle–Granger în doi pași, cu valorile critice MacKinnon, și testul Phillips–Ouliaris; modele cu corecția erorii, viteza de ajustare și timpul de înjumătățire', 'Modelul VECM și teorema de reprezentare a lui Granger; testele Johansen ale urmei și ale valorii proprii maxime, termenii determiniști; vectori de cointegrare, coeficienți de ajustare și exogenitate slabă; VECM comparat cu un VAR în diferențe, în afara eșantionului', 'Aplicații: structura la termen a ratelor dobînzii din SUA, paritatea puterii de cumpărare pentru EUR/RON, ROBOR și Euribor, monedele central-europene, pairs trading la Bursa de Valori București, în afara eșantionului și după costuri']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter7_cointegration_vecm.pdf'), pdf('seminar', 'EN/Seminars/seminar7_cointegration_vecm.pdf'),
                         nb('notebooks/EN/chapter7_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter7_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_07')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol7_cointegrare_vecm.pdf'), pdf('seminar', 'RO/Seminarii/seminar7_cointegrare_vecm_ro.pdf'),
                         nb('notebooks/EN/chapter7_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter7_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_07')]
                },
                quantinar: q('tsaPython')
            },
            {
                id: 'long-memory', num: 8,
                title: { en: 'Long memory and ARFIMA', ro: 'Memorie lungă și ARFIMA' },
                topics: {
                    en: ['Short and long memory: exponential and hyperbolic ACF decay, the spectral pole at zero, the Hurst exponent; the Nile, Romanian and US inflation, US unemployment', 'Fractional differencing (1−L)^d and ARFIMA(p,d,q): ranges of d (stationarity, invertibility, mean reversion), fBm and fGn; estimation by R/S, DFA, GPH, local Whittle and maximum likelihood; ARFIMA forecasts against AR', 'Long memory in volatility (|r|, realised volatility, FIGARCH, HAR) for the S&P 500, BET and EUR/RON; spurious long memory from breaks and regime shifts'],
                    ro: ['Memorie scurtă și memorie lungă: descreșterea exponențială și hiperbolică a ACF, polul spectral la zero, exponentul Hurst; Nilul, inflația din România și din SUA, șomajul din SUA', 'Diferențierea fracționară (1−L)^d și modelele ARFIMA(p,d,q): intervalele lui d (staționaritate, invertibilitate, revenire la medie), fBm și fGn; estimarea prin R/S, DFA, GPH, Whittle local și verosimilitate maximă; prognoze ARFIMA comparate cu AR', 'Memoria lungă a volatilității (|r|, volatilitate realizată, FIGARCH, HAR) pentru S&P 500, BET și EUR/RON; memoria lungă aparentă, produsă de rupturi și schimbări de regim']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter8_long_memory_arfima.pdf'), pdf('seminar', 'EN/Seminars/seminar8_long_memory_arfima.pdf'),
                         nb('notebooks/EN/chapter8_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter8_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_08')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol8_memorie_lunga_arfima.pdf'), pdf('seminar', 'RO/Seminarii/seminar8_memorie_lunga_arfima_ro.pdf'),
                         nb('notebooks/EN/chapter8_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter8_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_08')]
                },
                quantinar: q('cryptoEfficiency')
            },
            {
                id: 'ml', num: 9,
                title: { en: 'Machine learning for time series', ro: 'Învățare automată pentru serii de timp' },
                topics: {
                    en: ['Forecasting as supervised learning: lags, rolling windows and calendar features; recursive, direct and multi-output strategies; walk-forward validation and leakage', 'Ridge and lasso, regression trees, random forest, gradient boosting, small neural networks and the LSTM; prediction intervals by quantile regression and conformal prediction', 'Machine learning against ARIMA, ETS and naive benchmarks with MASE and Diebold–Mariano: Romanian electricity load and inflation (local and global models), realised volatility, the sign of returns; the M4 and M5 competitions'],
                    ro: ['Prognoza ca învățare supervizată: decalaje, ferestre mobile și variabile de calendar; strategiile recursivă, directă și cu ieșiri multiple; validarea walk-forward și scurgerea de informație', 'Ridge și lasso, arbori de regresie, random forest, gradient boosting, rețele neuronale mici și LSTM; intervale de prognoză prin regresie cuantilică și predicție conformală', 'Machine learning comparat cu ARIMA, ETS și metodele naive, cu MASE și Diebold–Mariano: consumul de energie electrică și inflația din România (modele locale și globale), volatilitatea realizată, semnul randamentelor; competițiile M4 și M5']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter9_machine_learning_time_series.pdf'), pdf('seminar', 'EN/Seminars/seminar9_machine_learning_time_series.pdf'),
                         nb('notebooks/EN/chapter9_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter9_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_09')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol9_invatare_automata_serii_timp.pdf'), pdf('seminar', 'RO/Seminarii/seminar9_invatare_automata_serii_timp_ro.pdf'),
                         nb('notebooks/EN/chapter9_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter9_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_09')]
                },
                quantinar: q('rf', 'mlRisk')
            },
            {
                id: 'state-space', num: 10,
                title: { en: 'State space models, Kalman filter and Markov switching', ro: 'Modele în spațiul stărilor, filtrul Kalman și modele Markov switching' },
                topics: {
                    en: ['The state space form: measurement and transition equations; local level, local linear trend, ARMA and time-varying regression; the Kalman filter by hand and exponential smoothing as its steady state', 'Smoothing, likelihood, diagnostics and missing data (the Nile); trend and cycle of Romanian GDP with unobserved components, the HP filter and the Hamilton (2018) critique; dynamic factors and nowcasting', 'Markov-switching models (Hamilton, 1989): transition probabilities, durations, filtered and smoothed regime probabilities; US recessions against the NBER dates, Romanian growth regimes, volatility regimes, links to GARCH and long memory'],
                    ro: ['Forma în spațiul stărilor: ecuația de măsurare și ecuația de tranziție; local level, local linear trend, ARMA și regresia cu parametri variabili; filtrul Kalman calculat de mînă și netezirea exponențială ca stare de echilibru a acestuia', 'Netezirea, verosimilitatea, diagnosticarea și valorile lipsă (Nilul); trendul și ciclul PIB-ului României cu componente neobservate, filtrul HP și critica lui Hamilton (2018); factori dinamici și nowcasting', 'Modele Markov switching (Hamilton, 1989): probabilități de tranziție, durate, probabilitățile filtrate și netezite ale regimurilor; recesiunile din SUA comparate cu datările NBER, regimuri ale creșterii în România, regimuri de volatilitate, legături cu GARCH și memoria lungă']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter10_state_space_kalman_markov_switching.pdf'), pdf('seminar', 'EN/Seminars/seminar10_state_space_kalman_markov_switching.pdf'),
                         nb('notebooks/EN/chapter10_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter10_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_10')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol10_spatiul_starilor_kalman_markov_switching.pdf'), pdf('seminar', 'RO/Seminarii/seminar10_spatiul_starilor_kalman_markov_switching_ro.pdf'),
                         nb('notebooks/EN/chapter10_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter10_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_10')]
                },
                quantinar: q('kalman')
            },
            {
                id: 'foundation-models', num: 11, selfStudy: true,
                title: { en: 'Foundation models for time series', ro: 'Foundation models pentru serii de timp' },
                topics: {
                    en: ['Foundation models and zero-shot forecasting: attention, mean scaling, quantisation and patching; Chronos, Chronos-Bolt, Chronos-2, TimesFM, Moirai, Lag-Llama (TimeGPT as a closed API)', 'Probabilistic forecasts and their scores: pinball loss, CRPS and weighted quantile loss, MASE, coverage of the 80% interval', 'An honest test against seasonal naive, ETS and ARIMA on Romanian and US series; context length, model size, leakage and benchmark contamination'],
                    ro: ['Foundation models și prognoza zero-shot: atenția, scalarea prin medie, cuantizarea și patching-ul; Chronos, Chronos-Bolt, Chronos-2, TimesFM, Moirai, Lag-Llama (TimeGPT ca API închis)', 'Prognozele probabiliste și scorurile lor: pierderea pinball, CRPS și pierderea cuantilică ponderată, MASE, acoperirea intervalului de 80%', 'Un test corect față de prognoza sezonieră naivă, ETS și ARIMA pe serii din România și SUA; lungimea contextului, dimensiunea modelului, scurgerea de informație și contaminarea benchmark-urilor']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter11_foundation_models_time_series.pdf'),
                         nb('notebooks/EN/chapter11_lecture_notebook.ipynb', NB_LECT), ql('Quantlets/Ch_11')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol11_foundation_models_serii_timp.pdf'),
                         nb('notebooks/EN/chapter11_lecture_notebook.ipynb', NB_LECT), ql('Quantlets/Ch_11')]
                },
                quantinar: q('nextWord')
            },
            {
                id: 'spectral', num: 12, selfStudy: true,
                title: { en: 'Spectral analysis', ro: 'Analiză spectrală' },
                topics: {
                    en: ['Cycles and Fourier frequencies, the periodogram and the spectral density of white noise and ARMA models; aliasing and leakage', 'Estimating the spectrum: tapering, Daniell and Welch smoothing, chi-square confidence bands, Fisher\'s test; cycles in sunspots, US and Romanian GDP and the Romanian electricity load', 'The spectral pole of long memory, filter gains (differencing, Hodrick–Prescott), coherence and phase between two series; wavelets as a pointer'],
                    ro: ['Cicluri și frecvențe Fourier, periodograma și densitatea spectrală a zgomotului alb și a modelelor ARMA; aliasing și scurgerea spectrală', 'Estimarea spectrului: ferestre de atenuare, netezirea Daniell și Welch, benzi de încredere chi-pătrat, testul lui Fisher; cicluri în petele solare, în PIB-ul SUA și al României și în consumul de energie electrică al României', 'Polul spectral al memoriei lungi, cîștigul filtrelor (diferențiere, Hodrick–Prescott), coerența și faza a două serii; o trimitere spre wavelets']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter12_spectral_analysis.pdf'),
                         nb('notebooks/EN/chapter12_lecture_notebook.ipynb', NB_LECT), ql('Quantlets/Ch_12')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol12_analiza_spectrala.pdf'),
                         nb('notebooks/EN/chapter12_lecture_notebook.ipynb', NB_LECT), ql('Quantlets/Ch_12')]
                },
                quantinar: q('xfg')
            },
            {
                id: 'lppl', num: 13, selfStudy: true,
                title: { en: 'Speculative bubbles: LPPL models', ro: 'Bule speculative: modele LPPL' },
                topics: {
                    en: ['Bubbles in history and their statistical signatures; rational bubbles and explosive roots: right-tailed ADF, SADF, GSADF and BSADF date-stamping (Nasdaq 100, Bitcoin, BET, Shanghai)', 'The LPPL model of Johansen, Ledoit and Sornette: super-exponential growth, log-periodic oscillations, the critical time; Filimonov–Sornette two-step estimation and the filter conditions', 'The LPPLS confidence indicator from many windows; an honest evaluation of crash prediction with false alarms and base rates (S&P 500, Bitcoin)'],
                    ro: ['Bule în istorie și semnăturile lor statistice; bule raționale și rădăcini explozive: ADF la dreapta, SADF, GSADF și datarea BSADF (Nasdaq 100, Bitcoin, BET, Shanghai)', 'Modelul LPPL al lui Johansen, Ledoit și Sornette: creștere superexponențială, oscilații log-periodice, timpul critic; estimarea în doi pași Filimonov–Sornette și condițiile de filtrare', 'Indicatorul de încredere LPPLS din multe ferestre; o evaluare onestă a prognozei crahurilor, cu alarme false și frecvențe de bază (S&P 500, Bitcoin)']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter13_bubbles_lppl.pdf'),
                         nb('notebooks/EN/chapter13_lecture_notebook.ipynb', NB_LECT), ql('Quantlets/Ch_13')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol13_bule_lppl.pdf'),
                         nb('notebooks/EN/chapter13_lecture_notebook.ipynb', NB_LECT), ql('Quantlets/Ch_13')]
                },
                quantinar: q('sfm')
            },
            {
                id: 'mgarch', num: 14, selfStudy: true,
                title: { en: 'Multivariate GARCH models', ro: 'Modele GARCH multivariate' },
                topics: {
                    en: ['Why multivariate volatility: portfolio risk, hedging, contagion; rolling correlations of the S&P 500, DAX, BET and Bitcoin; asynchronous trading', 'The conditional covariance matrix; VEC, BEKK, CCC and DCC models and their number of parameters; two-step DCC estimation and the test against constant correlation', 'Applications: correlations in the crises of 2008 and 2020, a five-asset DCC with BET and EUR/RON, portfolio VaR 1% with backtesting, dynamic hedge ratios'],
                    ro: ['Volatilitatea multivariată și utilitatea ei: riscul portofoliului, acoperirea riscului, contagiunea; corelațiile pe ferestre mobile ale S&P 500, DAX, BET și Bitcoin; tranzacționarea asincronă', 'Matricea de covarianță condiționată; modelele VEC, BEKK, CCC și DCC și numărul lor de parametri; estimarea DCC în doi pași și testul față de corelația constantă', 'Aplicații: corelațiile în crizele din 2008 și 2020, un DCC cu cinci active, inclusiv BET și EUR/RON, VaR 1% al unui portofoliu, cu backtesting, rapoarte dinamice de acoperire']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter14_multivariate_garch_models.pdf'),
                         nb('notebooks/EN/chapter14_lecture_notebook.ipynb', NB_LECT), ql('Quantlets/Ch_14')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol14_modele_garch_multivariate.pdf'),
                         nb('notebooks/EN/chapter14_lecture_notebook.ipynb', NB_LECT), ql('Quantlets/Ch_14')]
                },
                quantinar: q('statRisk')
            },
            {
                id: 'review', num: 15,
                title: { en: 'Review and exam preparation', ro: 'Recapitulare și pregătire pentru examen' },
                topics: {
                    en: ['The course map and one recap per chapter: key formulas, empirical facts on Romanian and international series, common mistakes; Chapters 11–14 for self-study', 'The toolbox and the Box–Jenkins method from start to finish on Romanian inflation: tests, identification, SARIMA, diagnostics, out-of-sample evaluation', 'The written exam (70%): format, grading and eight worked problems on software output; the team project (20%), its oral defence and AI_USE.md; attendance (10%)'],
                    ro: ['Harta cursului și cîte o recapitulare pentru fiecare capitol: formule-cheie, fapte empirice pe serii românești și internaționale, greșeli frecvente; capitolele 11–14 pentru studiu individual', 'Trusa de instrumente și metoda Box–Jenkins de la un capăt la altul pe inflația din România: teste, identificare, SARIMA, diagnosticare, evaluare în afara eșantionului', 'Examenul scris (70%): format, criterii și opt probleme rezolvate pe rezultate obținute cu software; proiectul de echipă (20%), susținerea orală și AI_USE.md; prezența (10%)']
                },
                links: {
                    en: [pdf('slides', 'EN/Courses/chapter15_review.pdf'), pdf('seminar', 'EN/Seminars/seminar15_review.pdf'), pdf('seminarExtra', 'exam/practice/exam_problems_en.pdf', { en: 'Exam practice problems', ro: 'Probleme de examen' }),
                         nb('notebooks/EN/chapter15_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter15_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_15')],
                    ro: [pdf('slides', 'RO/Cursuri/capitol15_recapitulare.pdf'), pdf('seminar', 'RO/Seminarii/seminar15_recapitulare_ro.pdf'), pdf('seminarExtra', 'exam/practice/probleme_examen_ro.pdf', { en: 'Exam practice problems', ro: 'Probleme de examen' }),
                         nb('notebooks/EN/chapter15_lecture_notebook.ipynb', NB_LECT), nb('notebooks/EN/chapter15_seminar_notebook.ipynb', NB_SEM), ql('Quantlets/Ch_15')]
                }
            }
        ],

        // ---------------------------------------------------------------
        // Team project (section #project) and AI policy
        // ---------------------------------------------------------------
        project: {
            en: [
                { h: 'Content', p: ['A team analysis (2–4 students) of real time series with the methods of the course: decomposition and smoothing, ARIMA/SARIMA, GARCH, VAR or VECM, Granger causality, impulse responses and forecast evaluation.', 'Romanian data (INS, BNR, Eurostat, BVB) are encouraged. The project counts for 20% of the final grade.'] },
                { h: 'Deliverables', p: ['A GitHub repository whose code reproduces every number and chart from the data.', 'A short report and a presentation of the results, followed by an oral defence.'] },
                { h: 'Grading criteria', p: ['A clear question, correct methods and diagnostics, honest out-of-sample evaluation, and the interpretation of the results.', 'Each member must be able to explain the code and the results.'] }
            ],
            ro: [
                { h: 'Conținut', p: ['O analiză în echipă (2–4 studenți) a unor serii de timp reale cu metodele cursului: descompunere și netezire, ARIMA/SARIMA, GARCH, VAR sau VECM, cauzalitate Granger, funcții de răspuns la impuls și evaluarea prognozei.', 'Sînt recomandate datele românești (INS, BNR, Eurostat, BVB). Proiectul reprezintă 20% din nota finală.'] },
                { h: 'Livrabile', p: ['Un repository GitHub cu codul care reproduce, din date, fiecare rezultat numeric și fiecare grafic.', 'Un raport scurt și o prezentare a rezultatelor, urmate de susținerea orală.'] },
                { h: 'Criterii de evaluare', p: ['Claritatea întrebării, corectitudinea metodelor și a diagnosticării, evaluarea corectă în afara eșantionului și interpretarea rezultatelor.', 'Fiecare membru trebuie să poată explica codul și rezultatele.'] }
            ]
        },
        aiPolicy: {
            en: ['AI tools are allowed and must be declared in AI_USE.md (tool, prompts, what was kept)', 'Every number, every piece of code and every reference produced with AI is checked by the team', 'The oral defence of the project checks that each member understands the code and the results', 'Each chapter ends with a short section on the possible contribution of AI to its topic'],
            ro: ['Instrumentele AI sînt permise și se declară în AI_USE.md (instrument, prompturi, ce s-a păstrat)', 'Fiecare rezultat numeric, fiecare secvență de cod și fiecare referință obținute cu AI sînt verificate de echipă', 'Susținerea orală a proiectului verifică dacă fiecare membru înțelege codul și rezultatele', 'Fiecare capitol se încheie cu o secțiune scurtă despre contribuția posibilă a AI la tema lui']
        },

        // ---------------------------------------------------------------
        // Resources
        // ---------------------------------------------------------------
        resources: [
            { icon: '&#128187;', href: REPO, en: ['GitHub Repository', 'Slides, seminars, notebooks and Quantlets'], ro: ['Repository GitHub', 'Slide-uri, seminarii, notebook-uri și Quantlets'] },
            { icon: '&#128214;', href: 'https://otexts.com/fpp3/', en: ['FPP3 (free online)', 'Hyndman &amp; Athanasopoulos, Forecasting: Principles and Practice'], ro: ['FPP3 (gratuit, online)', 'Hyndman și Athanasopoulos, Forecasting: Principles and Practice'] },
            { icon: '&#127891;', img: 'logos/qr_logo.png', href: 'https://quantinar.com', en: ['Quantinar', 'P2P platform with advanced courses'], ro: ['Quantinar', 'Platformă P2P cu cursuri avansate'] },
            { icon: '&#128190;', img: 'logos/ql_logo.png', href: 'https://quantlet.com', en: ['Quantlet', 'Reproducible code for every chart'], ro: ['Quantlet', 'Cod reproductibil pentru fiecare grafic'] },
            { icon: '&#127963;', img: 'logos/ida_square.png', href: 'https://theida.net', en: ['IDA', 'Institute for Digital Assets'], ro: ['IDA', 'Institute for Digital Assets'] }
        ],

        dataSources: [
            { name: 'INS TEMPO', href: 'http://statistici.insse.ro:8077/tempo-online/', en: 'National Institute of Statistics (Romania): GDP, prices, unemployment', ro: 'Institutul Național de Statistică: PIB, prețuri, șomaj' },
            { name: 'BNR', href: 'https://www.bnr.ro', en: 'National Bank of Romania: reference exchange rates, interest rates', ro: 'Banca Națională a României: cursuri de referință, dobînzi' },
            { name: 'Eurostat', href: 'https://ec.europa.eu/eurostat/data/database', en: 'European macroeconomic series (GDP, HICP, unemployment)', ro: 'Serii macroeconomice europene (PIB, IAPC, șomaj)' },
            { name: 'FRED', href: 'https://fred.stlouisfed.org', en: 'US macro and interest-rate data (St. Louis Fed)', ro: 'Date macroeconomice și de dobîndă pentru SUA (St. Louis Fed)' },
            { name: 'BVB', href: 'https://www.bvb.ro', en: 'Bucharest Stock Exchange', ro: 'Bursa de Valori București' },
            { name: 'statsmodels datasets', href: 'https://www.statsmodels.org/stable/datasets/index.html', en: 'Classic textbook series (sunspots, CO2, US macro data)', ro: 'Serii clasice din manuale (pete solare, CO2, date macroeconomice SUA)' }
        ],

        bibliography: [
            'Huang, C., &amp; Petukhina, A. (2022). <a href="https://doi.org/10.1007/978-3-031-13584-2" target="_blank" rel="noopener"><em>Applied Time Series Analysis and Forecasting with Python</em></a>. Springer.',
            'Hyndman, R. J., &amp; Athanasopoulos, G. (2021). <a href="https://otexts.com/fpp3/" target="_blank" rel="noopener"><em>Forecasting: Principles and Practice</em></a> (3rd ed.). OTexts.',
            'Brockwell, P. J., &amp; Davis, R. A. (2016). <a href="https://doi.org/10.1007/978-3-319-29854-2" target="_blank" rel="noopener"><em>Introduction to Time Series and Forecasting</em></a> (3rd ed.). Springer.',
            'Hamilton, J. D. (1994). <a href="https://doi.org/10.2307/j.ctv14jx6sm" target="_blank" rel="noopener"><em>Time Series Analysis</em></a>. Princeton University Press.',
            'Box, G. E. P., Jenkins, G. M., Reinsel, G. C., &amp; Ljung, G. M. (2015). <a href="https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021" target="_blank" rel="noopener"><em>Time Series Analysis: Forecasting and Control</em></a> (5th ed.). Wiley.',
            'Shumway, R. H., &amp; Stoffer, D. S. (2017). <a href="https://doi.org/10.1007/978-3-319-52452-8" target="_blank" rel="noopener"><em>Time Series Analysis and Its Applications</em></a> (4th ed.). Springer.',
            'Tsay, R. S. (2010). <a href="https://doi.org/10.1002/9780470644560" target="_blank" rel="noopener"><em>Analysis of Financial Time Series</em></a> (3rd ed.). Wiley.',
            'Durbin, J., &amp; Koopman, S. J. (2012). <a href="https://doi.org/10.1093/acprof:oso/9780199641178.001.0001" target="_blank" rel="noopener"><em>Time Series Analysis by State Space Methods</em></a> (2nd ed.). Oxford University Press.',
            'Bollerslev, T. (1986). <a href="https://doi.org/10.1016/0304-4076(86)90063-1" target="_blank" rel="noopener">Generalized autoregressive conditional heteroskedasticity</a>. <em>Journal of Econometrics</em>, 31(3), 307–327.',
            'De Livera, A. M., Hyndman, R. J., &amp; Snyder, R. D. (2011). <a href="https://doi.org/10.1198/jasa.2011.tm09771" target="_blank" rel="noopener">Forecasting time series with complex seasonal patterns using exponential smoothing</a>. <em>Journal of the American Statistical Association</em>, 106(496), 1513–1527.',
            'Dickey, D. A., &amp; Fuller, W. A. (1979). <a href="https://doi.org/10.1080/01621459.1979.10482531" target="_blank" rel="noopener">Distribution of the estimators for autoregressive time series with a unit root</a>. <em>Journal of the American Statistical Association</em>, 74(366), 427–431.',
            'Engle, R. F. (1982). <a href="https://doi.org/10.2307/1912773" target="_blank" rel="noopener">Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation</a>. <em>Econometrica</em>, 50(4), 987–1007.',
            'Engle, R. F., &amp; Granger, C. W. J. (1987). <a href="https://doi.org/10.2307/1913236" target="_blank" rel="noopener">Co-integration and error correction: representation, estimation, and testing</a>. <em>Econometrica</em>, 55(2), 251–276.',
            'Granger, C. W. J., &amp; Joyeux, R. (1980). <a href="https://doi.org/10.1111/j.1467-9892.1980.tb00297.x" target="_blank" rel="noopener">An introduction to long-memory time series models and fractional differencing</a>. <em>Journal of Time Series Analysis</em>, 1(1), 15–29.',
            'Hamilton, J. D. (1989). <a href="https://doi.org/10.2307/1912559" target="_blank" rel="noopener">A new approach to the economic analysis of nonstationary time series and the business cycle</a>. <em>Econometrica</em>, 57(2), 357–384.',
            'Johansen, S. (1991). <a href="https://doi.org/10.2307/2938278" target="_blank" rel="noopener">Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models</a>. <em>Econometrica</em>, 59(6), 1551–1580.',
            'Kalman, R. E. (1960). <a href="https://doi.org/10.1115/1.3662552" target="_blank" rel="noopener">A new approach to linear filtering and prediction problems</a>. <em>Journal of Basic Engineering</em>, 82(1), 35–45.',
            'Sims, C. A. (1980). <a href="https://doi.org/10.2307/1912017" target="_blank" rel="noopener">Macroeconomics and reality</a>. <em>Econometrica</em>, 48(1), 1–48.',
            'Taylor, S. J., &amp; Letham, B. (2018). <a href="https://doi.org/10.1080/00031305.2017.1380080" target="_blank" rel="noopener">Forecasting at scale</a>. <em>The American Statistician</em>, 72(1), 37–45.'
        ],

        contact: {
            name: 'Prof. dr. Daniel Traian Pele',
            email: 'danpele@ase.ro',
            en: ['Bucharest University of Economic Studies', 'Department of Statistics and Econometrics', 'Faculty of Cybernetics, Statistics and Economic Informatics'],
            ro: ['Academia de Studii Economice din București', 'Departamentul de Statistică și Econometrie', 'Facultatea de Cibernetică, Statistică și Informatică Economică'],
            // TODO: seminar instructor not yet decided. Fill in name and e-mail;
            // the Seminar card stays hidden while the name starts with 'TODO'.
            seminar: { name: 'TODO_SEMINAR_INSTRUCTOR', email: '' }
        },

        footerLogos: [
            ['https://www.ase.ro', 'logos/ase_logo.png', 'ASE'],
            ['https://www.theida.net/', 'logos/ida_logo.png', 'IDA'],
            ['https://quantinar.com', 'logos/qr_logo.png', 'Quantinar'],
            ['https://quantlet.com', 'logos/ql_logo.png', 'Quantlet'],
            ['https://ai4efin.ase.ro', 'logos/ai4efin_logo.png', 'AI4EFin'],
            ['https://www.digital-finance-msca.com/', 'logos/msca_logo.png', 'MSCA Digital Finance'],
            ['https://blockchain-research-center.com/', 'logos/brc_logo.png', 'Blockchain Research Center'],
            ['https://ipe.ro/new/', 'logos/acad_logo.png', 'Romanian Academy']
        ],

        // Quiz banks register themselves here by chapter id (see assets/quizzes/<id>.js)
        quizzes: {}
    };
})();
