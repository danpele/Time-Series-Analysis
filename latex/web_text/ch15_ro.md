<!--
Textul de curs al versiunii web, capitolul 15 (RO). Generat în pagină de latex/build_web_lecture.py.
Format: # titlul secțiunii (introducere), ## titlul slide-ului {h=hash} (text), @after (text după blocul interactiv), ## @end (sinteză).
Ecuații numerotate: $$ ... $$ {#eq:eticheta}, trimitere @eq:eticheta. Citări: @Cheie -> Autor (an), [@Cheie] -> (Autor, an).
Hash-urile curente: python3 build_web_lecture.py 15 --hashes
-->

## @intro

@text
Cincisprezece capitole au tratat o serie de timp din unghiuri diferite: componentele ei, staționaritatea, dependența de propriul trecut, rădăcinile unitare, sezonalitatea, varianța care se schimbă în timp, legăturile cu alte serii, memoria lungă, metodele de învățare automată și stările ascunse. Fiecare capitol a adus propriile formule, propriile teste și propriile capcane. Întrebarea acestui capitol este ce rămîne din toate acestea cînd o serie reală trebuie analizată de la un capăt la altul: ce întrebare se pune întîi, ce instrument răspunde la ea și cum se citește rezultatul.

Capitolul are două părți. Prima recapitulează cursul: o hartă a capitolelor, cîte două slide-uri pentru fiecare capitol, cu formulele-cheie, faptele empirice principale și greșelile cele mai frecvente, apoi o sinteză a testelor și a modelelor care leagă fiecare întrebare despre o serie de testul sau de modelul potrivit. A doua parte aplică metoda Box–Jenkins, în șase pași, pe indicele armonizat al prețurilor de consum (IAPC) al României, din ianuarie 2005 pînă în august 2026: transformare, teste, identificare, estimare, verificarea reziduurilor și prognoză.

Exemplul arată de ce pașii nu pot fi săriți. Modelul sugerat de corelogramă nu trece verificarea reziduurilor, iar modelul ales, deși are cele mai mici erori de prognoză pe eșantionul de test, nu se separă statistic de metodele naive pe 24 de luni. Cele mai mari erori vin din decizii fiscale și din prețul energiei, pe care un model univariat nu le poate anticipa.

@chart tsa_ch15_bj_data
IAPC al României, ianuarie 2005 – august 2026. Sus: indicele, 2015 = 100. Jos: inflația lunară $100\Delta\ln P_t$ (bare) și inflația anuală $100\Delta_{12}\ln P_t$ (linia roșie), în procente.

@goals
- **Situați** fiecare capitol al cursului pe drumul de la descrierea unei serii la modelarea și prognoza ei.
- **Reamintiți** formula-cheie, faptul empiric principal și greșeala cea mai frecventă din fiecare capitol.
- **Alegeți** testul sau modelul potrivit pentru o întrebare dată despre o serie de timp.
- **Aplicați** metoda Box–Jenkins pe o serie reală și **interpretați** fiecare rezultat în cîteva fraze.
- **Evaluați** prognozele în afara eșantionului, față de repere simple, cu testul Diebold–Mariano.

@plan
- Harta cursului | Cum se leagă cele cincisprezece capitole și pe ce serii reale au fost ilustrate.
- Capitolele 0–3: descriere, staționaritate, ARMA, ARIMA | Componente și netezire, staționaritate și corelogramă, modele ARMA, rădăcini unitare.
- Capitolele 4–7: sezonalitate, volatilitate, VAR, VECM | SARIMA și evaluarea prognozelor, GARCH, VAR și cauzalitate Granger, cointegrare.
- Capitolele 8–10: memorie lungă, învățare automată, spațiul stărilor | ARFIMA, validarea walk-forward, filtrul Kalman și regimurile Markov.
- Capitolele 11–14: studiu individual | Foundation models, analiză spectrală, bule LPPL, GARCH multivariat.
- Sinteza testelor și a modelelor | De la întrebarea despre serie la test și la decizie; convențiile cursului.
- Box–Jenkins de la un capăt la altul: inflația din România | Șase pași pe IAPC: teste, corelogramă, candidați, reziduuri, prognoze.
- Proiectul de echipă și prezența | Conținutul proiectului, criteriile, susținerea orală și declararea utilizării AI.
- Contribuția posibilă a AI | Ce poate face un asistent și ce trebuie verificat.
- Rezumat | Ideile de reținut și autoevaluarea.

## Întrebarea de azi și traseul {h=f0ffea8f}

Întrebarea capitolului este ce spune întregul curs despre o serie de timp. Răspunsul nu este o listă de modele, ci o succesiune de întrebări: cum arată seria, este staționară, ce dependență are de propriul trecut, are sezonalitate, se schimbă varianța ei, se mișcă împreună cu alte serii și cît de bine poate fi prognozată. Fiecare capitol al cursului răspunde la una dintre aceste întrebări, iar recapitularea le pune în ordinea în care apar într-o analiză reală.

Traseul are trei părți. Prima trece prin harta cursului, prin cîte două slide-uri pentru fiecare capitol și prin sinteza testelor și a modelelor. A doua parcurge metoda Box–Jenkins pe inflația din România, pas cu pas. A treia descrie proiectul de echipă. [Capitolele 11](../RO/Cursuri/capitol11_foundation_models_serii_timp.pdf)–[14](../RO/Cursuri/capitol14_modele_garch_multivariate.pdf) sînt pentru studiu individual și apar în recapitulare doar prin ideile principale și prin legătura lor cu nucleul cursului.

## Rezultatele învățării {h=fbb728a0}

Primele trei rezultate privesc cursul ca întreg. Situarea unui capitol pe drumul de la descriere la prognoză înseamnă a ști ce întrebare a rămas deschisă la capitolul anterior și ce instrument nou o rezolvă: de exemplu, testele de rădăcină unitară din [Capitolul 3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf) există pentru că modelele ARMA din [Capitolul 2](../RO/Cursuri/capitol2_modele_arma.pdf) presupun o serie staționară. Pentru fiecare capitol merită reținute trei lucruri: formula-cheie, faptul empiric principal din datele cursului și greșeala care apare cel mai des în aplicații.

Celelalte rezultate sînt practice. Metoda Box–Jenkins leagă într-o singură analiză transformarea, testele, identificarea, estimarea, verificarea și prognoza. Citirea rezultatelor obținute cu un program, estimări, teste și corelograme, cere de fiecare dată aceeași structură: ce ipoteză este testată, ce arată statistica și ce înseamnă decizia pentru serie, în trei pînă la cinci fraze. Ultimul rezultat privește proiectul de echipă, susținerea lui orală și declararea folosirii instrumentelor de inteligență artificială (*artificial intelligence*, AI).

## Bibliografie și instrumente {h=5da01b8e}

Manualul cursului este @HuangPetukhina2022, care tratează metodele în Python; @HyndmanAthanasopoulos2021 este un manual însoțitor gratuit, centrat pe prognoză. Pentru teorie, @BrockwellDavis2016 și @Hamilton1994 dau demonstrațiile și proprietățile asimptotice ale estimatorilor, iar @BoxJenkinsReinsel2008 este sursa metodei aplicate în a doua parte a capitolului.

Quantlet-urile capitolului conțin codul cazului Box–Jenkins pe IAPC: fiecare tabel și fiecare grafic al secțiunii respective poate fi refăcut din date. Notebook-ul cursului adună tot cursul într-un singur fișier care rulează în Google Colab. Quiz-ul capitolului are 20 de întrebări extrase din toate capitolele și este util pentru a vedea ce idei trebuie recitite.

# Harta cursului

Secțiunea arată cum se leagă capitolele între ele și pe ce serii reale au fost ilustrate metodele.

## Harta cursului: Capitolele 0–14 {h=0585468c}

Diagrama așază capitolele pe patru rînduri. Primul rînd, o serie și media ei, merge de la [Capitolul 0](../RO/Cursuri/capitol0_introducere.pdf) (componente și netezire) la staționaritate și corelogramă, la modelele ARMA, la rădăcinile unitare și ARIMA și la sezonalitate și evaluarea prognozelor. Al doilea rînd, varianța și mai multe serii, conține volatilitatea ARCH și GARCH, modelele VAR și cointegrarea cu VECM. Chenarele roșii marchează acest nucleu, iar săgețile roșii arată dependențele principale: ARMA duce atît la ARIMA, cît și la GARCH, iar rădăcinile unitare duc la cointegrare.

Al treilea rînd conține extensiile, [Capitolele 8](../RO/Cursuri/capitol8_memorie_lunga_arfima.pdf)–[10](../RO/Cursuri/capitol10_spatiul_starilor_kalman_markov_switching.pdf): memoria lungă pornește de la volatilitate, învățarea automată de la VAR, iar spațiul stărilor de la sezonalitate și prognoză. Ultimul rînd, cu chenare punctate, conține capitolele pentru studiu individual. Fiecare este construit pe un capitol de bază: foundation models pe învățarea automată, analiza spectrală pe corelogramă și pe memoria lungă, bulele LPPL pe rădăcinile unitare, iar GARCH multivariat pe GARCH și pe VAR.

Citită de la stînga la dreapta, harta urmează ordinea unei analize: întîi descrierea, apoi întrebarea dacă seria este staționară, apoi un model pentru medie, apoi unul pentru varianță și, la final, modelarea mai multor serii împreună.

## Seriile cursului {h=e156372e}

Tabelul enumeră seriile pe care au fost ilustrate metodele, cu sursa lor, tema pe care o ilustrează și capitolele în care apar. Cele mai multe sînt românești. PIB-ul real trimestrial al României are trend, sezonalitate, o rădăcină unitară și schimbări de regim, deci servește în [Capitolele 0](../RO/Cursuri/capitol0_introducere.pdf)–[4](../RO/Cursuri/capitol4_sezonalitate_prognoza.pdf) și în [Capitolul 10](../RO/Cursuri/capitol10_spatiul_starilor_kalman_markov_switching.pdf). Inflația IAPC lunară ilustrează persistența, sezonalitatea și memoria lungă, iar ROBOR 3M (*Romanian Interbank Offer Rate*) și șomajul intră, alături de creștere și inflație, într-un VAR pentru economia României.

Cursul EUR/RON este exemplul unui mers aleator (*random walk*) greu de depășit în prognoză. Indicii bursieri BET, S&P 500 și DAX, zilnici, ilustrează volatility clustering și spillover-ul între piețe. Consumul de electricitate, zilnic și orar, are sezonalitate multiplă și se potrivește metodelor de învățare automată. Seriile clasice ale manualelor, Nilul, petele solare, randamentele obligațiunilor și PIB-ul SUA, permit compararea rezultatelor cu literatura.

## @end

Harta și seriile arată aceeași idee din două direcții: metodele cursului formează un drum, de la descriere la mai multe serii modelate împreună, iar fiecare metodă a fost învățată pe o serie care o face necesară.

# Capitolele 0–3: descriere, staționaritate, ARMA, ARIMA

Primele patru capitole construiesc instrumentele pentru o singură serie și media ei: descrierea, staționaritatea, modelele ARMA și rădăcinile unitare.

## Capitolul 0: introducere: componente și netezire exponențială (1/2) {h=f5c8773a}

O serie se descompune într-un trend $T_t$, o componentă sezonieră $S_t$ și un rest $R_t$. În modelul aditiv, $y_t = T_t + S_t + R_t$, amplitudinea sezonieră este constantă în unități ale seriei; în modelul multiplicativ, $y_t = T_t S_t R_t$, ea crește odată cu nivelul. Logaritmul transformă produsul în sumă, așa că un model multiplicativ pentru $y_t$ este un model aditiv pentru $\ln y_t$. Din acest motiv, aproape toate seriile macroeconomice ale cursului sînt analizate în logaritm.

Netezirea exponențială simplă (*simple exponential smoothing*, SES) actualizează un nivel cu o medie ponderată între observația nouă și nivelul anterior:

$$\ell_t = \alpha y_t + (1 - \alpha)\ell_{t-1}, \qquad \alpha \in (0, 1).$$ {#eq:ses}

Un $\alpha$ apropiat de 1 urmărește îndeaproape ultima observație, iar un $\alpha$ mic netezește mult. Acuratețea se măsoară cu MASE (*mean absolute scaled error*): eroarea absolută medie (*mean absolute error*, MAE) a prognozei, împărțită la MAE a metodei naive sezoniere în eșantionul de estimare. O valoare sub 1 înseamnă că metoda bate reperul naiv; scalarea face comparabile serii cu unități diferite.

## Capitolul 0: introducere: componente și netezire exponențială (2/2) {h=7b6bc91e}

PIB-ul României a crescut de 2,3 ori din 1995, adică 2,8% pe an, iar componenta sezonieră este mare: trimestrul I este cu circa 23,8% sub trend, iar trimestrul IV cu 20,7% peste. Pentru cursul EUR/RON, parametrul estimat al netezirii exponențiale este $\hat\alpha = 1{,}00$, deci cea mai bună prognoză este ultima valoare observată, comportamentul unui mers aleator. Pentru consumul de electricitate, metoda Holt–Winters, care adaugă trend și sezonalitate, are MASE 0,51, față de 0,82 pentru metoda naivă.

Prima greșeală frecventă este judecarea unei prognoze pe datele folosite la estimare: în eșantion, un model mai complex se potrivește aproape întotdeauna mai bine, ceea ce nu spune nimic despre viitor. A doua este folosirea erorii procentuale absolute medii (*mean absolute percentage error*, MAPE) pentru serii care trec prin zero, cum este inflația lunară: împărțirea la valori apropiate de zero face eroarea arbitrar de mare. A treia este un model aditiv pentru o oscilație sezonieră care crește odată cu nivelul, situație în care logaritmul sau modelul multiplicativ este alegerea corectă.

## Capitolul 1: procese stochastice și staționaritate (1/2) {h=a27c066b}

Un proces este slab staționar (*weakly stationary*) dacă media lui este constantă, $E X_t = \mu$, și dacă autocovarianța $\mathrm{Cov}(X_t, X_{t+h}) = \gamma(h)$ depinde doar de distanța $h$, nu de momentul $t$. Staționaritatea face posibilă estimarea: dacă proprietățile seriei nu se schimbă în timp, observațiile din perioade diferite informează despre aceiași parametri. Autocorelația de selecție $\hat\rho(h)$ se compară cu banda $\pm 1{,}96/\sqrt{T}$, care este banda aproximativă de 95% pentru zgomot alb (*white noise*).

Testul Ljung–Box [@LjungBox1978] combină primele $m$ autocorelații într-o singură statistică:

$$Q^*(m) = T(T + 2)\sum_{h=1}^{m}\frac{\hat\rho(h)^2}{T - h} \sim \chi^2(m).$$ {#eq:lb}

O valoare mare, adică un p-value mic, respinge ipoteza de zgomot alb. Mersul aleator $X_t = X_{t-1} + \varepsilon_t$ este exemplul de bază al unei serii nestaționare: varianța lui, $\mathrm{Var}(X_t) = t\sigma^2$, crește liniar în timp, deci nu există o varianță constantă care să poată fi estimată.

## Capitolul 1: procese stochastice și staționaritate (2/2) {h=bc50bf8e}

Randamentele zilnice ale BET au autocorelația de ordinul 1 egală cu 0,11, iar randamentele absolute au 0,40, ambele față de o bandă de $\pm 0{,}024$. Diferența dintre cele două cifre este una dintre ideile centrale ale cursului: o serie poate fi aproape necorelată fără să fie independentă. Mărimea randamentelor este previzibilă chiar cînd semnul lor nu este, iar această observație pregătește modelele GARCH din [Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf).

Pentru PIB-ul României, transformarea Box–Cox are parametrul estimat $\hat\lambda = 0{,}02$, practic logaritmul. O diferențiere în plus aplicată unei serii deja staționare dă o autocorelație de ordinul 1 de $-0{,}54$: supradiferențierea introduce o autocorelație negativă artificială, semnul unei rădăcini unitare în partea MA. Cealaltă greșeală frecventă este interpretarea unei singure bare din afara benzii ca semnal: la 20 de laguri, una iese din bandă din întîmplare, chiar pentru zgomot alb.

## Capitolul 2: modele ARMA (1/2) {h=bb2203e1}

Modelul ARMA$(p, q)$ se scrie cu operatorul lag $L$, definit prin $LX_t = X_{t-1}$:

$$\phi(L)(X_t - \mu) = \theta(L)\varepsilon_t,$$ {#eq:arma}

unde $\phi(L)$ și $\theta(L)$ sînt polinoamele AR și MA, de ordinele $p$ și $q$, iar $\varepsilon_t$ este zgomot alb. Cele două cazuri simple dau tiparele de identificare: un AR(1) are autocorelațiile $\rho(h) = \phi^h$, care scad geometric, iar un MA(1) are o singură autocorelație nenulă, $\rho(1) = \theta/(1 + \theta^2)$, și se oprește brusc după lagul 1.

Între modelele candidate alegem cu criteriile informaționale. Criteriul Akaike [@Akaike1974], $\mathrm{AIC} = -2\ln\hat{\mathcal{L}} + 2k$, și criteriul Schwarz [@Schwarz1978], $\mathrm{BIC} = -2\ln\hat{\mathcal{L}} + k\ln T$, pornesc de la logaritmul verosimilității maxime și penalizează numărul $k$ de parametri; valoarea mai mică este mai bună. Penalizarea BIC este mai mare decît cea AIC de îndată ce $\ln T > 2$, adică pentru $T \ge 8$, deci BIC alege modele mai mici. După estimare, testul Ljung–Box pe reziduuri folosește $m - p - q$ grade de libertate, pentru că parametrii estimați absorb o parte din autocorelație.

## Capitolul 2: modele ARMA (2/2) {h=9fe2d487}

Pentru creșterea anuală a PIB-ului României, BIC alege un MA(3). Explicația este construcția seriei: creșterea față de același trimestru al anului anterior se calculează pe ferestre de patru trimestre care se suprapun, iar suprapunerea produce o dependență care se oprește după trei laguri. Reziduurile modelului au un p-value Ljung–Box de 0,37, deci testul nu separă autocorelația rămasă de zgomot. Inflația din România se potrivește cu un AR(2) cu $\hat\phi_1 + \hat\phi_2 = 0{,}976$, foarte aproape de o rădăcină unitară, iar petele solare au un ciclu AR(2) de 10,6 ani.

Prima greșeală frecventă este citirea ordinului AR din ACF: la un proces AR se oprește brusc PACF, nu ACF. A doua este testul Ljung–Box pe reziduuri cu $m$ grade de libertate în loc de $m - p - q$, care face testul prea îngăduitor. A treia este un factor comun în $\phi(L)$ și $\theta(L)$: dacă cele două polinoame au o rădăcină comună, ea se simplifică, iar modelul are parametri care nu pot fi identificați.

## Capitolul 3: rădăcini unitare și modele ARIMA (1/2) {h=85ad3256}

Testul Dickey–Fuller augmentat (*augmented Dickey–Fuller*, ADF) [@DickeyFuller1979] estimează regresia

$$\Delta y_t = c + bt + \gamma y_{t-1} + \sum_{j}\delta_j\Delta y_{t-j} + \varepsilon_t$$ {#eq:adf}

și testează $H_0$: $\gamma = 0$, adică o rădăcină unitară (*unit root*). Diferențele cu lag, cu coeficienții $\delta_j$, elimină autocorelația erorilor. Sub $H_0$ statistica $t$ a lui $\gamma$ nu are distribuția Student, ci o distribuție proprie, cu valori critice mai negative. Testul KPSS [@KwiatkowskiEtAl1992] inversează ipotezele: $H_0$ este staționaritatea. Folosite împreună, cele două teste dau patru combinații de decizii, dintre care două sînt clare și două neconcludente.

Modelul ARIMA$(p, d, q)$ aplică un ARMA seriei diferențiate de $d$ ori:

$$\phi(L)(1 - L)^d y_t = c + \theta(L)\varepsilon_t.$$ {#eq:arima}

Cînd $d \ge 1$, varianța erorii de prognoză crește cu orizontul $h$ fără limită, deoarece șocurile au efect permanent asupra nivelului. Aceasta este diferența practică dintre un trend stochastic și unul determinist: în jurul unui trend determinist intervalele de prognoză rămîn de lățime mărginită.

## Capitolul 3: rădăcini unitare și modele ARIMA (2/2) {h=810aa3d6}

Pentru logaritmul PIB-ului României, statistica ADF este $\tau = -2{,}31$, cu p-value 0,43, iar valoarea critică la 5% este $-3{,}45$; rădăcina unitară nu este respinsă. Pentru creștere, $\tau = -10{,}77$ respinge clar. Modelul rezultat este un ARIMA(0,1,0) cu derivă de 0,81% pe trimestru: un mers aleator cu creștere medie. Experimentul lui @GrangerNewbold1974 arată pericolul regresiilor între serii integrate: pentru două mersuri aleatoare independente, statistica $t$ a pantei este semnificativă în 90% din regresii.

Testul ADF are putere mică: pentru $\phi = 0{,}95$ și $T = 100$, el respinge rădăcina unitară doar în 12,5% din cazuri. De aceea nerespingerea nu dovedește existența unei rădăcini unitare, ci arată doar că datele nu o pot separa de o rădăcină apropiată de 1. Celelalte greșeli frecvente sînt valorile critice Student sau Normale pentru statistica ADF și o regresie în niveluri între serii $I(1)$ fără un test de cointegrare.

## @end

Primele patru capitole dau ordinea de lucru pentru o singură serie: transformarea care stabilizează varianța, testele care decid de cîte ori se diferențiază, corelograma care sugerează ordinele ARMA și criteriile informaționale care aleg între candidați, cu reziduurile verificate la sfîrșit.

# Capitolele 4–7: sezonalitate, volatilitate, VAR, VECM

Capitolele 4–7 adaugă sezonalitatea și evaluarea prognozelor, apoi trec de la medie la varianță și de la o serie la mai multe.

## Capitolul 4: sezonalitate și prognoză (1/2) {h=b6f41105}

Modelul SARIMA înmulțește un ARIMA obișnuit cu unul sezonier, scris în $L^s$:

$$\phi(L)\Phi(L^s)(1 - L)^d(1 - L^s)^D y_t = \theta(L)\Theta(L^s)\varepsilon_t,$$ {#eq:sarima}

unde $s$ este lungimea sezonului, 12 pentru date lunare, $\Phi$ și $\Theta$ sînt polinoamele sezoniere, iar $D$ este numărul de diferențe sezoniere $(1 - L^s)$. Modelul „airline”, $(0,1,1)(0,1,1)_s$, cu o diferență obișnuită, una sezonieră și cîte un termen MA în fiecare parte, este punctul de plecare clasic pentru serii lunare cu trend și sezonalitate.

Pentru a compara două prognoze se folosește testul Diebold–Mariano [@DieboldMariano1995]. Pentru fiecare perioadă se calculează diferența pierderilor $d_t = L(e_{1t}) - L(e_{2t})$, de exemplu diferența erorilor pătratice, iar statistica este media $\bar d$ împărțită la eroarea ei standard, $\bar d/\sqrt{\hat V/n}$, cu $\hat V$ varianța pe termen lung a lui $d_t$. Sub ipoteza acurateței egale statistica are aproximativ distribuția $N(0, 1)$, deci $|\mathrm{DM}| > 1{,}96$ respinge la 5%. Corecția lui @HarveyLeybourneNewbold1997 (HLN) ajustează statistica pentru eșantioane mici și o compară cu distribuția Student.

## Capitolul 4: sezonalitate și prognoză (2/2) {h=990d5bd3}

Pentru seria pasagerilor aerieni, modelul airline are o MAPE de 8,5%, față de 15,5% pentru metoda naivă sezonieră. Paștele ortodox, care nu cade în aceeași lună în fiecare an, crește vînzările de alimente din România cu circa 5,9%; un astfel de efect de calendar intră în model ca o variabilă explicativă, nu prin diferențiere sezonieră. Pentru consumul zilnic de electricitate, regresia armonică dinamică (*dynamic harmonic regression*, DHR), cu termeni Fourier pentru sezonalitatea anuală, are MASE 0,89, față de 1,28 pentru metoda naivă sezonieră.

Prima greșeală frecventă este compararea AICc pentru modele cu $d$ sau $D$ diferite: diferențierea schimbă datele pe care este calculată verosimilitatea, deci criteriile nu mai sînt comparabile. A doua este judecarea prognozelor la o singură origine, care poate depinde de un singur episod neobișnuit; originile mobile dau o imagine mai stabilă. A treia este folosirea datelor ajustate sezonier într-un model SARIMA care trebuie să prognozeze datele brute.

## Capitolul 5: volatilitate condiționată: ARCH și GARCH (1/2) {h=a201b2a9}

Modelul GARCH(1,1) [@Engle1982; @Bollerslev1986] scrie varianța condiționată ca funcție de șocul de ieri și de varianța de ieri:

$$\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2,$$ {#eq:garch}

cu $\omega > 0$, $\alpha \ge 0$, $\beta \ge 0$. Parametrul $\alpha$ măsoară reacția la șocuri, iar $\beta$ persistența. Dacă $\alpha + \beta < 1$, varianța revine spre nivelul pe termen lung $\bar\sigma^2 = \omega/(1 - \alpha - \beta)$, iar un șoc de volatilitate se înjumătățește în $h_{1/2} = \ln 0{,}5/\ln(\alpha + \beta)$ perioade. Cu cît $\alpha + \beta$ este mai aproape de 1, cu atît volatilitatea este mai persistentă.

Testul ARCH-LM verifică dacă varianța se schimbă: se regresează $\varepsilon_t^2$ pe $q$ laguri ale sale, iar $nR^2$ are, sub ipoteza de varianță constantă, distribuția $\chi^2(q)$. Modelul se folosește apoi pentru riscul de pierdere. Cu media condiționată $\mu$, volatilitatea prognozată $\sigma_{t+1}$ și cuantila de 1% $q_{0{,}01}(z)$ a inovațiilor standardizate, VaR 1% este

$$\mathrm{VaR}_{1\%} = -\bigl(\mu + \sigma_{t+1}\,q_{0{,}01}(z)\bigr),$$

o pierdere pozitivă, depășită în medie într-o zi din o sută.

## Capitolul 5: volatilitate condiționată: ARCH și GARCH (2/2) {h=ea75d3da}

Un GARCH(1,1) cu inovații Student are $\alpha + \beta = 0{,}995$ pentru S&P 500, cu un timp de înjumătățire de 131 de zile, și $0{,}991$ pentru BET, cu 74 de zile: volatilitatea este foarte persistentă pe ambele piețe. Modelul GJR, care adaugă un termen pentru șocurile negative, are pentru S&P 500 $\hat\gamma = 0{,}21$, cu statistica $t$ 10,0; scăderile de preț cresc volatilitatea mai mult decît creșterile de aceeași mărime, adică leverage effect. Kurtosis-ul este 13,7 pentru randamente și 5,0 pentru reziduurile standardizate: modelul explică o mare parte din cozile groase, dar nu pe toate.

Greșelile frecvente privesc formulele. Timpul de înjumătățire nu este $1/(1 - \alpha - \beta)$, ci $\ln 0{,}5/\ln(\alpha + \beta)$. Varianța pe termen lung nu există cînd $\alpha + \beta \ge 1$, deci nu trebuie raportată. În curs, nivelul VaR este probabilitatea cozii, iar VaR este o pierdere pozitivă: se scrie VaR 1%, niciodată „VaR 99%”, iar o valoare negativă indică o greșeală de semn.

## Capitolul 6: modele VAR și cauzalitate Granger (1/2) {h=23112972}

Un VAR$(p)$ [@Sims1980] modelează împreună $K$ variabile, fiecare regresată pe $p$ laguri ale tuturor:

$$\bY_t = \bc + \bA_1\bY_{t-1} + \dots + \bA_p\bY_{t-p} + \bepsilon_t.$$ {#eq:var}

Matricele $\bA_i$ au dimensiunea $K \times K$. Scris ca un VAR(1) de dimensiune $Kp$, prin matricea companion, modelul este stabil dacă toate valorile proprii ale acestei matrice sînt în interiorul cercului unitate; atunci efectul oricărui șoc se stinge în timp.

Cauzalitatea Granger [@Granger1969] întreabă dacă lagurile unei variabile $x$ îmbunătățesc prognoza lui $y$, dată fiind istoria lui $y$. Testul compară suma pătratelor reziduurilor fără lagurile lui $x$, $RSS_R$, cu cea cu lagurile lui $x$, $RSS_U$:

$$F = \frac{(RSS_R - RSS_U)/p}{RSS_U/(T - Kp - 1)}.$$

Funcțiile de răspuns la impuls (*impulse response functions*, IRF) și descompunerea varianței erorii de prognoză (*forecast error variance decomposition*, FEVD) cer șocuri necorelate, obținute de obicei prin descompunerea Cholesky, care depinde de ordinea variabilelor.

## Capitolul 6: modele VAR și cauzalitate Granger (2/2) {h=edafe586}

Un VAR(2) pentru România, cu creșterea economică, inflația și ROBOR 3M, arată că creșterea cauzează Granger ROBOR: statistica $F$ este 7,26, cu p-value 0,001. Lagurile creșterii conțin deci informație despre dobînda interbancară viitoare, dincolo de propriul trecut al dobînzii. Indicele de spillover dintre S&P 500, DAX și BET, construit din FEVD, este în medie 26,4% și ajunge la 57,5% pe 17 martie 2020: în crize, o parte mult mai mare din variația fiecărei piețe vine din șocurile celorlalte.

Prima greșeală frecventă este interpretarea cauzalității Granger ca o cauzalitate în sens economic; testul măsoară doar predictibilitatea suplimentară, iar o a treia variabilă omisă poate produce rezultatul. A doua este raportarea IRF fără ordinea Cholesky folosită, deși răspunsurile se pot schimba odată cu ordinea. A treia este folosirea prea multor laguri pe un eșantion scurt: numărul de parametri crește cu $K^2p$.

## Capitolul 7: cointegrare și VECM (1/2) {h=8641f589}

Două serii $I(1)$ sînt cointegrate dacă o combinație liniară a lor este staționară. Metoda Engle–Granger [@EngleGranger1987] estimează relația prin OLS în niveluri, apoi aplică un test ADF reziduurilor, cu valorile critice MacKinnon, mai negative decît cele Dickey–Fuller, pentru că reziduurile provin dintr-o regresie estimată. În modelul cu corecția erorii, coeficientul $\gamma < 0$ al dezechilibrului din perioada anterioară este viteza de ajustare, iar timpul de înjumătățire al unei abateri este $\ln 0{,}5/\ln(1 + \gamma)$.

Modelul vectorial cu corecția erorii (*vector error correction model*, VECM) scrie diferențele în funcție de dezechilibrele din perioada anterioară și de dinamica pe termen scurt:

$$\Delta\mathbf y_t = \alpha\beta^\top\mathbf y_{t-1} + \sum_i\Gamma_i\Delta\mathbf y_{t-i} + \mathbf u_t.$$ {#eq:vecm}

Coloanele lui $\beta$ sînt relațiile de echilibru, iar $\alpha$ spune cît de repede reacționează fiecare variabilă la dezechilibru. Testul urmei al lui @Johansen1991, cu statistica $-T\sum_{i>r}\ln(1 - \hat\lambda_i)$, testează succesiv rangurile $r = 0, 1, \dots$; primul rang care nu este respins este rangul de cointegrare.

## Capitolul 7: cointegrare și VECM (2/2) {h=3cb84eb1}

Pentru randamentele obligațiunilor SUA la 1, 5 și 10 ani, statistica urmei este 54,6 pentru $r = 0$, peste valoarea critică de 29,8, iar pentru $r \le 2$ este 3,12, sub valoarea critică de 3,84: rangul de cointegrare este 2. Trei dobînzi cu două relații de echilibru au o singură tendință stochastică comună, nivelul general al dobînzilor. Timpul de înjumătățire al corecției erorii pentru dobînda pe termen lung este de 20,8 luni.

Paritatea puterii de cumpărare (*purchasing power parity*, PPP) pentru leu nu este susținută: testul ADF pentru cursul real are p-value 0,24, deci rădăcina unitară nu este respinsă. Pentru pairs trading la Bursa de Valori București, raportul Sharpe al strategiei este 0,15 înainte de costuri și 0,05 după. Greșelile frecvente sînt valorile critice Dickey–Fuller pentru reziduurile Engle–Granger și un VAR în diferențe pentru serii cointegrate, care omite termenul de corecție a erorii și, odată cu el, informația pe termen lung.

## @end

Capitolele 4–7 extind modelul pentru o serie în trei direcții: sezonalitatea, cu SARIMA și evaluarea corectă a prognozelor; varianța, cu GARCH și VaR; și mai multe serii, cu VAR, cauzalitate Granger și, cînd seriile sînt integrate și legate printr-un echilibru, VECM.

# Capitolele 8–10: memorie lungă, învățare automată, spațiul stărilor

Capitolele 8–10 extind nucleul cursului cu memoria lungă, metodele de învățare automată și modelele cu stări ascunse.

## Capitolul 8: memorie lungă și ARFIMA (1/2) {h=bc7335f0}

Între $d = 0$, un proces staționar cu memorie scurtă, și $d = 1$, o rădăcină unitară, există valori fracționare ale lui $d$ [@GrangerJoyeux1980; @Hosking1981]. Operatorul $(1 - L)^d$ se dezvoltă într-o serie infinită de laguri, cu ponderile date de recursia

$$\pi_k = \pi_{k-1}\,\frac{k - 1 - d}{k}, \qquad \pi_0 = 1.$$ {#eq:frac}

Pentru $0 < d < 1/2$ procesul este staționar, dar autocorelațiile scad hiperbolic, $\rho(k) \sim Ck^{2d-1}$, nu exponențial ca la un ARMA: un șoc are efect pe sute de perioade.

Exponentul Hurst este $H = d + 1/2$, peste 0,5 pentru memorie lungă; în curs este estimat cu statistica rescaled range (R/S). Procesul este staționar pentru $d < 1/2$ și revine la medie pentru $d < 1$, chiar dacă varianța lui nu mai este finită. Estimatorii GPH și Whittle local folosesc doar cele mai joase $m$ frecvențe ale periodogramei, unde memoria lungă își lasă amprenta; alegerea lui $m$ schimbă estimarea.

## Capitolul 8: memorie lungă și ARFIMA (2/2) {h=2406fa59}

Pentru inflația lunară din România, un ARFIMA(0,$d$,0) cu $\hat d = 0{,}31$ și eroarea standard 0,05 bate modelele ARMA atît după AIC, cît și după BIC: un singur parametru fracționar descrie persistența mai bine decît mai mulți termeni AR și MA. Pentru S&P 500, exponentul Hurst estimat cu R/S este 0,53 pentru randamente, aproape de 0,5, și 0,86 pentru randamentele absolute: memoria lungă este în volatilitate, nu în randamente.

Nilul arată capcana principală a capitolului. Pe toată seria, $\hat d = 0{,}40$, dar după eliminarea rupturii din 1898, cînd nivelul seriei se schimbă, estimarea devine $-0{,}17$. O ruptură sau o schimbare de regim produce în corelogramă aceeași scădere lentă ca memoria lungă. Celelalte greșeli frecvente sînt raportarea unei singure lățimi de bandă $m$ și confundarea memoriei lungi a randamentelor cu cea a volatilității.

## Capitolul 9: învățare automată pentru serii de timp (1/2) {h=f40a7c79}

Metodele de învățare automată prognozează de obicei direct: pentru fiecare orizont $h$ se estimează o funcție separată a ultimelor $p$ valori și a regresorilor cunoscuți în avans, cum este calendarul,

$$\hat y_{T+h} = \hat f_h(y_T, \dots, y_{T-p+1}, \mathbf z_{T+h}).$$

Regresiile penalizate controlează complexitatea: ridge adaugă la suma pătratelor erorilor termenul $\lambda\sum\beta_j^2$ și micșorează toți coeficienții, iar lasso adaugă $\lambda\sum|\beta_j|$ și anulează o parte dintre ei, deci selectează predictorii.

Descompunerea deplasare–varianță (*bias–variance*) explică de ce un model mai flexibil nu este automat mai bun: eroarea pătratică medie de prognoză pentru o observație nouă este

$$E(y_0 - \hat f)^2 = \text{deplasare}^2 + \mathrm{Var}(\hat f) + \sigma^2.$$

Flexibilitatea reduce deplasarea și crește varianța, iar $\sigma^2$ nu poate fi eliminată de niciun model. Validarea trebuie să respecte ordinea timpului: validarea walk-forward estimează pe trecut și evaluează pe perioada imediat următoare, apoi avansează.

## Capitolul 9: învățare automată pentru serii de timp (2/2) {h=149544ce}

Pentru S&P 500, același model are $R^2 = 0{,}37$ cu validare încrucișată $k$-fold aleatoare și $-0{,}39$ cu validare walk-forward. Diferența este leakage: împărțirea aleatoare lasă modelul să învețe din observații ulterioare celor pe care le prognozează. Pentru semnul randamentelor, clasa majoritară este corectă în 54,4% din zile, iar un model de boosting doar în 52,5%. Pentru consumul zilnic de electricitate, ridge are MASE 0,90, iar o rețea LSTM (*long short-term memory*) 1,16, în acord cu concluzia competiției M4 [@MakridakisSpiliotisAssimakopoulos2020] că metodele complexe nu bat automat metodele simple.

Greșelile frecvente sînt validarea încrucișată aleatoare și scalarea variabilelor pe tot eșantionul, ambele forme de look-ahead. Arborii de decizie nu pot extrapola: aplicați pe niveluri cu trend, prognozele lor rămîn în intervalul valorilor de antrenare, deci trendul trebuie eliminat înainte. Orice comparație are nevoie de un reper simplu, metoda naivă sau naivă sezonieră.

## Capitolul 10: spațiul stărilor, filtrul Kalman și modele Markov switching (1/2) {h=8954944e}

Un model în spațiul stărilor are două ecuații: o ecuație de măsurare, care leagă observația de o stare ascunsă, și o ecuație de tranziție, care descrie evoluția stării:

$$y_t = Z\alpha_t + \varepsilon_t, \qquad \alpha_{t+1} = T\alpha_t + R\eta_t.$$ {#eq:ss}

Filtrul Kalman [@Kalman1960] actualizează estimarea stării cînd sosește o observație nouă: $a_{t|t} = a_t + K_tv_t$, unde $v_t = y_t - Za_t$ este eroarea de predicție, iar cîștigul $K_t = P_tZ'F_t^{-1}$ este ponderea acestei surprize. Cînd starea prezisă este incertă, $P_t$ mare, cîștigul este mare și filtrul se bazează mai mult pe observație.

În modelul Markov switching [@Hamilton1989], parametrii depind de un regim neobservat care trece dintr-o stare în alta cu probabilitățile $p_{ij}$. Durata medie a regimului $i$ este $1/(1 - p_{ii})$, iar cu două regimuri ponderea pe termen lung a regimului 1 este

$$\pi_1 = \frac{1 - p_{22}}{2 - p_{11} - p_{22}}.$$

## Capitolul 10: spațiul stărilor, filtrul Kalman și modele Markov switching (2/2) {h=8ed94d73}

Pentru Nil, modelul cu nivel local are raportul semnal–zgomot estimat $\hat q = \hat\sigma^2_\eta/\hat\sigma^2_\varepsilon = 0{,}097$: nivelul se schimbă puțin în raport cu zgomotul de măsurare. Cîștigul Kalman converge la $\bar K = 0{,}267$, care este exact ponderea $\alpha$ a netezirii exponențiale din [Capitolul 0](../RO/Cursuri/capitol0_introducere.pdf); netezirea exponențială este filtrul Kalman al modelului cu nivel local. Pentru PIB-ul SUA, regimurile estimate coincid cu datările recesiunilor NBER (*National Bureau of Economic Research*) în 90% din trimestre; pentru PIB-ul României, regimul stabil durează în medie 15,2 trimestre, iar cel volatil 11,2.

Prima greșeală frecventă este folosirea probabilităților netezite, care folosesc tot eșantionul, ca și cum ar fi fost cunoscute în timp real; pentru decizii în timp real contează probabilitățile filtrate. A doua este durata scrisă $p_{ii}/(1 - p_{ii})$ în loc de $1/(1 - p_{ii})$. A treia privește filtrul Hodrick–Prescott: ultima valoare a deviației de la trend este revizuită puternic cînd sosesc date noi, deci nu este o estimare în timp real.

## @end

Extensiile au aceeași structură ca nucleul: fiecare introduce un model nou pentru o întrebare pe care modelele anterioare nu o puteau trata și vine cu o capcană specifică, ruptura confundată cu memoria lungă, look-ahead în validare și informația din viitor în probabilitățile netezite.

# Capitolele 11–14: studiu individual

Ultimele patru capitole sînt pentru studiu individual. Recapitularea dă ideile lor principale și legătura cu nucleul cursului.

## Capitolul 11: foundation models pentru serii de timp (studiu individual) {h=c7b04d2f}

Foundation models sînt modele transformer pre-antrenate pe un număr foarte mare de serii. Valorile unei serii sînt transformate în tokeni, prin scalare și cuantizare, sau împărțite în segmente consecutive (*patching*), iar modelul învață să continue secvența. Chronos, TimesFM, Moirai și Lag-Llama pot prognoza zero-shot, adică pe o serie pe care nu au văzut-o la antrenare, fără nicio estimare suplimentară, sau pot fi adaptate prin fine-tuning.

Legătura cu nucleul este dublă. Un astfel de model este un model global, estimat pe multe serii deodată, ca modelele de învățare automată din [Capitolul 9](../RO/Cursuri/capitol9_invatare_automata_serii_timp.pdf). Prognozele lui se evaluează ca orice prognoză din [Capitolul 4](../RO/Cursuri/capitol4_sezonalitate_prognoza.pdf): același eșantion de test, MASE, scoruri pentru prognozele probabilistice și comparație cu repere simple.

## Capitolul 12: analiză spectrală (studiu individual) {h=6e06c1b4}

Analiza spectrală descrie o serie staționară prin frecvențele care o compun. Transformata Fourier descompune seria în sinusoide, periodograma $I(\nu_j)$ măsoară cît din varianță revine fiecărei frecvențe Fourier $\nu_j$, iar densitatea spectrală este versiunea teoretică a periodogramei. Periodograma brută fluctuează mult, așa că se netezește, de exemplu prin metoda Welch. Filtrele, cum este Hodrick–Prescott, păstrează anumite frecvențe și le elimină pe celelalte, ceea ce face din ciclul economic o bandă de frecvențe.

Spectrul conține aceeași informație ca funcția de autocovarianță din [Capitolul 1](../RO/Cursuri/capitol1_procese_stochastice_stationaritate.pdf), dar ordonată pe frecvențe în loc de laguri. Memoria lungă din [Capitolul 8](../RO/Cursuri/capitol8_memorie_lunga_arfima.pdf) apare ca un pol al densității spectrale la frecvența zero. Coerența măsoară legătura dintre două serii pe fiecare frecvență, iar wavelets permit studiul frecvențelor care se schimbă în timp.

## Capitolul 13: bule speculative: modele LPPL (studiu individual) {h=c256a492}

Capitolul 13 descrie bulele speculative ca o creștere superexponențială a prețurilor, mai rapidă decît orice creștere exponențială. Modelul LPPLS (*log-periodic power law singularity*) combină o lege de putere care duce la un timp critic cu oscilații log-periodice tot mai rapide. Estimarea are nevoie de condiții de filtrare care elimină ajustările fără sens economic, iar multe ferestre de estimare se combină într-un indicator de încredere, aplicat pe bule și crahuri istorice.

Legătura cu nucleul este testul Dickey–Fuller din [Capitolul 3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf), folosit invers: testele de rădăcini explozive caută o rădăcină autoregresivă mai mare decît 1, în coada din dreapta. A doua legătură este volatilitatea dinaintea crahurilor, studiată cu instrumentele din [Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf).

## Capitolul 14: modele GARCH multivariate (studiu individual) {h=499c67d7}

Capitolul 14 trece de la volatilitatea unei serii la matricea de covarianță a mai multor serii, care se schimbă în timp. Problema principală este numărul de parametri, care crește foarte repede cu numărul de serii. Modelele VEC și BEKK modelează direct covarianțele, cu multe restricții, iar modelele CCC (*constant conditional correlation*) și DCC (*dynamic conditional correlation*) separă volatilitățile individuale de corelații. DCC se estimează în doi pași: întîi un GARCH pentru fiecare serie, apoi dinamica corelațiilor pe reziduurile standardizate.

Rezultatele sînt corelații dinamice, varianța unui portofoliu și VaR 1% al portofoliului. Legătura cu nucleul este directă: fiecare serie are propriul GARCH din [Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf), iar perspectiva multivariată, cu matrice de coeficienți și numărul de parametri care crește rapid, este cea a modelului VAR din [Capitolul 6](../RO/Cursuri/capitol6_modele_var_cauzalitate_granger.pdf).

## @end

Capitolele pentru studiu individual folosesc aceleași idei ca nucleul în contexte noi: evaluarea în afara eșantionului pentru foundation models, autocovarianța citită pe frecvențe, testul Dickey–Fuller în coada din dreapta și GARCH extins la mai multe serii.

# Sinteza testelor și a modelelor

Sinteza leagă fiecare întrebare despre o serie de testul care răspunde la ea și de decizia care urmează.

## Teste și modele (1/2): serii univariate și media {h=0e3d2efa}

Tabelul are patru coloane: întrebarea, instrumentul, decizia și capitolul. Primele două rînduri privesc staționaritatea și tipul trendului. Graficul și ACF dau prima impresie, iar ADF și KPSS decid: cînd ADF respinge rădăcina unitară și KPSS nu respinge staționaritatea, seria este staționară. Cu trend inclus în regresie, același cuplu de teste decide între un trend determinist, care se elimină prin regresie, și unul stochastic, care se elimină prin diferențiere.

Sezonalitatea se vede în graficul sezonier și în ACF la lagurile $s$ și $2s$; testele OCSB și HEGY decid dacă este nevoie de o diferență sezonieră $\Delta_s$ sau dacă variabilele dummy și termenii Fourier sînt suficienți. Ordinul ARMA se identifică din ACF și PACF și se alege cu AIC și BIC: modelul cu cel mai mic BIC dintre cele cu reziduuri zgomot alb. Reziduurile se verifică cu Ljung–Box cu $m - k$ grade de libertate și cu Jarque–Bera pentru Normalitate. Ultimul rînd compară prognozele: un eșantion de test, origini mobile, MASE și testul Diebold–Mariano; dacă p-value-ul este peste 0,05, cele două prognoze nu se separă statistic.

## Teste și modele (2/2): varianța, serii multivariate și extensii {h=2f9e7cb6}

Partea a doua începe cu varianța. Testul Ljung–Box pe randamentele la pătrat $r_t^2$ și testul ARCH-LM verifică dacă varianța se schimbă; dacă testele resping, modelele potrivite sînt GARCH(1,1) cu inovații Student și GJR, pentru asimetrie. Pentru mai multe serii, întrebarea dacă $x$ ajută la prognoza lui $y$ are ca răspuns testul $F$ de cauzalitate Granger într-un VAR. Efectul unui șoc se măsoară cu IRF și FEVD, cu benzi bootstrap și cu mai multe ordini Cholesky, pentru a vedea cît de mult depind rezultatele de ordine. Un echilibru pe termen lung se testează cu Engle–Granger sau Johansen, iar VECM dă relațiile $\beta$, coeficienții de ajustare $\alpha$ și timpul de înjumătățire.

Ultimele rînduri privesc extensiile. Memoria lungă se estimează cu GPH, Whittle local și ARFIMA, cu mai multe valori ale lui $m$ și cu verificarea rupturilor. Pentru mulți predictori și efecte neliniare, ridge, lasso, arborii și boosting-ul se evaluează walk-forward, față de un reper. Pentru un nivel sau un regim ascuns, filtrul Kalman și Markov switching dau estimări filtrate pentru timp real și netezite pentru istorie.

## Convențiile folosite în tot cursul {h=9bfa5d0e}

Transformările sînt scrise mereu în aceeași formă. Logaritmul se înmulțește cu 100, astfel încît diferențele să fie aproximativ procente: $100\Delta\ln y_t$ este creșterea față de perioada anterioară, iar $100\Delta_{12}\ln y_t$ pentru date lunare și $100\Delta_4\ln y_t$ pentru date trimestriale sînt ratele anuale. În rezultatele EViews, „DLOG(IPC,1,12)” înseamnă diferența obișnuită a diferenței sezoniere a logaritmului, $\Delta\Delta_{12}\ln$ IPC.

Fiecare test se raportează la fel: ipoteza nulă, statistica, distribuția ei sub $H_0$, valoarea critică sau p-value-ul și decizia la 5%, urmate de o frază despre ce înseamnă decizia pentru serie. Criteriile informaționale AIC, AICc și BIC se compară doar pe același eșantion și pentru aceleași $d$ și $D$, iar prognozele se judecă în afara eșantionului, față de metodele naivă și naivă sezonieră. Pentru risc, nivelul este probabilitatea cozii: VaR 1% este o pierdere pozitivă, $\mathrm{VaR}_\alpha = -q_\alpha$, unde $q_\alpha$ este cuantila de nivel $\alpha$ a randamentului.

## @end

Sinteza transformă cursul într-o listă de întrebări cu răspunsuri verificabile. O analiză bună le parcurge în ordine, raportează fiecare test cu ipoteza lui nulă și decizia lui și respectă convențiile de transformare, de comparare a modelelor și de raportare a riscului.

# Box–Jenkins de la un capăt la altul: inflația din România

Secțiunea aplică metoda Box–Jenkins, în șase pași, pe IAPC al României. Fiecare pas folosește instrumente din capitolele anterioare, iar fiecare decizie se sprijină pe un rezultat raportat.

## Metoda în șase pași {h=a8b074f9}

Metoda lui @BoxJenkinsReinsel2008 este o buclă, nu o rețetă liniară. Primul pas este graficul și transformarea: logaritmul pentru a stabiliza varianța, diferența obișnuită $\Delta$ pentru trend și diferența sezonieră $\Delta_{12}$ pentru sezonalitate. Al doilea pas aplică testele ADF și KPSS fiecărei transformări. Al treilea identifică ordinele din ACF și PACF ale seriei staționare, iar al patrulea estimează mai multe modele SARIMA candidate prin verosimilitate maximă și le compară cu AICc și BIC.

Al cincilea pas verifică reziduurile: autocorelația cu Ljung–Box, Normalitatea cu Jarque–Bera și valorile extreme. Dacă verificarea eșuează, analiza se întoarce la pasul 3. Al șaselea pas prognozează și evaluează: un eșantion de test păstrat deoparte, comparația cu reperele naive și abia apoi intervalele de prognoză. Datele sînt IAPC al României, de la Eurostat, din ianuarie 2005 pînă în august 2026, $T = 260$ de luni.

## Pasul 1: datele {h=4ff3f00e}

Graficul are două panouri cu aceeași axă a timpului, din 2005 pînă în 2026. Panoul de sus arată nivelul IAPC, cu baza 2015 = 100: o linie care crește aproape continuu, cu o pantă mai mică între 2013 și 2017 și una mai mare după 2021. Panoul de jos arată, în procente, inflația lunară $100\Delta\ln P_t$ ca bare și inflația anuală $100\Delta_{12}\ln P_t$ ca linie roșie.

Cele două măsuri ale inflației se citesc diferit. Barele lunare sînt mici și zgomotoase, cu cîteva valori izolate mari, care corespund modificărilor de taxe. Linia anuală, care adună ultimele douăsprezece luni, este netedă și arată episoadele mari: inflația ridicată dinainte de 2012, perioada cu valori negative din 2015–2016, vîrful din 2022 și o nouă creștere în 2025.

## Interpretarea datelor {h=6811fade}

Indicele crește aproape în fiecare lună, deci logaritmul este scala firească: pe această scală, o creștere procentuală constantă devine o pantă constantă, iar diferențele sînt rate de creștere. Inflația lunară are media 0,39%, iar inflația anuală variază între $-3{,}0\%$, în mai 2016, și 13,6%, în noiembrie 2022. Cu un interval atît de larg, o singură medie nu descrie bine seria, iar persistența episoadelor sugerează o rădăcină unitară sau aproape unitară.

Sezonalitatea este vizibilă în mediile lunare ale inflației: ianuarie are în medie 0,74%, iar iunie $-0{,}03\%$, din cauza prețurilor alimentelor proaspete. Un tipar sezonier în $\Delta\ln P_t$ înseamnă că modelul va avea nevoie de diferența sezonieră $\Delta_{12}$ sau de termeni sezonieri. Șocurile de preț vin în mare parte din decizii fiscale, care mută prețurile într-o singură lună. În iulie–august 2025, încetarea plafonării prețului la electricitate și creșterea TVA au dus inflația anuală de la 6,4% la 8,2%.

## Pasul 2: teste de rădăcină unitară și de staționaritate {h=eb32d238}

Tabelul aplică ADF și KPSS pe patru transformări. Pentru $100\ln P_t$, cu constantă și trend, ADF are $\tau = -0{,}96$ și p-value 0,949, deci nu respinge rădăcina unitară, iar KPSS are $\eta = 0{,}314$, peste valoarea critică 0,146, deci respinge staționaritatea: cele două teste sînt de acord că nivelul are o rădăcină unitară. Pentru inflația lunară $\Delta\ln P_t$, ADF respinge, cu $\tau = -4{,}00$ și p-value 0,001, iar KPSS nu respinge, cu 0,431 sub 0,463: seria este staționară, dar sezonieră.

Inflația anuală $\Delta_{12}\ln P_t$ dă un rezultat neconcludent: ADF nu respinge, cu $\tau = -1{,}69$ și p-value 0,435, dar nici KPSS nu respinge, cu 0,424 sub 0,463. Situația este cea discutată în [Capitolul 3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf): datele nu au destulă informație pentru a decide. Pentru $z_t = \Delta\Delta_{12}\ln P_t$, ADF respinge clar, cu $\tau = -4{,}83$ și p-value sub 0,001, iar KPSS are 0,113, mult sub valoarea critică.

Decizia este $d = 1$ și $D = 1$: modelul se construiește pentru $z_t$, singura transformare pe care ambele teste o declară staționară și care elimină și tiparul sezonier. Valorile critice ADF la 5% sînt $-3{,}43$ cu trend și $-2{,}87$ cu constantă, iar numărul de laguri este ales cu AIC.

## Pasul 3: identificarea din corelograma lui zₜ {h=d6631c3f}

Graficul are două panouri: în stînga ACF, în dreapta PACF ale lui $z_t = \Delta\Delta_{12}\ln P_t$, pentru lagurile 1–36, cu $T = 247$ de observații. Axa orizontală este lagul în luni, iar lagurile sezoniere 12, 24 și 36 sînt colorate în roșu. Liniile punctate verzi sînt banda $\pm 1{,}96/\sqrt{T} = \pm 0{,}125$.

În ACF se văd două grupuri: bare pozitive la lagurile 1–2 și 6–7, cea mai mare la lagul 1, și o bară negativă mare la lagul 12, însoțită de bare negative mai mici la lagurile vecine. În PACF, după bara de la lagul 1, cele mai vizibile sînt barele roșii negative de la 12, 24 și 36, care scad în mărime de la primul lag sezonier la următoarele.

## Interpretarea corelogramei {h=da900854}

În partea obișnuită, $\hat\rho_1 = 0{,}33$ și $\hat\rho_2 = 0{,}16$, iar PACF are $\hat\phi_{11} = 0{,}33$ și apoi valori mici: tiparul unui AR(1), cu PACF care se oprește după lagul 1. Este posibil și un MA(1), deoarece atît ACF, cît și PACF scad repede după lagul 1, iar cu o singură autocorelație mare cele două tipare sînt greu de separat.

În partea sezonieră, ACF are $-0{,}45$ la lagul 12 și apoi $-0{,}01$ la lagul 24, deci se oprește după primul lag sezonier. PACF are $-0{,}50$, $-0{,}28$ și $-0{,}29$ la lagurile 12, 24 și 36, adică scade treptat. O singură valoare sezonieră mare în ACF, cu o PACF sezonieră care descrește, este tiparul unui MA(1) sezonier, termenul care apare natural după o diferență sezonieră.

Primul candidat este deci SARIMA$(1,1,0)(0,1,1)_{12}$ pentru $\ln P_t$. Barele negative de la lagurile 11 și 13, vecine lagului sezonier, nu cer termeni separați: într-un model multiplicativ, autocorelația de la lagurile $12 \pm 1$ este aproximativ produsul $\rho_1\rho_{12}$.

## Pasul 4: estimarea candidaților {h=edfab3c9}

Identificarea dă un punct de plecare, nu un rezultat final, așa că au fost estimate toate cele 36 de modele SARIMA$(p,1,q)(P,1,Q)_{12}$ cu $p, q \le 2$ și $P, Q \le 1$, pe eșantionul de antrenare pînă în august 2024, cu $T = 236$. Tabelul dă, pe primul rînd, modelul identificat, apoi, în ordinea BIC, celelalte patru modele cu cel mai mic BIC, cu numărul de parametri estimați, inclusiv varianța, AICc, BIC și p-value-urile testului Ljung–Box pe reziduuri la 12 și 24 de laguri, cu $m - k$ grade de libertate, unde $k$ este numărul de parametri ARMA.

Modelul identificat, $(1,1,0)(0,1,1)_{12}$, are AICc 295,2 și BIC 305,3, dar p-value-urile Ljung–Box sînt 0,004 la ambele laguri: reziduurile lui sînt autocorelate. Modelul cu cel mai mic BIC, $(1,1,1)(0,1,1)_{12}$, cu 298,9, are p-value 0,029 la lagul 12, deci eșuează și el. Modelul $(2,1,1)(0,1,1)_{12}$ are BIC 299,4, foarte aproape de minim, cel mai mic AICc, 282,6, și p-value-uri de 0,233 și 0,237.

Regula de alegere combină cele două criterii: cel mai mic BIC dintre modelele ale căror reziduuri trec testul. Modelul ales este $(2,1,1)(0,1,1)_{12}$. Diferența de BIC față de minim este mică, iar un model ale cărui reziduuri păstrează autocorelație nu poate fi folosit pentru prognoză, oricît de bine ar fi clasat.

## Pasul 4: modelul ales {h=06415872}

Modelul ales se scrie pentru $z_t = \Delta\Delta_{12}\,100\ln P_t$:

$$(1 - \phi_1L - \phi_2L^2)\,z_t = (1 + \theta_1L)(1 + \Theta_1L^{12})\,\varepsilon_t.$$ {#eq:chosen}

Estimările sînt $\hat\phi_1 = 1{,}156$ (eroarea standard 0,149, $z = 7{,}76$), $\hat\phi_2 = -0{,}183$ (eroarea standard 0,122, $z = -1{,}50$, p-value 0,133), $\hat\theta_1 = -0{,}856$ și $\hat\Theta_1 = -0{,}935$, ambele cu p-value sub 0,001, iar varianța inovațiilor este 0,179. Coeficientul $\phi_2$ nu se separă de zero la 5%, dar fără el reziduurile păstrează autocorelație, așa cum arată pasul anterior.

Estimarea $\hat\Theta_1$ este aproape de $-1$. Factorul $(1 + \Theta_1L^{12})$ aproape anulează diferența sezonieră $(1 - L^{12})$, ceea ce înseamnă că tiparul sezonier este aproape fix de la un an la altul; variabilele dummy sezoniere ar fi o alternativă. Cea mai mică rădăcină a polinomului AR are modulul 1,034, foarte aproape de cercul unitate: șocurile inflației lunare sînt foarte persistente, rezultat care se potrivește cu AR(2) din [Capitolul 2](../RO/Cursuri/capitol2_modele_arma.pdf) și cu memoria lungă din [Capitolul 8](../RO/Cursuri/capitol8_memorie_lunga_arfima.pdf).

## Pasul 5: verificarea reziduurilor {h=b2deb0b2}

Graficul are trei panouri. Primul arată reziduurile standardizate ale modelului ales în timp, din 2006 pînă în 2024, cu liniile punctate la $\pm 2$: cele mai multe valori stau în bandă, cu două excepții mari, un vîrf pozitiv în 2010 și unul negativ în 2015. Al doilea arată ACF a reziduurilor pentru lagurile 1–24, pentru ambele modele: barele aurii sînt ale modelului identificat, iar cele albastre ale modelului ales.

La modelul identificat, barele de la lagurile 6 și 7 și cea de la lagul 15 depășesc banda; la modelul ales ele scad, iar doar bara de la lagul 7 ajunge la marginea benzii. Al treilea panou este un QQ plot al reziduurilor față de distribuția Normală standard: punctele urmează linia roșie în centru, dar se depărtează de ea la ambele capete, cu cîteva valori extreme mult în afara liniei.

## Interpretarea verificării reziduurilor {h=d2572b18}

Testul Ljung–Box nu mai găsește autocorelație în reziduurile modelului ales: $Q(12) = 10{,}49$, cu 8 grade de libertate și p-value 0,233, iar $Q(24) = 24{,}13$, cu p-value 0,237. Modelul identificat avea p-value-uri de 0,004 la ambele laguri. Termenii AR și MA în plus reduc tocmai autocorelația de la lagurile 6–7, pe care modelul mai simplu o lăsa în reziduuri.

Reziduurile nu au însă distribuția Normală: statistica Jarque–Bera este 450, iar kurtosis-ul este 9,9, mult peste valoarea 3 a distribuției Normale. Cele mai mari două reziduuri au cauze cunoscute: iunie 2015, cu $-6{,}1\sigma$, cînd TVA la alimente a fost redusă, și iulie 2010, cu $5{,}3\sigma$, cînd TVA a fost crescută.

Consecința este împărțită. Prognozele punctuale sînt bune, deoarece reziduurile nu mai conțin dependență exploatabilă. Intervalele de prognoză calculate cu distribuția Normală sînt însă prea înguste în lunile în care se schimbă taxele. Variabilele dummy pentru modificările de taxe cunoscute ar elimina cele mai mari reziduuri și ar face intervalele mai credibile.

## Pasul 6: prognoze {h=53120f6a}

Graficul are două panouri. În stînga sînt prognozele pe un pas ale inflației lunare pe eșantionul de test, din septembrie 2024 pînă în august 2026, cu parametrii estimați doar pe eșantionul de antrenare: valorile observate (cercuri negre), prognozele SARIMA$(2,1,1)(0,1,1)_{12}$ (pătrate albastre) și prognozele naive sezoniere (triunghiuri aurii, linie întreruptă). Pentru cele mai multe luni, prognozele SARIMA sînt aproape de valorile observate; în iulie și august 2025 valorile observate cresc brusc, iar ambele prognoze rămîn mult sub ele.

Panoul arată și slăbiciunea metodei naive sezoniere: ea repetă valoarea din aceeași lună a anului anterior, deci în iulie și august 2026 copiază șocurile fiscale din 2025, în timp ce valorile observate revin la normal. În dreapta este inflația anuală din 2015 (linia roșie), cu prognozele pînă în august 2027 (linia albastră) și intervalul de 95% (zona gri), din modelul reestimat pe toate datele. Linia verde punctată marchează ținta de inflație a BNR, de 2,5%.

## Interpretarea prognozelor {h=ecebb15c}

Pe eșantionul de test de 24 de luni, modelul SARIMA are cele mai mici erori: rădăcina erorii pătratice medii (*root mean squared error*, RMSE) este 0,45, MAE este 0,28, iar MASE 0,66, față de 0,64, 0,42 și 0,99 pentru metoda naivă sezonieră și 0,51, 0,38 și 0,88 pentru metoda naivă. Testul Diebold–Mariano cu corecția HLN nu respinge însă acuratețea egală: statistica este $-1{,}40$, cu p-value 0,174, față de metoda naivă sezonieră, și $-0{,}47$, cu p-value 0,642, față de metoda naivă. Avantajul modelului nu se separă de zgomot pe acest eșantion.

Explicația ține de mărimea eșantionului și de natura erorilor. Douăzeci și patru de luni sînt puține pentru un test de acuratețe egală, iar cele mai mari erori vin din două luni: august 2025, cu inflația lunară observată de 2,07% față de prognoza de 0,47%, și iulie 2025, cu 1,34% față de 0,37%. Ele corespund unor decizii privind taxele și prețul energiei, pe care niciun model univariat nu le poate anticipa.

Inflația anuală a fost 6,1% în august 2026. Prognoza este 6,1% pentru septembrie 2026, cu intervalul de 95% între 5,3% și 6,9%, și 6,3% pentru august 2027, cu intervalul între 0,9% și 11,8%. Saltul de TVA din august 2025 iese din fereastra de douăsprezece luni în august 2026, ceea ce explică coborîrea inflației anuale observate la sfîrșitul eșantionului. Intervalul la douăsprezece luni este foarte larg, deoarece rădăcina AR aproape unitară face ca incertitudinea să se acumuleze aproape ca la un mers aleator.

## Recapitulare: Box–Jenkins pe inflația din România {h=ee2b54bd}

Graficul, transformarea și testele arată că $\ln P_t$ are nevoie de o diferență obișnuită și de una sezonieră, $d = 1$ și $D = 1$. Corelograma lui $z_t$ sugerează $(1,1,0)(0,1,1)_{12}$, dar verificarea reziduurilor respinge acest model, iar căutarea printre candidați duce la $(2,1,1)(0,1,1)_{12}$, cel mai mic BIC dintre modelele care trec testul Ljung–Box.

În afara eșantionului, modelul are erori mai mici decît reperele naive, dar cîștigul nu este semnificativ statistic pe 24 de luni, iar erorile sînt dominate de șocurile fiscale. Exemplul arată de ce bucla Box–Jenkins include întoarcerea la identificare și de ce evaluarea finală se face pe date pe care modelul nu le-a văzut.

## @end

Cazul IAPC folosește aproape toată prima parte a cursului: logaritmul și diferențele din [Capitolele 0](../RO/Cursuri/capitol0_introducere.pdf)–[1](../RO/Cursuri/capitol1_procese_stochastice_stationaritate.pdf), ADF și KPSS din [Capitolul 3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf), corelograma și criteriile informaționale din [Capitolul 2](../RO/Cursuri/capitol2_modele_arma.pdf), SARIMA și testul Diebold–Mariano din [Capitolul 4](../RO/Cursuri/capitol4_sezonalitate_prognoza.pdf). Concluzia este una prudentă: un model bine construit și verificat dă prognoze punctuale bune, dar nu poate anticipa șocurile de politică fiscală, iar intervalele lui trebuie citite știind că reziduurile au cozi groase.

# Examenul

Examenul scris are 70% din nota finală, iar [proiectul de echipă](#s73) are 20%; restul de 10% vine din prezență și din quiz-urile capitolelor. Această secțiune descrie examenul scris și prezintă opt probleme recapitulative, fiecare urmată de rezolvarea ei; secțiunea următoare descrie proiectul. Problemele urmează tipurile de subiecte ale examenului: o derivare scurtă, citirea unor rezultate obținute cu software și o prognoză calculată de mînă.

## Examenul scris: formatul {h=ee1fab15}

Examenul scris reprezintă 70% din nota finală și durează două ore. Are cinci subiecte, toate obligatorii, notate împreună cu 9 puncte, la care se adaugă un punct din oficiu.

Subiectele sînt de trei tipuri. Primul este o derivare scurtă: media și varianța unui proces, condiția de staționaritate, funcția de autocorelație sau forma procesului obținut după o transformare. Al doilea este interpretarea unor rezultate obținute cu software: estimări, teste, corelograme, tabele VAR și VECM. Al treilea cere alegerea unui model și o prognoză calculată de mînă din coeficienții estimați. Valorile critice necesare sînt tipărite în rezultate, ca în problemele din această secțiune.

## Conținutul evaluat {h=d09bed24}

Materia examenului este cea din cursurile și seminariile [Capitolelor 0](../RO/Cursuri/capitol0_introducere.pdf)–[10](../RO/Cursuri/capitol10_spatiul_starilor_kalman_markov_switching.pdf). Definițiile și formulele sînt cele de pe slide-urile „Noțiuni necesare azi” ale fiecărui seminar, iar calculele sînt de tipul celor din Partea A a seminariilor, făcute pe hîrtie.

O parte importantă a subiectelor este citirea rezultatelor: tabele ca cele din Partea B a seminariilor și din acest capitol. Rezultatele pot veni din EViews sau din Python, unde aceleași cifre poartă alte nume: coeficientul AR(1) apare în Python ca ar.L1, iar SAR(12) ca ar.S.L12. La examen nu se scrie cod.

Ponderea principală o au modelele ARIMA și SARIMA, rădăcinile unitare, modelele VAR și cauzalitatea Granger, cointegrarea și VECM și modelele GARCH. [Capitolele 11](../RO/Cursuri/capitol11_foundation_models_serii_timp.pdf)–[14](../RO/Cursuri/capitol14_modele_garch_multivariate.pdf) sînt pentru studiu individual.

## Criterii de notare {h=bb3ebe82}

Un răspuns se evaluează pe trei componente. Prima este metoda: formula sau ecuația scrisă, cu cifrele înlocuite. O metodă corectă cu o greșeală de calcul primește cea mai mare parte a punctajului, pentru că raționamentul se vede din formulă.

A doua componentă este rezultatul: cifra, cu unitatea de măsură (procente, puncte procentuale, luni) și cu semnul ei. Pentru un test, rezultatul complet conține ipoteza nulă $H_0$, statistica, valoarea critică sau p-value-ul și decizia.

A treia componentă este interpretarea, în trei pînă la cinci fraze, cu o parte statistică și una economică. O cifră corectă însoțită de o interpretare greșită nu primește tot punctajul. Interpretarea spune și ce nu poate arăta modelul: cauzalitatea în sens economic, rupturile structurale, deciziile viitoare de politică economică.

## Tipuri de probleme {h=af5bce62}

Tabelul leagă fiecare tip de subiect de capitolele din care provine și de problema care îl ilustrează. Derivările (momente, staționaritate, diferențiere) și identificarea din corelogramă și din testele de rădăcină unitară vin din [Capitolele 1](../RO/Cursuri/capitol1_procese_stochastice_stationaritate.pdf)–[3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf) (Problemele 1 și 2), iar rezultatele SARIMA cu o prognoză de mînă din [Capitolele 2](../RO/Cursuri/capitol2_modele_arma.pdf)–[4](../RO/Cursuri/capitol4_sezonalitate_prognoza.pdf) (Problema 3). Rezultatele VAR și testele Granger vin din [Capitolul 6](../RO/Cursuri/capitol6_modele_var_cauzalitate_granger.pdf) (Problema 4), cointegrarea și VECM din [Capitolul 7](../RO/Cursuri/capitol7_cointegrare_vecm.pdf) (Problema 5), GARCH și VaR 1% din [Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf) (Problema 6), iar evaluarea prognozelor, spațiul stărilor și regimurile din Capitolele 4 și 8–10 (Problemele 7 și 8).

Cele opt probleme sînt exemple de recapitulare, cu rezolvări complete; examenul are subiecte proprii. Exerciții suplimentare se găsesc în Partea A a fiecărui seminar, în Seminarul 15 și în [setul de probleme de exercițiu](https://danpele.github.io/Time-Series-Analysis/exam/practice/probleme_examen_ro.pdf), cu 25 de probleme și rezultatele lor numerice.

## Problema 1: un trend cu erori AR(1) {h=d7182b96}

Seria este un trend liniar, $Y_t = 10 + 0{,}5t + u_t$, peste care se suprapun erori AR(1), $u_t = 0{,}6u_{t-1} + \varepsilon_t$, cu $\varepsilon_t$ zgomot alb de varianță 1. La momentul $T = 100$ se observă $Y_T = 62$. Problema separă două tipuri de nestaționaritate care arată asemănător într-un grafic: un trend determinist cu abateri staționare și o rădăcină unitară.

Cerințele cer media și varianța lui $Y_t$, forma procesului diferențiat $\Delta Y_t$, prognoza pe unul și pe zece pași cu modelul corect și, în final, ce se întîmplă dacă seria este diferențiată. Toate cele patru puncte se rezolvă lucrînd cu componenta $u_t$, singura parte aleatoare a seriei.

## Problema 1: rezolvare {h=5ea0e4e0}

Media $E(Y_t) = 10 + 0{,}5t$ depinde de timp, deci $Y_t$ nu este staționar. Varianța este cea a procesului AR(1), $\mathrm{Var}(u_t) = 1/(1 - 0{,}6^2) = 1{,}5625$, constantă în timp. Seria este staționară în jurul trendului (*trend-stationary*): după scăderea trendului rămîne un proces staționar. Prin diferențiere, $\Delta Y_t = 0{,}5 + \Delta u_t$, iar

$$ (1 - 0{,}6L)\,\Delta u_t = (1 - L)\,\varepsilon_t, $$

adică un ARMA(1,1) cu $\theta = -1$. Autocovarianțele lui sînt $\gamma(0) = 1{,}25$ și $\gamma(1) = -0{,}25$, deci $\rho(1) = -0{,}20$. Rădăcina polinomului MA este 1, iar procesul diferențiat nu este invertibil. O autocorelație de ordinul 1 negativă și un coeficient MA estimat aproape de $-1$ sînt semnele supradiferențierii (*overdifferencing*).

Prognoza folosește modelul corect, trendul plus AR(1). Abaterea de la trend la momentul $T$ este $u_T = 62 - 10 - 50 = 2{,}0$ și se stinge geometric, cu factorul 0,6 pe perioadă:

$$ \begin{aligned} \hat Y_{T+1} &= 10 + 50{,}5 + 0{,}6 \cdot 2 = 61{,}70, \\ \hat Y_{T+10} &= 10 + 55 + 0{,}6^{10} \cdot 2 = 65{,}01. \end{aligned} $$

Varianța erorii pe zece pași este $(1 - 0{,}6^{20})/(1 - 0{,}36) = 1{,}562$, practic varianța lui $u_t$: după zece perioade mai rămîne doar 0,012 din abatere, iar incertitudinea rămîne mărginită. Tratamentul urmează tipul trendului: dintr-o serie staționară în jurul trendului se elimină trendul, iar o serie cu rădăcină unitară se diferențiază ([Capitolul 3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf)).

## Problema 2: identificarea PIB-ului României {h=c6888fa9}

Datele sînt PIB-ul real al României, ajustat sezonier, din 2000Q1 pînă în 2026Q2 (Eurostat). Tabelul dă testele ADF și KPSS pentru logaritmul PIB-ului, $100\ln Y_t$, cu constantă și trend, și pentru creșterea trimestrială $g_t = 100\Delta\ln Y_t$, cu constantă. Statistica ADF este $-2{,}31$, cu p-value 0,428, pentru nivel și $-10{,}77$, cu p-value sub 0,001, pentru creștere; statistica KPSS este 0,198 pentru nivel, cu valoarea critică de 5% egală cu 0,146, și 0,263 pentru creștere, cu valoarea critică 0,463.

Pentru $g_t$ se dau și primele două autocorelații, $-0{,}05$ și $-0{,}14$, egale cu autocorelațiile parțiale, banda de încredere $\pm 0{,}19$ și statistica Ljung–Box $Q(8) = 7{,}28$, cu p-value 0,506. Problema parcurge primii pași ai metodei Box–Jenkins: decizia de staționaritate cu două teste care au ipoteze nule opuse, identificarea modelului din corelogramă și interpretarea modelului estimat, a cărui derivă este 0,81, cu eroarea standard 0,30.

## Problema 2: rezolvare {h=58a0a59c}

Testul ADF [@DickeyFuller1979] estimează regresia

$$ \Delta y_t = c + bt + \gamma y_{t-1} + \sum_j \delta_j \Delta y_{t-j} + \varepsilon_t $$

și testează $H_0$: $\gamma = 0$, adică rădăcina unitară. Pentru nivel, $-2{,}31$ este peste valoarea critică de 5%, $-3{,}45$, deci rădăcina unitară nu se respinge; pentru creștere, $-10{,}77$ este mult sub valoarea critică de 1%, $-3{,}49$, și se respinge. Testul KPSS [@KwiatkowskiEtAl1992] are ipoteza nulă inversă, staționaritatea: 0,198 depășește 0,146 și respinge staționaritatea nivelului, iar 0,263 este sub 0,463 și nu o respinge pentru creștere. Cele două teste duc la aceeași concluzie: $\ln Y_t$ este $I(1)$.

Toate autocorelațiile și autocorelațiile parțiale ale creșterii sînt în bandă, iar $Q(8) = 7{,}28$ este sub valoarea critică 15,51: creșterea nu se separă de un zgomot alb în jurul mediei. Modelul identificat este ARIMA(0,1,0) cu derivă,

$$ 100\ln Y_t = 100\ln Y_{t-1} + 0{,}81 + \varepsilon_t, $$

cu abaterea standard a reziduurilor 2,47 și statistica $t$ a derivei 2,72.

Deriva este creșterea medie: circa 0,81% pe trimestru, aproximativ 3,2% pe an. Surprizele trimestriale nu sînt previzibile din trecutul seriei, iar un șoc asupra nivelului este permanent: recesiunile din 2009 și 2020 au mutat definitiv traiectoria PIB-ului, spre deosebire de seria staționară în jurul trendului din Problema 1, unde abaterile se sting.

## Problema 3: rezultate SARIMA și o prognoză de mînă {h=9832616d}

Seria este $z_t = \Delta\Delta_{12}\,100\ln \mathrm{IAPC}_t$, logaritmul IAPC al României diferențiat o dată obișnuit și o dată sezonier (în EViews, „DLOG(IPC,1,12)”), din februarie 2006 pînă în august 2026, cu $T = 247$ observații. Tabelul dă un model cu un termen AR(1) și unul SAR(12): $\hat\phi = 0{,}353$, cu eroarea standard 0,054 și $z = 6{,}53$, și $\hat\Phi = -0{,}488$, cu eroarea standard 0,035 și $z = -13{,}90$; varianța reziduurilor este 0,254, iar statistica Ljung–Box $Q(12)$ este 22,64, cu p-value 0,012.

Pentru prognoză se dau valorile necesare: $z = -1{,}814$ în august 2026, $0{,}090$ în septembrie 2025 și $1{,}794$ în august 2025; inflația anuală în august 2026 este 6,085%. Cerințele sînt ecuația modelului cu interpretarea coeficienților, testul de zgomot alb al reziduurilor și prognoza pentru septembrie 2026, întîi pentru $z$, apoi pentru inflația anuală.

## Problema 3: rezolvare {h=1c5e2486}

Modelul multiplicativ se scrie cu operatorul de întîrziere și apoi se desface produsul celor două polinoame:

$$ \begin{aligned} (1 - \phi L)(1 - \Phi L^{12})\,z_t &= \varepsilon_t, \\ z_t = \phi z_{t-1} + \Phi z_{t-12} &- \phi\Phi z_{t-13} + \varepsilon_t. \end{aligned} $$

Valoarea $\hat\phi = 0{,}353$ arată că o surpriză lunară continuă parțial în luna următoare, iar $\hat\Phi = -0{,}488 < 0$ arată că o surpriză dintr-o lună se inversează parțial în aceeași lună a anului următor. Ambii coeficienți sînt semnificativi, iar rădăcinile AR inversate, 0,35 și 0,94, sînt sub 1, deci modelul este staționar. Statistica Ljung–Box [@LjungBox1978] $Q(12) = 22{,}64$ are $12 - 2 = 10$ grade de libertate, numărul de laguri minus numărul de parametri ARMA estimați; cu p-value 0,012, ipoteza de zgomot alb se respinge la 5%. Modelul lasă o parte din autocorelație în reziduuri, la fel ca modelul sugerat de corelogramă la [Pasul 5](#s48).

Prognoza pentru septembrie 2026 înlocuiește valorile trecute cu cele observate și șocul viitor cu media lui, zero; produsul coeficienților este $\hat\phi\hat\Phi = -0{,}173$:

$$ \begin{aligned} \hat z_{T+1} &= 0{,}353(-1{,}814) + (-0{,}488)(0{,}090) \\ &\quad - (-0{,}173)(1{,}794) \\ &= -0{,}641 - 0{,}044 + 0{,}310 = -0{,}375. \end{aligned} $$

Deoarece $z_t$ este prima diferență a inflației anuale, $\Delta_{12}\ln P_{T+1} = \Delta_{12}\ln P_T + z_{T+1}$, iar prognoza inflației anuale este $6{,}085 - 0{,}375 = 5{,}71\%$. Inflația ar urma să scadă în continuare în septembrie 2026, pe măsură ce șocul fiscal din anul anterior iese din fereastra de douăsprezece luni. Autocorelația reziduală arată că un model mai bogat, ca la Pasul 4, ar schimba puțin prognoza, iar un interval de încredere de 95% are nevoie și de $\hat\sigma$.

## Problema 4: rezultate VAR și cauzalitate Granger {h=379adc51}

Modelul este un VAR(2) pentru România, din trimestrul 3 al anului 2005 pînă în trimestrul 2 al anului 2026 ($T = 84$), cu trei variabile: creșterea PIB-ului $g$, inflația anuală $\pi$ și rata dobînzii interbancare la trei luni, ROBOR 3M, notată $i$. Primul tabel dă ecuația ROBOR: coeficienții celor două laguri ale fiecărei variabile, cu statisticile $t$, constanta și $R^2 = 0{,}93$.

Al doilea tabel dă trei teste de cauzalitate Granger: ipoteza că $g$ nu cauzează Granger $i$ are $F = 7{,}26$ și p-value 0,001, ipoteza că $\pi$ nu cauzează Granger $i$ are $F = 4{,}74$ și p-value 0,011, iar ipoteza că $i$ nu cauzează Granger $\pi$ are $F = 0{,}89$ și p-value 0,417. Cerințele sînt definiția cauzalității Granger și citirea tabelului, ecuația ROBOR cu efectul pe termen lung al unei creșteri permanente a inflației cu 1 punct procentual și recalcularea primei statistici $F$ din sumele pătratelor reziduurilor $RSS_R = 69{,}268$ și $RSS_U = 58{,}273$.

## Problema 4: rezolvare {h=83ec5d0f}

Variabila $x$ cauzează Granger variabila $y$ dacă lagurile lui $x$ îmbunătățesc prognoza lui $y$, dată fiind istoria lui $y$ [@Granger1969]; testul este un test $F$ comun pentru toți coeficienții lagurilor lui $x$ din ecuația lui $y$. Creșterea și inflația ajută la prognoza ROBOR, cu p-value 0,001 și 0,011, iar ROBOR nu ajută la prognoza inflației, cu p-value 0,417. Prima statistică se reface din regresia restrînsă, fără cele două laguri ale lui $g$, și din cea nerestrînsă, cu $T - 7 = 77$ grade de libertate:

$$ F = \frac{(69{,}268 - 58{,}273)/2}{58{,}273/77} = 7{,}26, $$

peste valoarea critică $F_{0,95}(2, 77) = 3{,}12$, deci $H_0$ se respinge.

Ecuația ROBOR este

$$ \begin{aligned} i_t = {} & -0{,}109 + 0{,}142\,g_{t-1} + 0{,}074\,g_{t-2} \\ & + 0{,}240\,\pi_{t-1} - 0{,}172\,\pi_{t-2} \\ & + 1{,}042\,i_{t-1} - 0{,}119\,i_{t-2} + u_t. \end{aligned} $$

Pe termen lung, cu $i_t = i_{t-1} = i_{t-2}$ și $\pi_t = \pi_{t-1} = \pi_{t-2}$, efectul inflației asupra ROBOR este suma coeficienților lui $\pi$ împărțită la unu minus suma coeficienților lui $i$:

$$ \begin{aligned} \frac{\partial i}{\partial \pi} &= \frac{0{,}2402 - 0{,}1718}{1 - 1{,}0424 + 0{,}1186} \\ &= \frac{0{,}0684}{0{,}0762} = 0{,}90 \text{ pp}, \end{aligned} $$

cu coeficienții luați cu patru zecimale, deoarece cu trei zecimale numitorul ar fi $1 - 1{,}042 + 0{,}119 = 0{,}077$ în loc de 0,076.

Dobînda interbancară urmează economia și inflația, așa cum ar implica reacția unei bănci centrale; legătura inversă nu apare în datele trimestriale. Efectul pe termen lung împarte la 0,0762, un număr apropiat de zero, deci este imprecis. Cauzalitatea Granger înseamnă predictibilitate, nu dovada unui efect al politicii monetare.

## Problema 5: rezultate de cointegrare și VECM {h=d4380010}

Datele sînt randamentele titlurilor de stat ale SUA la 1, 5 și 10 ani, lunare (FRED), din ianuarie 1960 pînă în septembrie 2026, modelate cu un VECM cu două laguri ale diferențelor. Testul urmei Johansen are statisticile 54,60 pentru $r = 0$, 21,57 pentru $r \le 1$ și 3,12 pentru $r \le 2$, cu valorile critice de 5% egale cu 29,80, 15,49 și 3,84.

Ecuațiile de cointegrare estimate sînt $\mathrm{EC1} = y^{(1)}_{t-1} - 1{,}045\,y^{(10)}_{t-1} + 1{,}223$ și $\mathrm{EC2} = y^{(5)}_{t-1} - 1{,}044\,y^{(10)}_{t-1} + 0{,}587$. Al doilea tabel dă coeficienții de ajustare $\alpha$, cu statisticile $t$, pentru fiecare randament și fiecare relație. Cerințele sînt definiția cointegrării și rangul, partea de corecție a erorii din ecuația lui $\Delta y^{(10)}_t$ cu precizarea randamentelor care se ajustează și interpretarea economică a relațiilor pe termen lung.

## Problema 5: rezolvare {h=1e702383}

Seriile $I(1)$ sînt cointegrate dacă o combinație liniară a lor este $I(0)$ [@EngleGranger1987]. Testul urmei [@Johansen1991] se citește secvențial, de la $r = 0$ în sus: $54{,}60 > 29{,}80$ respinge $r = 0$, $21{,}57 > 15{,}49$ respinge $r \le 1$, iar $3{,}12 < 3{,}84$ nu respinge $r \le 2$. Rangul este $r = 2$: trei serii cu două relații de cointegrare au un singur trend stochastic comun.

Partea de corecție a erorii din ecuația randamentului la 10 ani este

$$ \begin{aligned} \Delta y^{(10)}_t = {} & 0{,}040\,\mathrm{EC1}_{t-1} \\ & - 0{,}077\,\mathrm{EC2}_{t-1} + \dots \end{aligned} $$

Un randament se ajustează la o relație dacă coeficientul lui $\alpha$ este semnificativ. Pe EC1, randamentele la 5 și 10 ani au $|t| > 2$, iar randamentul la 1 an are $|t| < 1$ pe ambele relații: dobînda scurtă nu se ajustează, adică este slab exogenă (*weakly exogenous*).

Coeficienții $\beta$ apropiați de 1 arată că spread-urile $y^{(1)} - y^{(10)}$ și $y^{(5)} - y^{(10)}$ sînt staționare, cum implică ipoteza așteptărilor; timpii de înjumătățire ai abaterilor de la echilibru sînt 22,1 și 9,0 luni. Cele trei randamente au un singur trend comun, nivelul dobînzilor: curba se poate deforma, dar spread-urile revin la mediile lor. Dobînzile lungi sînt cele care se ajustează, deci piețele mută partea lungă a curbei spre partea scurtă, determinată de politica monetară.

## Problema 6: rezultate GARCH și VaR 1% {h=81994cfe}

Modelul este un GARCH(1,1) [@Bollerslev1986] cu inovații Student-$t$, estimat pe randamentele logaritmice zilnice ale indicelui BET, în procente, din 2000, cu $n = 6\,670$ observații. Estimările sînt $\mu = 0{,}083$, $\omega = 0{,}0448$, $\alpha = 0{,}192$, $\beta = 0{,}799$ și $\nu = 5{,}011$ grade de libertate, cu erorile standard 0,010, 0,0088, 0,025, 0,024 și 0,308.

Azi, varianța condiționată este $\sigma_t^2 = 1{,}0$, iar șocul zilei este $\varepsilon_t = -3{,}0$; cuantila de 1% a distribuției $t_5$ este $-3{,}3649$, iar un an are circa 250 de zile de tranzacționare. Cerințele sînt persistența, timpul de înjumătățire și volatilitatea anuală pe termen lung, varianța de mîine $\sigma_{t+1}^2$ și VaR 1% pe o zi cu cuantila $t_5$ standardizată, comparat cu VaR-ul Normal.

## Problema 6: rezolvare {h=11cc23a4}

Persistența este $\alpha + \beta = 0{,}9907$, iar timpul de înjumătățire al unui șoc al volatilității este $h_{1/2} = \ln 0{,}5/\ln 0{,}9907 = 74$ de zile de tranzacționare. Varianța pe termen lung este $\bar\sigma^2 = \omega/(1 - \alpha - \beta) = 0{,}0448/0{,}0093 = 4{,}82$, iar volatilitatea anuală corespunzătoare este $\sqrt{250 \times 4{,}82} = 34{,}7\%$. Cînd $\alpha + \beta$ este aproape de 1, ambele cifre sînt foarte sensibile la a treia zecimală a coeficienților, de aceea persistența este dată cu patru zecimale: rotunjită la 0,991, ar da un timp de înjumătățire de 77 de zile și o varianță pe termen lung de 4,98. Volatilitatea pe termen lung, 34,7%, este mult peste cea de selecție, 22,1%, tocmai pentru că raportul $\omega/(1 - \alpha - \beta)$ este foarte imprecis în această situație.

Varianța de mîine combină constanta, șocul de azi la pătrat și varianța de azi:

$$ \begin{aligned} \sigma_{t+1}^2 &= 0{,}0448 + 0{,}19162 \times 9 + 0{,}7991 \times 1{,}0 \\ &= 0{,}0448 + 1{,}7246 + 0{,}7991 = 2{,}5685, \end{aligned} $$

deci $\sigma_{t+1} = 1{,}603\%$. Scăderea de 3% de azi a mai mult decît dublat varianța pentru ziua următoare.

Distribuția $t_\nu$ are varianța $\nu/(\nu - 2)$, deci cuantila ei se rescalează cu $\sqrt{(\nu - 2)/\nu} = \sqrt{3/5} = 0{,}7746$ pentru inovații cu varianța 1: $q_{0,01}(z) = -3{,}3649 \times 0{,}7746 = -2{,}606$. Cu $\mathrm{VaR}_\alpha = -q_\alpha$,

$$ \begin{aligned} \mathrm{VaR}\ 1\% &= -\bigl(0{,}083 + 1{,}603 \times (-2{,}606)\bigr) \\ &= 4{,}09\%, \end{aligned} $$

față de 3,65% cu distribuția Normală. Cozile groase ridică VaR 1% peste valoarea Normală: o pierdere mai mare de 4,09% este așteptată într-o zi din 100.

## Problema 7: evaluarea prognozelor {h=01aa8570}

Două metode, un SARIMA și metoda naivă sezonieră, prognozează inflația lunară pe un eșantion de test de șase luni. Erorile SARIMA sînt 0,3, −0,5, 0,2, 0,6, −0,4 și 0,1, iar erorile metodei naive sezoniere sînt 0,5, −0,7, 0,6, 0,4, −0,9 și 0,3; eroarea absolută medie (*mean absolute error*, MAE) a metodei naive sezoniere în eșantion este 0,6. A doua parte privește un coleg care raportează $R^2 = 0{,}37$ pentru un model boosting al randamentelor zilnice S&P 500, evaluat cu validare încrucișată aleatoare în 5 grupuri.

Cerințele sînt RMSE, MAE și MASE pentru ambele metode, statistica Diebold–Mariano pe erorile pătratice, $\bar d/(s_d/\sqrt{n})$, cu decizia la 5% (valoarea critică $t_{0,975}(5) = 2{,}571$), și un comentariu despre $R^2$-ul raportat de coleg.

## Problema 7: rezolvare {h=0aec3287}

RMSE este rădăcina mediei erorilor pătratice, MAE media erorilor absolute, iar MASE împarte MAE din eșantionul de test la MAE în eșantion al metodei naive sezoniere, 0,6. Pentru SARIMA, RMSE este 0,389, MAE 0,350 și MASE 0,58; pentru metoda naivă sezonieră, 0,600, 0,567 și 0,94. Un MASE sub 1 arată că metoda are, în medie, erori mai mici decît reperul naiv.

Testul Diebold–Mariano [@DieboldMariano1995] lucrează cu diferențele pierderilor, $d_t = e_{A,t}^2 - e_{B,t}^2$, cu $A$ SARIMA și $B$ metoda naivă sezonieră: $-0{,}16$, $-0{,}24$, $-0{,}32$, $0{,}20$, $-0{,}65$ și $-0{,}08$. Media lor este $\bar d = -0{,}208$, iar abaterea standard $s_d = 0{,}281$, deci

$$ \mathrm{DM} = \frac{\bar d}{s_d/\sqrt{n}} = \frac{-0{,}208}{0{,}281/\sqrt 6} = -1{,}82. $$

Cum $|-1{,}82| < 2{,}571$, acuratețea egală nu se respinge: SARIMA bate în medie metoda naivă sezonieră, dar șase luni nu pot separa acest cîștig de zgomot.

Validarea încrucișată cu grupuri aleatoare pune zile din viitor în setul de antrenare, o formă de leakage. Validarea walk-forward, care respectă ordinea timpului, dă $R^2 = -0{,}39$ pentru același model ([Capitolul 9](../RO/Cursuri/capitol9_invatare_automata_serii_timp.pdf)). Doar evaluarea în afara eșantionului, în ordinea timpului, este o dovadă a capacității de prognoză.

## Problema 8: spațiul stărilor, regimuri și memorie lungă {h=47b303db}

Problema are trei părți independente. Prima privește modelul local level, cu varianța zgomotului de observație $\sigma^2_\varepsilon = 4$ și varianța șocului nivelului $\sigma^2_\eta = 1$; predicția nivelului este $a_t = 10$, cu varianța $P_t = 2$, iar observația este $y_t = 13$. Se cer un pas de actualizare și de predicție Kalman, cîștigul de echilibru și ponderea netezirii exponențiale simple (SES) echivalente.

A doua parte privește creșterea PIB-ului României cu două regimuri, cu probabilitățile de a rămîne în regim $p_{11} = 0{,}9108$ pentru regimul volatil și $p_{22} = 0{,}9341$ pentru cel stabil ([Capitolul 10](../RO/Cursuri/capitol10_spatiul_starilor_kalman_markov_switching.pdf)); se cer duratele așteptate ale regimurilor și ponderea pe termen lung a regimului volatil. A treia parte privește inflația lunară din România, modelată ca ARFIMA(0,$d$,0) cu $\hat d = 0{,}31$ și eroarea standard 0,05 ([Capitolul 8](../RO/Cursuri/capitol8_memorie_lunga_arfima.pdf)); se cer exponentul Hurst $H$ și $\rho(1)$, cu precizarea dacă seria este staționară și revine la medie.

## Problema 8: rezolvare {h=c5c41f89}

Filtrul Kalman [@Kalman1960] compară observația cu predicția. Varianța erorii de predicție este $F_t = P_t + \sigma^2_\varepsilon = 2 + 4 = 6$, cîștigul este $K_t = P_t/F_t = 0{,}333$, iar surpriza este $13 - 10 = 3$. Actualizarea dă $a_{t|t} = 10 + 0{,}333 \times 3 = 11{,}00$ și $P_{t|t} = 2(1 - 0{,}333) = 1{,}333$; în modelul local level nivelul prognozat rămîne $a_{t+1} = 11{,}00$, iar varianța crește cu $\sigma^2_\eta$, la $P_{t+1} = 2{,}333$. Cu raportul semnal–zgomot $q = \sigma^2_\eta/\sigma^2_\varepsilon = 0{,}25$, varianța de echilibru și cîștigul de echilibru sînt

$$ \begin{aligned} \frac{\bar P}{\sigma^2_\varepsilon} &= \frac{q + \sqrt{q^2 + 4q}}{2} = 0{,}6404, \\ \bar K &= \frac{0{,}6404}{1{,}6404} = 0{,}390, \end{aligned} $$

iar $\bar K$ este exact ponderea $\alpha$ a netezirii exponențiale simple.

Într-un model cu schimbări de regim markoviene [@Hamilton1989], durata unui regim are distribuție geometrică, cu media $1/(1 - p_{ii})$: $E(D_1) = 1/(1 - 0{,}9108) = 11{,}2$ trimestre pentru regimul volatil și $E(D_2) = 15{,}2$ trimestre pentru cel stabil. Ponderea pe termen lung a regimului volatil este $\pi_1 = (1 - p_{22})/(2 - p_{11} - p_{22}) = 0{,}425$: creșterea din România se află în regimul volatil aproximativ 42% din timp, perioade care acoperă tranziția, 2009 și 2020.

Pentru ARFIMA [@GrangerJoyeux1980; @Hosking1981], $H = d + 0{,}5 = 0{,}81$ și $\rho(1) = d/(1 - d) = 0{,}45$. Cum $0 < d < 0{,}5$, inflația este staționară, cu memorie lungă, și revine la medie. Șocurile ei se sting lent, hiperbolic, iar prognozele revin la medie mai încet decît cele ale unui model AR.

## @end

Cele opt probleme acoperă tipurile de subiecte ale examenului scris: derivări pentru un proces dat, decizii de staționaritate și identificare, citirea rezultatelor SARIMA, VAR, VECM și GARCH, prognoze calculate de mînă și evaluarea prognozelor. Fiecare rezolvare are aceeași structură: formula cu cifrele înlocuite, rezultatul cu unitatea și semnul lui, decizia testului și o interpretare statistică și economică. Examenul scris dă 70% din nota finală; a doua componentă notată, [proiectul de echipă](#s73), cu 20%, aplică aceleași metode pe date reale, cu cod, și este descrisă în secțiunea următoare.

# Proiectul de echipă și prezența

Secțiunea descrie proiectul de echipă: ce conține, ce se predă, cum se evaluează și cum se declară folosirea instrumentelor AI.

## Proiectul de echipă: conținut și livrabile {h=ae501239}

Proiectul este o analiză făcută de o echipă de doi pînă la patru studenți, pe serii reale, cu metodele cursului. Punctul de plecare este o întrebare concretă, de exemplu prognoza cursului EUR/RON, modelarea inflației din România sau legătura dintre politica monetară și dobînzile bancare. Pentru o singură serie, analiza parcurge trendul și staționaritatea, netezirea, un model ARIMA sau SARIMA, împărțirea în set de antrenare și set de test, orizontul, prognozele punctuale și pe interval și comparația a două metode. Pentru mai multe serii, ea adaugă rădăcinile unitare, cointegrarea, un VAR sau un VECM, cauzalitatea Granger, IRF și FEVD.

Livrabilele sînt un repository al cărui cod reproduce, pornind de la date, fiecare rezultat numeric și fiecare grafic, un raport scurt, o prezentare și fișierul AI_USE.md. Raportul are structura unui articol scurt: întrebarea și literatura, sursele de date, modelele, concluziile și bibliografia.

## Criterii de evaluare a proiectului și susținerea orală {h=c3d77c34}

Primul criteriu este întrebarea: ea trebuie să fie clară, să poată primi un răspuns pe baza datelor și să fie utilă cuiva. „Poate un SARIMA să prognozeze inflația din România mai bine decît metoda naivă sezonieră?” este o întrebare, pentru că are un răspuns care poate ieși „da” sau „nu”, iar cazul Box–Jenkins din acest capitol arată cum se ajunge la el. „O analiză a inflației” nu este o întrebare, ci un subiect.

Celelalte criterii privesc metodele, verificările, interpretarea și reproductibilitatea: testul potrivit pentru întrebare, ales din sinteza testelor și a modelelor, diagnosticarea reziduurilor, evaluarea în afara eșantionului față de un reper, o interpretare care spune ce înseamnă cifrele și ce nu pot arăta datele și un repository care rulează de la date pînă la fiecare grafic. Proiectul se notează doar după susținerea orală, în care fiecare membru al echipei explică codul și rezultatele. O linie de cod pe care autorul ei nu o poate explica nu este considerată muncă proprie, indiferent dacă a fost scrisă cu sau fără ajutorul unui instrument AI.

## Fișierul AI_USE.md {h=426643c8}

Instrumentele AI sînt permise în proiect, cu condiția să fie declarate. Fișierul AI_USE.md înregistrează, pentru fiecare utilizare, instrumentul, prompt-ul, ce s-a păstrat din răspuns și ce s-a corectat. El conține și erorile instrumentului pe care echipa le-a găsit, împreună cu modul în care le-a găsit; această parte arată cel mai bine dacă rezultatele au fost verificate.

Responsabilitatea pentru fiecare cifră și fiecare referință rămîne a echipei. Fiecare cifră trebuie recalculată cu propriul cod, din date, iar fiecare referință trebuie verificată prin deschiderea identificatorului ei digital (*digital object identifier*, DOI) și compararea titlului. Erorile tipice ale instrumentelor AI în acest curs sînt adunate în slide-ul „Verificări necesare” din secțiunea următoare.

## Nota finală {h=842ce075}

Nota finală are trei componente. [Examenul scris](#s53), dat în sesiune, are ponderea 70% și se notează după metodă, rezultat și interpretare. [Proiectul de echipă](#s73), predat la finalul semestrului, are ponderea 20% și se notează după întrebare, metode, verificări și [susținerea orală](#s74). Prezența la cursuri și seminarii, urmărită în fiecare săptămînă, și quiz-urile capitolelor de pe site au împreună ponderea 10%.

Examenul și proiectul verifică lucruri diferite. Examenul cere calcule pe hîrtie și citirea unor rezultate date, fără cod; proiectul cere aplicarea metodelor pe date reale, cu un cod care reproduce fiecare rezultat și pe care fiecare membru al echipei îl poate explica. Seminariile au rol de exercițiu și nu se notează, iar studenții nu predau nimic de la seminarii. Quiz-urile se rezolvă cu contul Google ASE, iar numele și scorul se înregistrează pentru notă.

## @end

Proiectul cere același lucru ca analiza IAPC din acest capitol: o întrebare precisă, metodele potrivite, verificări raportate și un cod care reproduce fiecare rezultat, explicat de autorii lui.

# Contribuția posibilă a AI

Secțiunea descrie ce poate face un asistent AI pentru metodele cursului și ce trebuie verificat de fiecare dată.

## Contribuția posibilă a AI {h=7b2887ae}

Un asistent AI poate fi util în trei moduri. Pentru exercițiu, poate formula probleme noi, cu rezultate generate din date reale de propriul cod, nu inventate. Pentru explicații, poate da o a doua explicație, în alte cuvinte, pentru un tabel de rezultate, o derivare sau un test care nu a fost înțeles la prima lectură. Pentru proiect, poate scrie o primă versiune a codului pentru pașii Box–Jenkins, pentru o evaluare cu origini mobile sau pentru un VECM.

Prompt-ul de pe slide este un exemplu de cerere precisă. El numește sursa datelor și codul exact al seriei Eurostat, transformarea ($100\ln$ IAPC), familia de modele și limitele ordinelor, eșantionul de antrenare, criteriile de comparare, gradele de libertate ale testului Ljung–Box și reperul pentru prognoză. Un prompt atît de precis produce cod care poate fi verificat pas cu pas, comparînd fiecare rezultat cu tabelele din secțiunea Box–Jenkins.

## Verificări necesare {h=62b9b345}

Primele verificări privesc ipotezele și gradele de libertate. La ADF, ipoteza nulă este rădăcina unitară, iar la KPSS staționaritatea; un răspuns care le inversează duce la decizii opuse. Testul Ljung–Box pe reziduuri folosește $m - k$ grade de libertate, iar testul Granger $F$ folosește $p$ și $T - Kp - 1$. Semnele termenilor MA și ale coeficientului de corecție a erorii depind de convenția programului: EViews și Python le scriu diferit, deci un semn trebuie citit împreună cu ecuația pe care o folosește programul.

Celelalte verificări privesc convențiile și ordinea timpului. AIC și BIC se compară doar pentru aceleași $d$ și $D$, iar riscul se raportează ca VaR 1%, niciodată „VaR 99%”. Nicio informație din eșantionul de test nu are voie să intre în estimare, în scalarea variabilelor sau în alegerea modelului. Fiecare cifră trebuie recalculată cu propriul cod, iar fiecare referință verificată prin DOI.

## @end

Un asistent scurtează scrierea codului și oferă explicații alternative, dar verificarea ipotezelor nule, a gradelor de libertate, a semnelor și a ordinii timpului rămîne responsabilitatea analistului.

# Rezumat

Rezumatul adună ideile cursului într-o ordine de lucru și propune trei întrebări de autoevaluare.

## Idei de reținut {h=7103fb86}

Analiza unei serii începe cu datele: graficul, componentele, transformarea potrivită și corelograma. Apoi $d$ și $D$ se decid cu ADF și KPSS împreună, ordinele se identifică din ACF și PACF, modelul se alege cu BIC, iar reziduurile se verifică înainte ca modelul să fie folosit. Cînd varianța se grupează în timp, ea se modelează cu GARCH; cînd mai multe serii se influențează, ele se modelează împreună, cu VAR sau, dacă sînt cointegrate, cu VECM.

Prognozele se judecă în afara eșantionului, față de repere simple. O potrivire mai bună în eșantion nu înseamnă o prognoză mai bună, iar un cîștig față de metoda naivă trebuie testat, nu doar observat. În proiect contează o întrebare clară și un răspuns onest, inclusiv atunci cînd răspunsul este că datele nu pot decide.

## Autoevaluare {h=04480bee}

Prima întrebare combină cele două teste de rădăcină unitară. Dacă ADF nu respinge rădăcina unitară, iar KPSS respinge staționaritatea, ambele teste indică o rădăcină unitară, deci seria se diferențiază. Este exact situația nivelului IAPC din pasul 2.

A doua întrebare aplică testul Ljung–Box pe reziduurile unui ARMA(1,1), cu $Q(10) = 16$. Gradele de libertate sînt $10 - 2 = 8$, iar valoarea critică a distribuției $\chi^2(8)$ la 5% este 15,51. Statistica o depășește, deci rămîne autocorelație în reziduuri și modelul nu este adecvat.

A treia întrebare privește două serii $I(1)$ cointegrate. Un VAR în diferențe nu este potrivit, deoarece omite termenul de corecție a erorii din @eq:vecm și pierde informația despre relația pe termen lung; modelul corect este VECM.

## @end

Cursul se reduce la o succesiune de întrebări cu instrumente precise: cum arată seria, de cîte ori trebuie diferențiată, ce dependență are, cum se schimbă varianța ei, cum se leagă de alte serii și cît de bine poate fi prognozată, totul verificat pe date pe care modelul nu le-a văzut.
