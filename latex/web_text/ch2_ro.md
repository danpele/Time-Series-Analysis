<!--
Textul de curs al versiunii web, Capitolul 2 (RO). Inserat în pagină de latex/build_web_lecture.py.
Format: # titlul secțiunii (introducere), ## titlul slide-ului {h=hash} (text), @after (text după blocul live), ## @end (sinteză).
Ecuații numerotate: $$ ... $$ {#eq:eticheta}, referință @eq:eticheta. Citări: @Cheie -> Autor (an), [@Cheie] -> (Autor, an).
Hash-urile curente: python3 build_web_lecture.py 2 --hashes
-->

## @intro

@epigraph Yule1927 | On a method of investigating periodicities in disturbed series, with special reference to Wolfer's sunspot numbers
> If we take a curve representing a simple harmonic function of the time, and superpose on the ordinates small random errors, the only effect is to make the graph somewhat irregular, leaving the suggestion of periodicity still quite clear to the eye.

@translation
> Dacă luăm o curbă care reprezintă o funcție armonică simplă de timp și suprapunem peste ordonate mici erori aleatoare, singurul efect este că graficul devine oarecum neregulat, iar sugestia de periodicitate rămîne în continuare foarte clară pentru ochi.

@text
@Yule1927 a pus acestei imagini în contrast un mecanism diferit: un pendul a cărui mișcare este perturbată de șocuri aleatoare. În prima imagine ciclul este o funcție fixă de timp, iar erorile doar îl estompează. În a doua, șocurile intră în dinamică, fiecare șoc schimbă traiectoria viitoare, iar seria este corelată cu propriul trecut. Yule a aplicat acest al doilea mecanism numărului anual de pete solare, cu doi coeficienți, și a obținut prima autoregresie. Capitolul pornește de la aceeași întrebare: o serie staționară este corelată cu propriul trecut și avem nevoie de un model mic, care să surprindă această memorie și să o transforme în prognoze cu o incertitudine onestă.

Descompunerea Wold din Capitolul 1 arată că orice proces staționar pur nedeterminist este o sumă ponderată infinită de șocuri trecute. Ea nu spune însă cum se estimează o infinitate de ponderi din $T$ observații. Modelele ARMA($p,q$) ale lui @BoxJenkinsReinsel2008 scriu aceste ponderi ca raport a două polinoame scurte, cu $p + q + 2$ parametri. Capitolul derivă proprietățile lor (staționaritate, invertibilitate, autocorelații, răspunsuri la impuls), estimatorii, criteriile informaționale, testele pentru reziduuri și prognozele, apoi aplică întreaga metodă pe serii reale.

Aplicația principală este creșterea anuală a PIB-ului real al României, 106 trimestre, din T1 2000 pînă în T2 2026. Funcția ei de autocorelație se anulează după lagul 3, criteriul informațional bayesian (BIC, *Bayesian information criterion*) alege un model MA(3), iar MA(3) lasă reziduuri fără autocorelație, dar cu cozi groase din cauza anului 2020. Aceleași instrumente se aplică apoi inflației din România, randamentelor zilnice ale indicelui BET și ale cursului EUR/RON și petelor solare ale lui Yule.

@chart tsa_ch2_gdp_ident
Creșterea anuală a PIB-ului real al României, T1 2000–T2 2026, în % (sus), cu funcția de autocorelație (ACF) și funcția de autocorelație parțială (PACF) de selecție pentru laguri de la 1 la 16 trimestre (jos); liniile roșii punctate sînt banda $\pm 1{,}96/\sqrt{T} = \pm 0{,}19$.

@goals
- **Verificați** staționaritatea unui AR($p$) și invertibilitatea unui MA($q$) din rădăcinile polinoamelor lor.
- **Derivați** media, autocovarianțele și ponderile $\psi$ pentru AR(1), AR(2), MA($q$) și ARMA(1,1).
- **Identificați** ordine candidate din ACF și PACF și alegeți între ele cu AIC și BIC.
- **Estimați** modele ARMA prin Yule–Walker, cele mai mici pătrate condiționate și verosimilitate maximă și interpretați rezultatele.
- **Testați** reziduurile cu Ljung–Box cu $m - p - q$ grade de libertate și cu Jarque–Bera.
- **Prognozați** cu modele ARMA, calculați intervalele și explicați revenirea la medie.

@plan
- De la Capitolul 1 la modelele ARMA | Staționaritatea, zgomotul alb, operatorul lag, descompunerea Wold și problema infinității de ponderi.
- Modele autoregresive | AR(1) și AR($p$): staționaritatea din rădăcini, momentele, triunghiul AR(2) și anularea PACF.
- Modele de medie mobilă | MA($q$): anularea ACF, limita lui $\rho(1)$ și invertibilitatea.
- Modele ARMA(p,q) | Ponderile $\psi$, răspunsurile la impuls, factorii comuni și tabelul de identificare ACF–PACF.
- Estimarea | Yule–Walker, cele mai mici pătrate condiționate și verosimilitatea maximă gaussiană, cu comportamentul lor în eșantioane mici.
- Selecția modelului și diagnosticarea | AIC și BIC, testele pentru reziduuri și gradele de libertate ale testului Ljung–Box.
- Prognoza cu modele ARMA | Prognoza optimă, varianța erorii, intervalele și revenirea la medie.
- Metoda Box–Jenkins | Bucla identificare, estimare, verificare și prognoză.
- Studiu de caz: creșterea PIB-ului României | Întreaga metodă pe 106 trimestre de creștere anuală a PIB-ului.
- Alte serii reale | Inflația, randamentele zilnice ale BET și EUR/RON, petele solare ale lui Yule și competițiile de prognoză.
- Contribuția posibilă a AI | Ce poate schița AI și verificările pe care analistul trebuie să le facă.
- Rezumat | Idei de reținut, formule de reținut și autoevaluare.

# De la Capitolul 1 la modelele ARMA

Secțiunea adună instrumentele din Capitolul 1 pe care le folosește capitolul și formulează problema pe care o rezolvă modelele ARMA: descompunerea Wold descrie orice proces staționar, dar cu o infinitate de ponderi.

## Instrumente din Capitolul 1 {h=b894cdb1}

Un proces $X_t$ este slab staționar (*weakly stationary*) cînd media lui $\mu$ este constantă, iar autocovarianța $\gamma(h) = \mathrm{Cov}(X_t, X_{t+h})$ depinde doar de lagul $h$, nu și de momentul $t$. Funcția de autocorelație (ACF, *autocorrelation function*), $\rho(h) = \gamma(h)/\gamma(0)$, aduce autocovarianțele pe intervalul $[-1, 1]$. Funcția de autocorelație parțială (PACF, *partial autocorrelation function*), $\phi_{hh}$, măsoară corelația dintre $X_t$ și $X_{t+h}$ după ce s-a eliminat efectul liniar al valorilor intermediare $X_{t+1}, \dots, X_{t+h-1}$. Aceste două funcții sînt instrumentele de identificare ale capitolului: forma lor indică modelul care trebuie estimat.

Zgomotul alb (*white noise*), $\varepsilon_t \sim \mathrm{WN}(0, \sigma^2)$, are media 0, varianța $\sigma^2$ și nicio autocorelație la laguri nenule. El este materia primă a tuturor modelelor din capitol: fiecare model este un filtru liniar aplicat unui zgomot alb. Operatorul lag (*lag operator*) mută o serie înapoi în timp, $LX_t = X_{t-1}$ și $L^kX_t = X_{t-k}$. Un polinom în $L$, de exemplu $1 - 0{,}5L$, acționează deci asupra unei serii ca o sumă ponderată a lagurilor ei, iar algebra polinoamelor (produse, rapoarte, rădăcini) devine algebra modelelor.

Descompunerea Wold [@Wold1938] leagă aceste piese. Orice proces staționar pur nedeterminist (*purely nondeterministic*) se poate scrie ca $X_t = \mu + \sum_{j \ge 0}\psi_j\varepsilon_{t-j}$, cu $\psi_0 = 1$ și $\sum_j\psi_j^2 < \infty$. Condiția „pur nedeterminist” exclude componentele pe care trecutul infinit le prezice exact, cum ar fi o sinusoidă fixă. Teorema spune că un astfel de proces este o medie mobilă de ordin infinit a șocurilor sale trecute; restul capitolului scrie această sumă cu puțini parametri.

## Reprezentarea parcimonioasă a ponderilor Wold {h=625666bc}

Descompunerea Wold spune cum arată un proces staționar, nu cum îl estimăm. Ponderile $\psi_1, \psi_2, \dots$ sînt o infinitate de necunoscute, iar eșantionul are doar $T$ observații. Dacă trunchiem suma la un lag mare, rămîn tot mulți parametri, fiecare estimat cu o eroare mare.

Ideea este să scriem polinomul în lag $\psi(L) = \sum_j\psi_jL^j$ ca raport a două polinoame scurte, $\psi(L) = \theta(L)/\phi(L)$. Înmulțind ambii membri ai formei Wold cu $\phi(L)$, obținem modelul ARMA($p,q$)

$$\begin{gathered}\phi(L)(X_t - \mu) = \theta(L)\varepsilon_t, \\ \phi(L) = 1 - \phi_1L - \dots - \phi_pL^p, \\ \theta(L) = 1 + \theta_1L + \dots + \theta_qL^q.\end{gathered} $$ {#eq:arma}

Modelul are $p + q + 2$ parametri: cei $p$ coeficienți autoregresivi, cei $q$ coeficienți de medie mobilă, media $\mu$ și varianța zgomotului $\sigma^2$. Un singur parametru poate genera deja o infinitate de ponderi: $1/(1 - 0{,}8L) = 1 + 0{,}8L + 0{,}64L^2 + \dots$.

Un raport a două polinoame de grad mic aproximează bine aproape orice $\psi(L)$ netedă. El reproduce descreșteri geometrice, oscilații amortizate și ponderi care se opresc după un lag finit, adică formele întîlnite în practică. Principiul care ghidează alegerea lui $p$ și $q$ este parcimonia (*parsimony*): cel mai mic model ale cărui reziduuri sînt zgomot alb. Fiecare parametru suplimentar adaugă eroare de estimare, iar după ce reziduurile au devenit zgomot alb el nu mai îmbunătățește prognozele.

## Modelele AR, MA și ARMA {h=19cf2471}

Modelul autoregresiv AR($p$) face ca valoarea de azi să depindă de ultimele $p$ valori ale seriei și de un șoc nou: $X_t = c + \phi_1X_{t-1} + \dots + \phi_pX_{t-p} + \varepsilon_t$. El provine de la @Yule1927 și din analiza petelor solare. Modelul de medie mobilă (*moving average*) MA($q$) face ca valoarea de azi să depindă de ultimele $q$ șocuri: $X_t = \mu + \varepsilon_t + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q}$. @Slutzky1937 a arătat că astfel de sume de cauze aleatoare produc serii care par ciclice. Modelul ARMA($p,q$) conține ambele părți; @BoxJenkinsReinsel2008, a căror primă ediție a apărut în 1970, l-au transformat într-o metodă completă de identificare, estimare, verificare și prognoză.

În forma autoregresivă, $c$ este termenul liber (*intercept*), nu media. Aplicînd media ambilor membri și folosind staționaritatea ($E[X_t] = E[X_{t-k}] = \mu$), obținem $\mu = c + (\phi_1 + \dots + \phi_p)\mu$, deci $\mu = c/(1 - \phi_1 - \dots - \phi_p)$. Distincția contează în practică: pentru $X_t = 2 + 0{,}8X_{t-1} + \varepsilon_t$ termenul liber este 2, iar media este 10, iar programele pot raporta oricare dintre ele.

Pe tot parcursul capitolului, $\varepsilon_t$ este zgomot alb; acest lucru ajunge pentru medii, autocovarianțe și prognozele punctuale. Verosimilitatea și intervalele de prognoză cer mai mult: $\varepsilon_t$ independente și identic distribuite (i.i.d.) $N(0, \sigma^2)$. Diferența dintre cele două ipoteze apare în studiul de caz, unde reziduurile modelului pentru PIB sînt necorelate, dar nu urmează distribuția Normală: estimațiile punctuale rămîn valabile, intervalele construite cu distribuția Normală nu.

## @end

Descompunerea Wold garantează că orice proces staționar pur nedeterminist este un filtru aplicat unui zgomot alb, iar modelul ARMA @eq:arma scrie acest filtru ca $\theta(L)/\phi(L)$, cu $p + q + 2$ parametri. [Secțiunea următoare](#sec-2) studiază separat partea autoregresivă, [cea care urmează după ea](#sec-3) partea de medie mobilă, iar [a patra secțiune](#sec-4) pe amîndouă împreună.

# Modele autoregresive

Un model autoregresiv regresează seria pe propriul trecut. Secțiunea derivă momentele modelului AR(1), extinde condiția de staționaritate la AR($p$) prin rădăcinile polinomului caracteristic, reprezintă modelele AR(2) staționare în planul coeficienților lor și arată proprietatea care identifică un AR($p$) în date: PACF se anulează după lagul $p$.

## Modelul AR(1) {h=a02a39b0}

Modelul AR(1) este $X_t = c + \phi X_{t-1} + \varepsilon_t$ sau, cu operatorul lag, $(1 - \phi L)X_t = c + \varepsilon_t$. Scris în abateri de la medie, $X_t - \mu = \phi(X_{t-1} - \mu) + \varepsilon_t$: coeficientul $\phi$ este partea din abaterea de azi față de medie care se păstrează și mîine. Cu $\phi = 0{,}8$, o abatere de 2 astăzi se așteaptă să fie 1,6 mîine.

Substituind ecuația în ea însăși de $k$ ori, exprimăm $X_t$ printr-o valoare îndepărtată și șocurile dintre ele (primul rînd); pentru $|\phi| < 1$ și $k \to \infty$ obținem al doilea rînd:

$$\begin{aligned} X_t &= c(1 + \phi + \dots + \phi^{k-1}) + \phi^kX_{t-k} \\ &\quad + \sum_{j=0}^{k-1}\phi^j\varepsilon_{t-j}, \\ X_t &= \frac{c}{1 - \phi} + \sum_{j=0}^{\infty}\phi^j\varepsilon_{t-j}. \end{aligned} $$ {#eq:ar1-wold}

Cînd $|\phi| < 1$, termenul $\phi^kX_{t-k}$ dispare: procesul își uită punctul de plecare. Limita este forma Wold cu $\psi_j = \phi^j$, numită reprezentarea cauzală (*causal representation*) sau MA($\infty$), pentru că $X_t$ depinde doar de șocurile prezente și trecute.

Condiția $|\phi| < 1$ este esențială. Cu $\phi = 1$ modelul este mersul aleator (*random walk*) din Capitolul 1: șocurile nu se sting niciodată, varianța crește în timp și nu există o soluție staționară. Cu $|\phi| > 1$ substituția înapoi explodează; o soluție staționară există doar dacă $X_t$ este scris ca funcție de șocuri viitoare, ceea ce nu folosește la prognoza din trecut, așa că acest caz este exclus.

## Momentele unui AR(1) staționar {h=a8efa83d}

Media se obține aplicînd media modelului. Aici se folosește staționaritatea: $E[X_t] = E[X_{t-1}] = \mu$, deci $\mu = c + \phi\mu$ și $\mu = c/(1 - \phi)$. Varianța se obține la fel. Deoarece $\varepsilon_t$ este necorelat cu $X_{t-1}$, care după @eq:ar1-wold depinde doar de șocurile pînă la $t - 1$, varianța sumei este suma varianțelor: $\gamma(0) = \phi^2\gamma(0) + \sigma^2$.

$$\begin{gathered}\mu = \frac{c}{1 - \phi}, \qquad \gamma(0) = \frac{\sigma^2}{1 - \phi^2}, \\ \gamma(h) = \phi\,\gamma(h - 1), \qquad \rho(h) = \phi^h.\end{gathered} $$ {#eq:ar1-mom}

Varianța crește nelimitat cînd $|\phi| \to 1$: o serie persistentă se îndepărtează mult de medie, deși fiecare șoc are aceeași varianță $\sigma^2$. Autocovarianțele se obțin înmulțind $X_t - \mu = \phi(X_{t-1} - \mu) + \varepsilon_t$ cu $X_{t-h} - \mu$, $h \ge 1$, și aplicînd media; termenul cu șocul dispare, pentru că $\varepsilon_t$ este necorelat cu trecutul. Rezultă recurența $\gamma(h) = \phi\gamma(h - 1)$, deci $\gamma(h) = \phi^h\gamma(0)$ și $\rho(h) = \phi^h$: ACF descrește geometric, cu semne alternante cînd $\phi < 0$.

Un rezumat util al persistenței este timpul de înjumătățire (*half-life*) al unui șoc, orizontul $h$ după care jumătate din efectul lui a dispărut: $\phi^h = 1/2$, deci $h = \ln 0{,}5/\ln\phi$ pentru $0 < \phi < 1$. El transformă un coeficient într-un interval de timp care poate fi comparat între serii și frecvențe.

## Exemplu rezolvat: un AR(1) {h=0c96a85b}

Fie $X_t = 2 + 0{,}8X_{t-1} + \varepsilon_t$, cu $\sigma^2 = 1$. Modelul este staționar, deoarece $|\phi| = 0{,}8 < 1$. Media lui este $\mu = 2/(1 - 0{,}8) = 10$: termenul liber este 2, media este 10, iar confuzia dintre ele este o greșeală frecventă la citirea rezultatelor din programe.

Varianța este $\gamma(0) = 1/(1 - 0{,}64) = 2{,}78$, deci seria are o abatere standard de aproximativ 1,67, mai mare decît abaterea standard 1 a unui singur șoc, pentru că șocurile trecute se cumulează. Autocorelația la lagul 3 este $\rho(3) = 0{,}8^3 = 0{,}512$: observațiile aflate la trei perioade distanță au o corelație de 0,512.

Timpul de înjumătățire este $\ln 0{,}5/\ln 0{,}8 = 3{,}11$ perioade: după aproximativ trei perioade, jumătate dintr-un șoc a dispărut. Ponderile $\psi$ sînt $1; 0{,}8; 0{,}64; 0{,}512; \dots$, deci un șoc unitar de azi crește $X_{t+2}$ cu 0,64. Pentru un AR(1), răspunsul la impuls $\psi_j$ și autocorelația $\rho(j)$ sînt ambele egale cu $\phi^j$; această coincidență este specifică modelului AR(1) și nu se păstrează la celelalte modele din capitol.

## AR(1) pentru trei valori ale lui φ {h=bb4c9f25}

Graficul arată trei serii AR(1) simulate, de cîte 200 de observații, cu $\phi = 0{,}9$, $0{,}3$ și $-0{,}7$. Toate trei sînt generate de aceleași 200 de șocuri, așa că orice diferență dintre traiectoriile din rîndul de sus provine numai din $\phi$. Scara verticală este aceeași în cele trei panouri, ceea ce face vizibilă diferența de varianță.

Rîndul de jos arată, pentru lagurile de la 1 la 15, ACF de selecție a fiecărei traiectorii (bare) și ACF teoretică $\rho(h) = \phi^h$ (puncte portocalii). Liniile roșii punctate sînt banda $\pm 1{,}96/\sqrt{T}$, banda aproximativă de 95% pentru autocorelațiile de selecție ale unui zgomot alb. O bară în afara benzii este o autocorelație pe care această verificare o separă de zgomot; distanța dintre o bară și punctul ei este eroarea de selecție a unei singure serii de lungime 200.

## Interpretarea traiectoriilor AR(1) {h=574ce27c}

Cu $\phi = 0{,}9$ seria face oscilații lungi departe de 0. Autocorelația ei de ordinul întîi, de selecție, este $\hat\rho(1) = 0{,}88$, iar varianța de selecție este 4,98, față de valoarea teoretică $1/(1 - 0{,}81) = 5{,}26$. Pe o fereastră de cîteva zeci de observații o astfel de serie pare un trend: persistența puternică și un trend determinist se deosebesc greu cu ochiul liber, motiv pentru care Capitolul 3 introduce teste formale.

Cu $\phi = 0{,}3$ seria este apropiată de un zgomot alb: doar $\hat\rho(1) = 0{,}35$ este clar în afara benzii, iar varianța de selecție este 1,18, față de 1,10 teoretic. Cu $\phi = -0{,}7$ seria are o formă de zigzag: $\hat\rho(1) = -0{,}58$ și $\hat\rho(2) = 0{,}34$, cu semne alternante. În datele economice, autocorelația negativă apare la o depășire urmată de corecție sau la o eroare de măsurare corectată în perioada următoare.

În panoul pentru $\phi = 0{,}9$, ACF de selecție se află sub cea teoretică la laguri mari. Cu $T = 200$, autocorelațiile de selecție ale unei serii persistente sînt deplasate spre zero. Aceeași deplasare de eșantion mic afectează estimatorii lui $\phi$, așa cum arată [secțiunea despre estimare](#sec-5).

## Modelul AR(p) și polinomul lui caracteristic {h=e7a4ba59}

Modelul AR($p$) este $\phi(L)X_t = c + \varepsilon_t$, cu polinomul caracteristic (*characteristic polynomial*) $\phi(z) = 1 - \phi_1z - \dots - \phi_pz^p$ într-o variabilă complexă $z$. Conform teoremei fundamentale a algebrei, polinomul se descompune în $p$ factori liniari:

$$\phi(z) = (1 - \lambda_1z)(1 - \lambda_2z)\cdots(1 - \lambda_pz), \qquad z_i = 1/\lambda_i. $$ {#eq:charpoly}

Valorile $z_i$ sînt rădăcinile ecuației $\phi(z) = 0$, iar $\lambda_i$ sînt rădăcinile inverse (*inverse roots*). Fiecare factor $1 - \lambda_iL$ este operatorul unui AR(1) cu coeficientul $\lambda_i$, deci un AR($p$) este un produs de $p$ filtre AR(1), dintre care unele pot avea coeficienți complecși.

Condiția de staționaritate (cauzalitate) este ca toate rădăcinile să fie în afara cercului unitate (*unit circle*), $|z_i| > 1$, sau, echivalent, ca toate rădăcinile inverse să fie în interiorul lui, $|\lambda_i| < 1$; programele afișează de obicei rădăcinile inverse. Motivul este rezultatul pentru AR(1), aplicat fiecărui factor: $1/(1 - \lambda L) = \sum_j\lambda^jL^j$ converge doar dacă $|\lambda| < 1$, iar produsul celor $p$ serii convergente este forma cauzală MA($\infty$).

O rădăcină pe cercul unitate, $|z| = 1$, de exemplu $z = 1$, este o rădăcină unitară (*unit root*): procesul nu este staționar, iar Capitolul 3 tratează acest caz. O verificare rapidă necesară este $\phi_1 + \dots + \phi_p < 1$, adică $\phi(1) > 0$. Cum $\phi(0) = 1 > 0$, o valoare $\phi(1) \le 0$ ar plasa o rădăcină reală între 0 și 1, în interiorul cercului. Verificarea nu este suficientă: un model o poate trece și totuși să aibă o rădăcină în interiorul cercului, așa că rădăcinile trebuie calculate.

## AR(2): triunghiul de staționaritate {h=37d4ed3f}

Pentru modelul AR(2) $X_t = \phi_1X_{t-1} + \phi_2X_{t-2} + \varepsilon_t$, condiția asupra rădăcinilor se reduce la trei inegalități liniare:

$$\phi_1 + \phi_2 < 1, \qquad \phi_2 - \phi_1 < 1, \qquad |\phi_2| < 1. $$ {#eq:ar2-tri}

Ele definesc în planul $(\phi_1, \phi_2)$ un triunghi cu vîrfurile $(-2, -1)$, $(2, -1)$ și $(0, 1)$. Orice punct din interiorul triunghiului este un AR(2) staționar, iar orice punct din afara lui are cel puțin o rădăcină pe cercul unitate sau în interiorul lui.

Rădăcinile sînt complexe cînd $\phi_1^2 + 4\phi_2 < 0$, sub o parabolă din interiorul triunghiului. ACF este atunci un cosinus amortizat: produce pseudo-cicluri cu lungimea medie de $2\pi/\arccos\big(\phi_1/(2\sqrt{-\phi_2})\big)$ perioade, iar amplitudinea lor scade cu factorul $\sqrt{-\phi_2}$ pe perioadă, modulul rădăcinilor inverse. Ciclurile sînt „pseudo” pentru că șocurile le schimbă continuu amplitudinea și faza; seria nu are o perioadă fixă, ci doar una tipică.

ACF a unui AR(2) rezultă din ecuațiile Yule–Walker: înmulțim modelul cu $X_{t-h}$, aplicăm media și împărțim la $\gamma(0)$.

$$\begin{gathered}\rho(1) = \frac{\phi_1}{1 - \phi_2}, \\ \rho(h) = \phi_1\rho(h - 1) + \phi_2\rho(h - 2), \quad h \ge 2.\end{gathered} $$ {#eq:ar2-yw}

Prima ecuație provine din $\rho(1) = \phi_1 + \phi_2\rho(1)$, folosind $\rho(-1) = \rho(1)$. A doua arată că ACF respectă aceeași recurență ca procesul fără șocuri, așa că forma ei (descreștere geometrică sau undă amortizată) este dată de rădăcinile lui $\phi(z)$.

## Trei procese AR(2) și petele solare {h=236b8af4}

Panoul din stînga desenează triunghiul de staționaritate @eq:ar2-tri, colorat, și parabola punctată $\phi_1^2 + 4\phi_2 = 0$, care separă rădăcinile reale (deasupra ei) de cele complexe (sub ea). Sînt marcate patru modele: A cu $\phi = (0{,}5; 0{,}3)$, B cu $(1{,}0; -0{,}6)$, C cu $(0{,}6; 0{,}5)$ și AR(2) estimat de Yule pentru petele solare, reluat în [secțiunea 10](#sec-10). A se află în triunghi, în regiunea rădăcinilor reale, B și petele solare în triunghi, în regiunea rădăcinilor complexe, iar C chiar în afara laturii din dreapta.

Panoul din mijloc arată rădăcinile inverse $\lambda_i$ în planul complex, împreună cu cercul unitate. Un model este staționar cînd toate punctele lui sînt în interiorul cercului. A are două rădăcini inverse reale în interior; B și petele solare au cîte o pereche de rădăcini inverse complex conjugate în interior; C are o rădăcină inversă pe axa reală, în afara cercului.

Panoul din dreapta arată ACF teoretică pentru A și B pînă la lagul 30. A descrește lent și neted; B oscilează cu amplitudine descrescătoare, cosinusul amortizat al rădăcinilor complexe.

## Interpretarea exemplelor AR(2) {h=839c83ba}

Modelul A are rădăcinile reale $-2{,}84$ și $1{,}17$, ambele în afara cercului unitate, deci este staționar. Autocorelațiile lui, din @eq:ar2-yw, sînt $\rho(1) = 0{,}5/0{,}7 = 0{,}714$ și $\rho(2) = 0{,}5\rho(1) + 0{,}3 = 0{,}657$: o descreștere lentă și netedă, dominată de rădăcina 1,17, cea mai apropiată de cerc.

Modelul B are rădăcinile complexe $0{,}833 \pm 0{,}986\,i$, cu modulul $1{,}291 > 1$, deci este staționar. ACF lui este o undă cu perioada de 7,2 laguri, amortizată cu factorul 0,775 pe lag. Un astfel de model generează un ciclu stochastic (*stochastic cycle*): oscilații cu o lungime tipică, dar cu amplitudine și momente aleatoare.

Modelul C are $\phi_1 + \phi_2 = 1{,}1 > 1$, deci nu trece condiția necesară; o rădăcină este $0{,}936 < 1$, în interiorul cercului, iar procesul este exploziv. Fiecare coeficient este sub 1, și totuși procesul explodează: staționaritatea este o proprietate a rădăcinilor, nu a coeficienților luați pe rînd. Rădăcinile complexe sînt mecanismul prin care modelele liniare produc ciclurile economice și, în [secțiunea 10](#sec-10), ciclul solar de 11 ani.

## PACF a unui AR(p) se anulează după lagul p {h=a5ca3937}

Autocorelația parțială $\phi_{hh}$ este ultimul coeficient din cea mai bună predicție liniară a lui $X_t$ pe baza lui $X_{t-1}, \dots, X_{t-h}$ (Capitolul 1). Nu este nevoie de cîte o regresie pentru fiecare $h$: recurența Durbin–Levinson [@Durbin1960] calculează toate valorile $\phi_{hh}$ din ACF.

Pentru un AR($p$) și $h > p$, cea mai bună predicție liniară a lui $X_t$ din $h$ laguri este chiar modelul, $\phi_1X_{t-1} + \dots + \phi_pX_{t-p}$: eroarea $\varepsilon_t$ este necorelată cu toate valorile trecute, deci niciun lag suplimentar nu poate îmbunătăți predicția. Coeficienții lagurilor $p + 1, \dots, h$ sînt așadar zero, de unde $\phi_{hh} = 0$ pentru $h > p$ și $\phi_{pp} = \phi_p$.

Aceasta este regula de identificare pentru autoregresii: cînd ACF descrește treptat, iar PACF are $p$ valori semnificative și apoi rămîne în bandă, candidatul este un AR($p$). Pentru PACF de selecție a unui AR($p$) la lagurile $h > p$, banda este $\pm 1{,}96/\sqrt{T}$. Cînd se examinează multe laguri, aproximativ o valoare din douăzeci iese întîmplător din bandă, așa că o valoare izolată aflată imediat în afara ei, la un lag mare, este o dovadă slabă.

## Recapitulare: modele autoregresive {h=32207a8e}

Un AR($p$), $\phi(L)X_t = c + \varepsilon_t$, este staționar dacă și numai dacă toate rădăcinile lui $\phi(z)$ sînt în afara cercului unitate; verificarea coeficienților unul cîte unul nu ajunge. Media lui este $\mu = c/\phi(1)$, iar ACF urmează recurența Yule–Walker @eq:ar2-yw, care produce o combinație de descreșteri geometrice (rădăcini reale) și unde amortizate (rădăcini complexe).

AR(1) este piesa de bază: $\rho(h) = \psi_h = \phi^h$, iar timpul de înjumătățire al unui șoc este $\ln 0{,}5/\ln\phi$. În date, un AR($p$) se recunoaște după PACF, care se anulează după lagul $p$.

## @end

Modelele autoregresive descriu memoria prin valorile trecute ale seriei. Staționaritatea cere ca rădăcinile lui $\phi(z)$ să fie în afara cercului unitate @eq:charpoly; pentru un AR(2), aceasta înseamnă triunghiul @eq:ar2-tri. Rădăcinile reale produc descreșteri geometrice ale ACF, cele complexe unde amortizate și cicluri stochastice. Semnătura unui AR($p$) în date este o PACF care se anulează după lagul $p$. [Secțiunea următoare](#sec-3) studiază cazul opus: memoria prin șocurile trecute.

# Modele de medie mobilă

Un model de medie mobilă descrie seria ca o sumă ponderată a șocului curent și a ultimelor $q$ șocuri. Un astfel de model este întotdeauna staționar, iar ACF lui se anulează după lagul $q$. Problema nouă este invertibilitatea: două modele MA diferite pot avea aceleași autocorelații și este nevoie de o convenție pentru a alege între ele.

## Modelul MA(q) {h=4d29a7b5}

Modelul MA($q$) este $X_t = \mu + \theta(L)\varepsilon_t = \mu + \varepsilon_t + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q}$. El este o sumă Wold cu un număr finit de ponderi nenule, $\psi_j = \theta_j$ pentru $j \le q$ și 0 după aceea. O sumă finită de termeni de zgomot alb are întotdeauna medie constantă și varianță finită, deci un MA($q$) este staționar pentru orice valori ale coeficienților.

Momentele rezultă din faptul că șocurile sînt necorelate. Cu $\theta_0 = 1$, $E[X_t] = \mu$ și

$$\begin{gathered}\gamma(h) = \sigma^2\sum_{j=0}^{q-h}\theta_j\theta_{j+h}, \quad 0 \le h \le q, \\ \gamma(h) = 0, \quad h > q.\end{gathered} $$ {#eq:ma-acov}

Două valori aflate la mai mult de $q$ perioade distanță nu au niciun șoc comun, deci covarianța lor este zero.

Regula de identificare este imaginea în oglindă a regulii pentru AR: ACF se anulează după lagul $q$, iar PACF descrește treptat. Pentru ACF de selecție la lagurile $h > q$, formula lui Bartlett dă banda $\pm 1{,}96\sqrt{(1 + 2\sum_{j=1}^{q}\hat\rho(j)^2)/T}$. Ea este mai largă decît $\pm 1{,}96/\sqrt{T}$, deoarece varianța de selecție a unei autocorelații de după lagul $q$ depinde de autocorelațiile pînă la lagul $q$; cu banda îngustă, o ACF de selecție de dincolo de ordinul adevărat ar părea semnificativă prea des.

## MA(1): θ = 0,8 și θ = -0,8 {h=0dbbb4a5}

Graficul arată două serii MA(1) simulate, de cîte $T = 300$ de observații, cu $\theta = 0{,}8$ (rîndul de sus) și $\theta = -0{,}8$ (rîndul de jos). Pentru fiecare, panoul din stînga este traiectoria, cel din mijloc ACF, iar cel din dreapta PACF, pentru lagurile de la 1 la 12; barele sînt valorile de selecție, punctele portocalii valorile teoretice, iar liniile roșii punctate banda $\pm 1{,}96/\sqrt{T}$.

În ambele panouri ACF există o singură valoare clară, la lagul 1: pozitivă pentru $\theta = 0{,}8$ și negativă pentru $\theta = -0{,}8$. PACF nu se anulează: descrește, cu semne alternante pentru $\theta = 0{,}8$ și cu toate valorile negative pentru $\theta = -0{,}8$. Traiectoria cu $\theta = 0{,}8$ este mai netedă, pentru că valorile vecine au în comun un șoc cu același semn; traiectoria cu $\theta = -0{,}8$ este mai zimțată.

## Interpretarea corelogramelor MA(1) {h=b1b502a7}

Pentru un MA(1), @eq:ma-acov dă $\rho(1) = \theta/(1 + \theta^2)$, adică $\pm 0{,}488$ pentru $\theta = \pm 0{,}8$, și $\rho(h) = 0$ pentru $h \ge 2$. Valorile de selecție sînt $\hat\rho(1) = 0{,}56$ și $-0{,}51$; diferențele față de $\pm 0{,}488$ sînt eroare de selecție cu 300 de observații.

PACF nu se anulează: pentru $\theta = 0{,}8$, valoarea de selecție la lagul 2 este $\hat\phi_{22} = -0{,}23$. Motivul, dezvoltat în cadrele următoare, este că un MA(1) invertibil se poate scrie ca AR($\infty$): fiecare valoare trecută a seriei conține puțină informație despre șocul curent, deci fiecare lag intră în cea mai bună predicție liniară cu un coeficient nenul.

Un MA(1) își amintește exact o perioadă: un șoc de azi afectează valoarea de mîine și apoi este uitat. Această structură apare în datele economice în mai multe feluri: la observațiile care se suprapun, ca ratele de creștere ale PIB-ului din [secțiunea 9](#sec-9); la o eroare de măsurare care intră într-o observație și este corectată în următoarea; și la oscilația bid–ask (*bid–ask bounce*), în care prețurile tranzacțiilor alternează între prețul de cumpărare și cel de vînzare, așa că variațiile consecutive de preț sînt corelate negativ.

## Cea mai mare autocorelație a unui MA(1) {h=07a36729}

Graficul reprezintă $\rho(1) = \theta/(1 + \theta^2)$ pentru $\theta$ între $-4$ și $4$. Banda colorată marchează regiunea invertibilă $|\theta| < 1$, iar liniile roșii punctate marchează $\pm 0{,}5$. Sînt evidențiate două puncte: $\theta = 0{,}5$ și $\theta = 2$, ambele cu $\rho(1) = 0{,}4$.

Curba este impară în $\theta$, atinge maximul 0,5 la $\theta = 1$ și minimul $-0{,}5$ la $\theta = -1$ și tinde spre 0 cînd $|\theta|$ crește. Orice valoare a lui $\rho(1)$ strict între $-0{,}5$ și $0{,}5$, cu excepția lui 0, este atinsă de două ori: o dată în regiunea colorată și o dată în afara ei.

## Interpretarea curbei ρ(1) {h=e60835ad}

Cea mai mare valoare posibilă este $|\rho(1)| = 0{,}5$, atinsă la $\theta = \pm 1$. Ea rezultă din $(1 - |\theta|)^2 \ge 0$, care dă $1 + \theta^2 \ge 2|\theta|$. Un MA(1) nu poate produce deci niciodată $|\rho(1)| > 0{,}5$. Dacă o ACF de selecție arată $\hat\rho(1) = 0{,}7$ și apoi nimic semnificativ, seria nu este un MA(1); candidații sînt un model AR sau un MA($q$) cu mai mulți termeni.

Valorile $\theta = 0{,}5$ și $\theta = 2$ dau același $\rho(1) = 0{,}4$. În general, $\theta$ și $1/\theta$ dau aceeași ACF: înlocuirea lui $\theta$ cu $1/\theta$ lasă neschimbat raportul $\theta/(1 + \theta^2)$, iar autocovarianțele coincid și ele dacă $\sigma^2$ este înmulțit cu $\theta^2$. Datele nu pot deosebi cele două modele; aceasta este o problemă de identificare (*identification problem*).

Convenția care elimină ambiguitatea este invertibilitatea, $|\theta| < 1$, subiectul cadrului următor.

## Invertibilitatea {h=fd010348}

Pentru un MA(1) cu $\mu = 0$, modelul poate fi rezolvat în raport cu șocul: $\varepsilon_t = X_t - \theta\varepsilon_{t-1}$. Substituind repetat această expresie în ea însăși, obținem

$$\begin{gathered}\varepsilon_t = X_t - \theta X_{t-1} + \theta^2X_{t-2} - \dots \\ = \sum_{j \ge 0}(-\theta)^jX_{t-j}, \qquad |\theta| < 1.\end{gathered} $$ {#eq:ma1-inv}

Rearanjată, aceasta este o reprezentare AR($\infty$), $X_t = \theta X_{t-1} - \theta^2X_{t-2} + \dots + \varepsilon_t$, care explică de ce PACF a unui MA(1) nu se anulează niciodată.

Un model MA este invertibil (*invertible*) cînd șocurile lui pot fi recuperate din valorile trecute ale lui $X$. Condiția este ca toate rădăcinile ecuației $\theta(z) = 0$ să fie în afara cercului unitate; pentru MA(1) aceasta înseamnă $|\theta| < 1$. Este aceeași algebră ca la staționaritatea unui AR, aplicată lui $\theta(z)$ în loc de $\phi(z)$.

Invertibilitatea este necesară din trei motive. Ea face modelul unic: dintre $\theta$ și $1/\theta$, doar unul este invertibil. Ea face posibilă prognoza: șocurile $\varepsilon_t$ se calculează din date prin @eq:ma1-inv, iar prognozele le folosesc. De aceea programele raportează estimări invertibile, iar o rădăcină estimată apropiată de cercul unitate este un semnal de alarmă: este semnul tipic al supradiferențierii (*over-differencing*, Capitolul 1), deoarece diferențierea unei serii deja staționare creează o rădăcină MA egală cu 1.

## Exemplu rezolvat: θ = 2 sau θ = 0,5? {h=69992177}

Fie $X_t = \varepsilon_t + 2\varepsilon_{t-1}$, cu $\sigma^2 = 1$, și $Y_t = u_t + 0{,}5u_{t-1}$, cu $\mathrm{Var}(u_t) = 4$. Din @eq:ma-acov, $X$ are $\gamma(0) = 1 \cdot (1 + 4) = 5$ și $\gamma(1) = 2 \cdot 1 = 2$, iar $Y$ are $\gamma(0) = 4 \cdot (1 + 0{,}25) = 5$ și $\gamma(1) = 0{,}5 \cdot 4 = 2$. Ambele au media zero.

Cele două procese au aceeași medie și aceleași autocovarianțe. Cu șocuri gaussiene, un proces staționar este complet determinat de media și autocovarianțele lui, așa că $X$ și $Y$ au aceeași distribuție: oricîte date am avea, nu le putem separa.

Doar $Y$ este invertibil: $u_t = Y_t - 0{,}5Y_{t-1} + 0{,}25Y_{t-2} - \dots$, o sumă convergentă. Pentru $X$, aceeași sumă are ponderile $(-2)^j$ și explodează. Regula este să raportăm versiunea invertibilă, $\theta = 0{,}5$ cu $\sigma^2 = 4$. Șocurile ei $u_t$ sînt erorile celei mai bune predicții cu un pas din trecutul seriei, motiv pentru care varianța zgomotului în versiunea invertibilă (4) este mai mare decît în cea neinvertibilă (1).

## Recapitulare: modele de medie mobilă {h=3909228b}

Un MA($q$) este întotdeauna staționar; ACF lui se anulează după lagul $q$, iar PACF descrește. Pentru un MA(1), $\rho(1) = \theta/(1 + \theta^2)$, cel mult 0,5 în valoare absolută; o primă autocorelație mai mare exclude modelul MA(1).

Un MA este invertibil dacă și numai dacă rădăcinile lui $\theta(z)$ sînt în afara cercului unitate; atunci el se poate scrie ca AR($\infty$). Parametrii $\theta$ și $1/\theta$ sînt echivalenți din punctul de vedere al datelor și se raportează cel invertibil.

## @end

Modelul MA($q$) are memorie finită: după @eq:ma-acov, autocorelațiile lui sînt exact zero după lagul $q$, iar această anulare îl identifică în date. Staționaritatea este automată; condiția care contează este invertibilitatea, care face modelul unic și permite calcularea șocurilor din date @eq:ma1-inv. [Secțiunea următoare](#sec-4) combină partea autoregresivă cu cea de medie mobilă.

# Modele ARMA(p,q)

Modelul ARMA($p,q$) are atît o parte autoregresivă, cît și una de medie mobilă. Secțiunea formulează cele trei condiții ale lui (cauzalitate, invertibilitate, absența factorilor comuni), calculează răspunsul la impuls prin ponderile $\psi$, derivă momentele modelului ARMA(1,1) și adună într-un singur tabel regulile de identificare din tot capitolul.

## Modelul ARMA(p,q) {h=e10d114d}

Modelul ARMA($p,q$) este $\phi(L)(X_t - \mu) = \theta(L)\varepsilon_t$, ca în @eq:arma, sau, scris desfășurat,

$$\begin{aligned} X_t &= c + \phi_1X_{t-1} + \dots + \phi_pX_{t-p} \\ &\quad + \varepsilon_t + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q}, \\ c &= \mu\,\phi(1). \end{aligned} $$ {#eq:arma-full}

Trei condiții fac modelul bine definit; @BrockwellDavis2016 le tratează în capitolul 3 al cărții lor. El este staționar și cauzal cînd rădăcinile lui $\phi(z)$ sînt în afara cercului unitate; atunci $X_t - \mu = \psi(L)\varepsilon_t$, cu $\psi(L) = \theta(L)/\phi(L)$. Este invertibil cînd rădăcinile lui $\theta(z)$ sînt în afara cercului unitate; atunci $\pi(L)(X_t - \mu) = \varepsilon_t$, cu $\pi(L) = \phi(L)/\theta(L)$. În sfîrșit, $\phi(z)$ și $\theta(z)$ nu trebuie să aibă nicio rădăcină comună, altfel modelul are mai mulți parametri decît procesul pe care îl descrie.

Avantajul combinării celor două părți se vede la ARMA(1,1). ACF lui descrește geometric cu rata $\phi$, ca la un AR(1), dar prima ei valoare diferă de $\phi$: se află sub $\phi$ cînd $\theta < 0$ și peste $\phi$ cînd $\theta > 0$. Un AR pur sau un MA pur ar avea nevoie de mulți termeni pentru a reproduce această formă; ARMA(1,1) o face cu doi coeficienți.

## Ponderile ψ și răspunsul la impuls {h=984e0124}

Forma cauzală cere coeficienții lui $\psi(L) = \theta(L)/\phi(L)$. Scriind $\phi(L)\psi(L) = \theta(L)$ și identificînd coeficienții fiecărei puteri $L^j$, obținem o recurență:

$$\begin{gathered}\psi_0 = 1, \qquad \psi_j = \theta_j + \sum_{i=1}^{\min(j,p)}\phi_i\psi_{j-i}, \\ \theta_j = 0 \text{ pentru } j > q.\end{gathered} $$ {#eq:psi}

Pentru ARMA(1,1), $\psi_1 = \phi + \theta$ și $\psi_j = \phi^{j-1}(\phi + \theta)$ pentru $j \ge 1$. Cu $\phi = 0{,}7$ și $\theta = 0{,}4$: $\psi_1 = 1{,}10$ și $\psi_2 = 0{,}77$.

Ponderile au o interpretare directă: $\psi_j = \partial X_{t+j}/\partial\varepsilon_t$, efectul unui șoc unitar de la momentul $t$ asupra valorii de peste $j$ perioade, cu toate celelalte șocuri fixate. Șirul $(\psi_j)$ este răspunsul la impuls (*impulse response*) al modelului. Același obiect determină varianța erorilor de prognoză ([secțiunea 7](#sec-7)) și, într-o formă multivariată, răspunsurile la impuls ale modelelor VAR din Capitolul 6.

## Răspunsurile la impuls ale a patru modele {h=06067b29}

Graficul arată $\psi_j$ pentru $j = 0, \dots, 15$, calculate cu recurența @eq:psi, pentru patru modele: AR(1) cu $\phi = 0{,}8$, AR(2) cu $\phi = (1{,}0; -0{,}6)$, MA(2) cu $\theta = (0{,}6; 0{,}3)$ și ARMA(1,1) cu $\phi = 0{,}7$, $\theta = 0{,}4$. Fiecare răspuns începe cu $\psi_0 = 1$, efectul imediat al șocului asupra valorii curente.

Înălțimea barei $j$ este modificarea lui $X_{t+j}$ după un șoc unitar la momentul $t$. Suma pătratelor înălțimilor, înmulțită cu $\sigma^2$, este varianța $\gamma(0)$ a procesului, așa că un răspuns care se stinge lent înseamnă atît memorie lungă, cît și varianță mare.

## Interpretarea răspunsurilor la impuls {h=737de244}

Pentru AR(1) răspunsul descrește geometric, $\psi_j = 0{,}8^j$. Șocul nu dispare niciodată complet, dar se estompează suficient de repede pentru ca $\sum_j\psi_j^2$ să fie finită, condiția din descompunerea Wold. Pentru AR(2) cu rădăcini complexe, $\psi_2 = 0{,}4$, după care răspunsul își schimbă semnul: $\psi_3 = -0{,}20$, $\psi_4 = -0{,}44$. Un singur șoc creează un ciclu: seria întîi crește, apoi coboară sub medie, apoi revine.

Pentru MA(2) răspunsul este exact zero după două perioade: memoria modelului este finită. Pentru ARMA(1,1), termenul de medie mobilă ridică primul răspuns la $\psi_1 = 1{,}10 > 1$, peste efectul imediat; după aceea preia partea autoregresivă, iar fiecare răspuns este 0,7 din cel precedent.

Aceste forme reapar în prognoză: viteza cu care descresc ponderile $\psi_j$ este viteza cu care prognozele revin la medie și cu care intervalele de prognoză se lărgesc pînă la lățimea lor de termen lung.

## Momentele procesului ARMA(1,1) {h=325a4689}

Fie $X_t = \phi X_{t-1} + \varepsilon_t + \theta\varepsilon_{t-1}$, cu $|\phi| < 1$ (cauzal), $|\theta| < 1$ (invertibil) și $\phi + \theta \ne 0$ (fără factor comun). Varianța rezultă din forma cauzală, $\gamma(0) = \sigma^2\sum_j\psi_j^2$, cu ponderile $\psi_j = \phi^{j-1}(\phi + \theta)$: însumînd seria geometrică, obținem $\gamma(0) = \sigma^2(1 + 2\phi\theta + \theta^2)/(1 - \phi^2)$. Pentru $\phi = 0{,}7$, $\theta = 0{,}4$ și $\sigma^2 = 1$, aceasta este 3,37.

Autocorelațiile sînt

$$\begin{gathered}\rho(1) = \frac{(1 + \phi\theta)(\phi + \theta)}{1 + 2\phi\theta + \theta^2}, \\ \rho(h) = \phi\,\rho(h - 1), \quad h \ge 2.\end{gathered} $$ {#eq:arma11-rho}

În exemplu, $\rho(1) = 0{,}819$ și $\rho(2) = 0{,}7 \cdot 0{,}819 = 0{,}573$.

De la lagul 1 încolo, ACF descrește ca la un AR(1) cu coeficientul $\phi$, dar valoarea de pornire este $\rho(1) \ne \phi$: partea de medie mobilă mută doar punctul de pornire. Acesta este tiparul din tabelul de identificare pentru modelele ARMA: ACF descrește după lagul $q - p$, aici după lagul 0, cu o primă valoare pe care partea AR singură nu ar da-o.

## Factori comuni: un model prea mare {h=d3ddb8db}

Fie $X_t = 0{,}5X_{t-1} + \varepsilon_t - 0{,}5\varepsilon_{t-1}$. Cu operatorul lag, modelul este $(1 - 0{,}5L)X_t = (1 - 0{,}5L)\varepsilon_t$; simplificînd factorul comun (*common factor*), obținem $X_t = \varepsilon_t$, un zgomot alb. Modelul ARMA(1,1) este o descriere supraparametrizată a unui proces fără nicio memorie.

Mai general, orice ARMA(1,1) cu $\phi = -\theta$ este zgomot alb, oricare ar fi valoarea comună. Verosimilitatea este deci plată de-a lungul dreptei $\phi = -\theta$, iar datele nu pot localiza parametrii pe ea: aceștia nu sînt identificați.

În rezultatele programelor, un factor comun apare ca estimări mari de semne opuse, de exemplu $\hat\phi = 0{,}9$ și $\hat\theta = -0{,}88$, cu erori standard foarte mari, și ca o log-verosimilitate care nu este mai bună decît a modelului mai mic. Remediul este să reducem $p$ și $q$ împreună și să preferăm cel mai mic model ale cărui reziduuri sînt zgomot alb.

## Identificarea: tabelul ACF–PACF {h=30f6ab77}

Tabelul rezumă tiparele teoretice. Zgomotul alb nu are autocorelații semnificative în niciuna dintre funcții. Un AR($p$) are o ACF care descrește (geometric sau ca o undă amortizată) și o PACF care se anulează după lagul $p$. Un MA($q$) are situația opusă: o ACF care se anulează după lagul $q$ și o PACF care descrește. Un ARMA($p,q$) are ambele funcții descrescătoare: ACF după lagul $q - p$ și PACF după lagul $p - q$, adică primele valori sînt fixate liber de partea MA, iar următoarele urmează recurența AR.

Ultimul rînd anticipează Capitolul 3: pentru o serie cu rădăcină unitară, ACF de selecție descrește foarte lent, aproape liniar, iar prima autocorelație parțială $\hat\phi_{11}$ este apropiată de 1. O astfel de serie nu se modelează ca ARMA în niveluri; ea se diferențiază mai întîi.

Tabelul propune candidați; el nu alege modelul. Criteriile informaționale și diagnosticarea reziduurilor din [secțiunile 5](#sec-5) și [6](#sec-6) decid între ei. Modelele ARMA mixte sînt cel mai greu de recunoscut cu ochiul liber, pentru că ambele funcții doar descresc.

## ACF și PACF teoretice și de selecție {h=4a7fea59}

Graficul aplică tabelul la trei serii simulate, de cîte $T = 500$ de observații: un AR(2) cu $\phi = (1{,}0; -0{,}6)$, un MA(2) cu $\theta = (0{,}6; 0{,}3)$ și un ARMA(1,1) cu $\phi = 0{,}7$, $\theta = 0{,}4$. Rîndul de sus arată ACF, iar cel de jos PACF, pentru lagurile de la 1 la 15.

În fiecare panou, barele sînt valorile de selecție din singura traiectorie simulată, punctele portocalii valorile teoretice, iar liniile roșii punctate banda $\pm 1{,}96/\sqrt{T}$. Comparația dintre bare și puncte arată cît zgomot conține o corelogramă de selecție chiar și cu 500 de observații; numărarea barelor din afara benzii, funcție cu funcție, este pasul de identificare.

## Interpretarea celor trei tipare {h=0c5265e3}

Pentru AR(2), PACF are două valori semnificative, a doua fiind $\phi_{22} = \phi_2 = -0{,}60$, apoi rămîne la zero; ACF este o undă amortizată. Identificarea este ușoară: ordinul $p$ se citește din PACF.

Pentru MA(2), ACF teoretică este $\rho(1) = 0{,}538$, $\rho(2) = 0{,}207$ și zero după aceea, iar PACF descrește lent. ACF de selecție la lagul 2 este doar puțin în afara benzii: cu date reale, un MA(1) și un MA(2) pot arăta la fel, iar alegerea dintre ele rămîne criteriilor informaționale.

Pentru ARMA(1,1), ambele funcții descresc. ACF arată ca a unui AR(1), iar PACF are o a doua valoare, mai mică și de semn opus. Un analist care citește doar PACF ar alege un AR(2) sau un AR(3): o aproximare bună, dar cu mai mulți parametri. Corelogramele de selecție sînt zgomotoase chiar cu $T = 500$; cu 100 de observații trimestriale banda este mai largă, iar tiparele sînt mult mai puțin clare.

## Recapitulare: modele ARMA {h=5c56c8e8}

Un ARMA($p,q$), $\phi(L)(X_t - \mu) = \theta(L)\varepsilon_t$, este cauzal cînd rădăcinile lui $\phi(z)$ sînt în afara cercului unitate și invertibil cînd rădăcinile lui $\theta(z)$ sînt în afara lui. Răspunsul lui la impuls, adică ponderile $\psi$, rezultă din recurența @eq:psi.

Cele două polinoame nu trebuie să aibă factori comuni; o verosimilitate plată și estimări mari de semne opuse sînt simptomele unui factor comun. Identificarea folosește tabelul: o ACF care se anulează indică un MA, o PACF care se anulează un AR, două funcții descrescătoare un ARMA.

## @end

Modelul ARMA($p,q$) combină un polinom AR scurt, care fixează descreșterea de termen lung a memoriei, cu un polinom MA scurt, care modelează primele laguri. Răspunsul la impuls @eq:psi rezumă dinamica, iar pentru ARMA(1,1) autocorelațiile @eq:arma11-rho descresc ca la un AR(1), dintr-un alt punct de pornire. Corelogramele doar sugerează ordine candidate. [Secțiunea următoare](#sec-5) estimează parametrii unui candidat, iar [cea de după ea](#sec-6) alege între candidați și verifică modelul ales.

# Estimarea

După alegerea ordinelor candidate, parametrii trebuie estimați. Secțiunea prezintă trei estimatori: Yule–Walker, o metodă a momentelor; cele mai mici pătrate condiționate; și verosimilitatea maximă gaussiană exactă. Apoi compară comportamentul lor în eșantioane de 100 de observații.

## Trei estimatori {h=e3ae0288}

Estimatorul Yule–Walker [@Yule1927; @Walker1931] este o metodă a momentelor (*method of moments*): înlocuiește autocorelațiile teoretice din ecuațiile Yule–Walker cu autocorelațiile de selecție și rezolvă sistemul în raport cu coeficienții. Se aplică doar modelelor AR pure. Este rapid și dă întotdeauna un model staționar, dar este mai puțin precis cînd o rădăcină este aproape de cercul unitate.

Metoda celor mai mici pătrate condiționate (*conditional least squares*) minimizează suma pătratelor șocurilor $\sum_t\varepsilon_t^2$, tratînd primele observații ca fixe. Pentru un AR($p$), aceasta este metoda celor mai mici pătrate (OLS, *ordinary least squares*) pentru $X_t$ pe $1, X_{t-1}, \dots, X_{t-p}$. Pentru modelele MA și ARMA șocurile nu sînt observate: ele se calculează recursiv, $\varepsilon_t = X_t - \mu - \theta\varepsilon_{t-1} - \dots$, pornind de la $\varepsilon_0 = 0$. Suma pătratelor este atunci o funcție neliniară de $\theta$ și se minimizează numeric.

Verosimilitatea maximă gaussiană exactă (MLE, *maximum likelihood estimation*) maximizează densitatea comună a lui $(X_1, \dots, X_T)$ și folosește deci și informația din primele observații. Ea este metoda implicită a clasei ARIMA din statsmodels, care calculează verosimilitatea cu filtrul Kalman (Capitolul 10). În eșantioane mari, cei trei estimatori coincid pentru modelele AR pure; pentru modelele MA și ARMA, MLE este referința.

## Yule–Walker pentru un AR(p) {h=20353451}

Pentru un AR($p$), înmulțind modelul cu $X_{t-h}$, aplicînd media și împărțind la $\gamma(0)$, obținem, pentru $h = 1, \dots, p$, ecuațiile $\rho(h) = \phi_1\rho(h - 1) + \dots + \phi_p\rho(h - p)$. Sub formă matriceală, ele sînt $R\,\phi = \rho$, unde $R$ este matricea $p \times p$ cu elementele $R_{ij} = \rho(|i - j|)$, $\phi = (\phi_1, \dots, \phi_p)^\top$ și $\rho = (\rho(1), \dots, \rho(p))^\top$. Estimatorul înlocuiește fiecare $\rho(h)$ cu valoarea de selecție, marcată cu căciulă:

$$\hat\phi = \hat R^{-1}\hat\rho, \qquad \hat\sigma^2 = \hat\gamma(0)\,(1 - \hat\phi^\top\hat\rho). $$ {#eq:yw}

A doua formulă împarte varianța seriei în partea explicată de trecut, $\hat\gamma(0)\hat\phi^\top\hat\rho$, și varianța șocului.

Pentru un AR(2), regula lui Cramer dă formule explicite: $\hat\phi_1 = \hat\rho_1(1 - \hat\rho_2)/(1 - \hat\rho_1^2)$ și $\hat\phi_2 = (\hat\rho_2 - \hat\rho_1^2)/(1 - \hat\rho_1^2)$. A doua este autocorelația parțială de selecție la lagul 2 din Capitolul 1, în acord cu rezultatul $\phi_{pp} = \phi_p$. Gilbert Walker (1868–1958), al cărui portret însoțește slide-ul, a extins metoda lui Yule la autoregresii de orice ordin în 1931.

## Exemplu rezolvat: Yule–Walker pentru creșterea PIB-ului României {h=02d1f66b}

Creșterea anuală a PIB-ului real al României, T1 2000–T2 2026, studiată în [secțiunea 9](#sec-9), are autocorelațiile de selecție $\hat\rho(1) = 0{,}62$ și $\hat\rho(2) = 0{,}40$ și varianța de selecție $\hat\gamma(0) = 18{,}77$, cu împărțire la $T$, ca în definiția din Capitolul 1. Determinantul lui $\hat R$ este $1 - \hat\rho_1^2 = 0{,}620$, calculat cu valoarea nerotunjită a lui $\hat\rho(1)$.

Formulele lui Cramer dau $\hat\phi_1 = 0{,}62\,(1 - 0{,}40)/0{,}620 = 0{,}600$ și $\hat\phi_2 = (0{,}40 - 0{,}62^2)/0{,}620 = 0{,}027$. Varianța zgomotului, din @eq:yw, este $\hat\sigma^2 = 18{,}77\,(1 - \hat\phi_1\hat\rho_1 - \hat\phi_2\hat\rho_2) = 11{,}62$, față de varianța 18,77 a seriei înseși.

Al doilea coeficient este practic zero: în familia AR, al doilea lag nu adaugă nimic, iar un AR(1) ar fi suficient. Yule–Walker nu întreabă însă dacă familia AR este cea potrivită. [Secțiunea 9](#sec-9) arată că pentru această serie un MA(3) se potrivește mai bine.

## Verosimilitatea maximă {h=162ebd04}

Verosimilitatea gaussiană a unui model ARMA se calculează prin descompunerea erorilor de predicție (*prediction-error decomposition*). Densitatea comună se descompune în densități condiționate, $f(x_1, \dots, x_T) = \prod_t f(x_t \mid x_{t-1}, \dots, x_1)$, iar cu șocuri gaussiene fiecare factor este o densitate Normală cu media $\hat x_{t|t-1}$, predicția cu un pas, și varianța $v_t$, varianța erorii ei:

$$\ln L = -\frac12\sum_{t=1}^{T}\Big[\ln(2\pi v_t) + \frac{(x_t - \hat x_{t|t-1})^2}{v_t}\Big]. $$ {#eq:loglik}

Varianțele $v_t$ sînt mai mari pentru primele observații, care sînt prezise dintr-un trecut scurt sau inexistent.

Pentru un AR(1), descompunerea se poate scrie de mînă. Prima observație are distribuția staționară, $X_1 \sim N(\mu, \sigma^2/(1 - \phi^2))$, iar fiecare observație următoare are $X_t \mid X_{t-1} \sim N(c + \phi X_{t-1}, \sigma^2)$. Dacă renunțăm la primul factor, rămîne verosimilitatea condiționată, a cărei maximizare este metoda celor mai mici pătrate condiționate. Diferența dintre cei doi estimatori provine deci din prima observație și contează mai ales în eșantioane mici și cînd $\phi$ este aproape de 1.

Erorile standard (se, *standard errors*) provin din curbura lui $\ln L$ în punctul de maxim. Asimptotic, $\mathrm{se}(\hat\phi) \approx \sqrt{(1 - \phi^2)/T}$ pentru un AR(1) și $\mathrm{se}(\hat\theta) \approx \sqrt{(1 - \theta^2)/T}$ pentru un MA(1). Rapoartele $t$, $\hat\phi/\mathrm{se}$, se compară cu $\pm 1{,}96$, ca în regresie. Formula pentru AR(1) tinde spre zero cînd $|\phi| \to 1$; cadrele următoare arată că aproape de cercul unitate această aproximare este prea optimistă. Implicit, statsmodels raportează o altă estimare, eroarea standard OPG (*outer product of gradients*, produsul exterior al gradienților), construită din pătratele scorurilor fiecărei observații. Dacă modelul este corect, cele două estimări au aceeași limită; cînd erorile au cozi groase, erorile standard OPG sînt mai mici decît cele din curbură. Toate erorile standard din acest capitol sînt OPG.

## Distribuțiile de selecție ale estimatorilor {h=1b31721f}

Graficul arată distribuțiile estimațiilor pe 400 de eșantioane simulate, de cîte $T = 100$ de observații. Panoul din stînga este un AR(1) cu $\phi = 0{,}9$, estimat prin Yule–Walker (portocaliu), cele mai mici pătrate condiționate (albastru) și verosimilitate maximă gaussiană (roșu); panoul din dreapta este un MA(1) cu $\theta = 0{,}5$, estimat prin cele mai mici pătrate condiționate și verosimilitate maximă. Linia verticală punctată marchează valoarea adevărată.

Poziția fiecărei histograme față de linia punctată arată deplasarea (*bias*) estimatorului, lățimea ei arată abaterea standard, iar forma ei arată dacă intervalele bazate pe aproximarea Normală sînt adecvate. Idealul este o histogramă simetrică, centrată pe linie; panoul din stînga se îndepărtează de acest ideal, cel din dreapta mult mai puțin.

## Interpretarea comparației estimatorilor {h=9e07da50}

Pentru AR(1) cu $\phi = 0{,}9$, mediile estimațiilor sînt 0,853 pentru Yule–Walker, 0,864 pentru cele mai mici pătrate condiționate și 0,866 pentru verosimilitatea maximă. Toți trei estimatorii sînt deplasați în jos. Aproximarea de manual a deplasării celor mai mici pătrate, $-(1 + 3\phi)/T = -0{,}037$, este apropiată de deplasarea estimațiilor simulate prin cele mai mici pătrate. Yule–Walker are cea mai mare deplasare aproape de cercul unitate.

Abaterile standard ale estimațiilor sînt 0,056, 0,055 și 0,054, toate peste valoarea asimptotică 0,044. Distribuția este asimetrică spre stînga: estimațiile nu pot depăși cu mult valoarea 1 fără să iasă din regiunea de staționaritate, dar pot coborî mult sub 0,9. Intervalele asimptotice sînt deci prea optimiste aproape de $\phi = 1$.

Pentru MA(1) cu $\theta = 0{,}5$, mediile sînt 0,504 pentru cele mai mici pătrate condiționate, cu abaterea standard 0,097, și 0,506 pentru verosimilitatea maximă, cu abaterea standard 0,098; abaterea standard asimptotică este 0,087. Aici ambii estimatori sînt practic nedeplasați și ambii variază mai mult decît spune formula asimptotică. Regula practică este să folosim verosimilitatea maximă, iar Yule–Walker doar pentru valori de pornire rapide sau în scop didactic.

## Recapitulare: estimarea {h=284ba399}

Yule–Walker este o metodă a momentelor pentru modelele AR pure, $\hat\phi = \hat R^{-1}\hat\rho$. Cele mai mici pătrate condiționate înseamnă OLS pentru un AR și o minimizare numerică pe șocuri calculate recursiv pentru un MA. Verosimilitatea maximă gaussiană exactă, calculată prin descompunerea erorilor de predicție @eq:loglik, este standardul din programe.

În eșantioane mici, coeficienții AR persistenți sînt deplasați în jos, iar distribuția lor de selecție este asimetrică, așa că erorile standard asimptotice subestimează incertitudinea aproape de cercul unitate.

## @end

Cei trei estimatori coincid asimptotic pentru modelele AR pure și diferă în eșantioane mici: cu $T = 100$ și $\phi = 0{,}9$ toți sînt deplasați în jos, Yule–Walker cel mai mult. Verosimilitatea maximă @eq:loglik tratează modelele MA și ARMA și este metoda implicită. Estimarea unui model nu spune dacă el este cel potrivit: [secțiunea următoare](#sec-6) compară modelele candidate și testează reziduurile lor.

# Selecția modelului și diagnosticarea

Corelogramele propun mai multe ordine candidate. Secțiunea alege între ele cu criterii informaționale (*information criteria*), care pun în balanță calitatea ajustării și numărul de parametri, și verifică modelul ales testînd dacă reziduurile lui se comportă ca un zgomot alb. Două experimente de simulare arată cît de des găsesc AIC și BIC ordinul adevărat și de ce testul Ljung–Box pentru reziduuri are nevoie de $m - p - q$ grade de libertate.

## Criterii informaționale {h=a1d73ade}

Calitatea ajustării unui model, măsurată prin log-verosimilitatea maximizată $\ln L$, crește cu fiecare parametru adăugat. Criteriile informaționale percep de aceea o penalizare pentru fiecare parametru. Fie $k$ numărul parametrilor estimați (coeficienții $\phi$ și $\theta$, $\mu$ și $\sigma^2$):

$$\begin{gathered}\mathrm{AIC} = -2\ln L + 2k, \\ \mathrm{AICc} = \mathrm{AIC} + \frac{2k(k + 1)}{T - k - 1}, \\ \mathrm{BIC} = -2\ln L + k\ln T.\end{gathered} $$ {#eq:ic}

Criteriul informațional Akaike (AIC, *Akaike information criterion*) provine de la @Akaike1974, corecția lui pentru eșantioane mici, AICc, de la @HurvichTsai1989, iar criteriul informațional bayesian (BIC) de la @Schwarz1978. Hirotugu Akaike, din fotografia de pe slide, a lucrat la Institutul de Matematică Statistică din Tokyo.

Cele două criterii diferă doar prin penalizarea pe parametru: 2 pentru AIC și $\ln T$ pentru BIC. Cum $\ln T > 2$ de îndată ce $T \ge 8$, BIC penalizează mai puternic în orice eșantion realist și preferă modele mai mici. Se alege modelul cu valoarea cea mai mică. Pot fi comparate doar modele estimate pe aceleași observații: dacă, de exemplu, modele AR de ordine diferite sînt estimate prin cele mai mici pătrate condiționate, fiecare renunță la un alt număr de observații inițiale, iar verosimilitățile lor nu mai sînt comparabile.

## Comparația criteriilor AIC și BIC {h=59403658}

BIC este consistent (*consistent*): dacă modelul adevărat se află printre candidați, probabilitatea ca BIC să îl aleagă tinde la 1 cînd $T$ crește. Prețul este că, în eșantioane mici, penalizarea lui mare poate alege un model prea mic, care omite un coeficient real, dar mic.

AIC este construit pentru prognoză: el estimează calitatea așteptată a ajustării în afara eșantionului (*out-of-sample*), măsurată prin distanța Kullback–Leibler dintre model și distribuția adevărată. Nu este consistent: chiar cu $T$ mare alege un model prea mare (*overfitting*) cu o probabilitate pozitivă, pentru că penalizarea 2 nu crește odată cu eșantionul.

În practică, așa cum arată @HyndmanAthanasopoulos2021 și @HyndmanKhandakar2008, căutările automate, de exemplu auto.arima în R sau pmdarima în Python, minimizează AICc pe o grilă de ordine $(p, q)$. Cînd AIC și BIC aleg modele diferite, se păstrează ambii candidați, iar diagnosticarea reziduurilor și prognozele în afara eșantionului decid între ei.

## Cît de des găsesc AIC și BIC ordinul adevărat {h=3f798340}

Experimentul simulează un AR(2) adevărat cu $\phi = (0{,}5; 0{,}25)$, estimează modele AR($p$) pentru $p = 0, \dots, 6$ prin OLS pe același eșantion și înregistrează ordinul ales de fiecare criteriu. Fiecare panou folosește 500 de eșantioane: în stînga de lungime $T = 100$, în dreapta de lungime $T = 1000$.

Barele dau procentul de eșantioane în care AIC (albastru) sau BIC (roșu) a ales fiecare ordin; linia verde punctată marchează ordinul adevărat $p = 2$. Masa de la $p = 2$ reprezintă alegeri corecte, cea din stînga un model prea mic, iar cea din dreapta un model prea mare.

## Interpretarea experimentului de selecție {h=0aec4158}

Cu $T = 100$, AIC găsește $p = 2$ în 54,4% dintre eșantioane, iar BIC în 47,4%. BIC alege un model prea mic în 50,4% dintre eșantioane: al doilea coeficient, $\phi_2 = 0{,}25$, este mic în raport cu eroarea lui de selecție la 100 de observații, iar penalizarea BIC depășește adesea contribuția lui. AIC alege un model prea mare în 24,8% dintre eșantioane.

Cu $T = 1000$, BIC găsește ordinul adevărat în 98,6% dintre eșantioane, iar AIC în 74,4%. Aceasta este consistența la lucru: BIC converge către modelul adevărat, în timp ce AIC alege încă un model prea mare în 25,6% dintre eșantioane.

Niciun criteriu nu dovedește ordinul. Cu date macroeconomice trimestriale, $T$ este în jur de 100, iar ordinul rămîne incert. Cele două erori au costuri diferite: un model prea mare adaugă puțin zgomot de estimare în prognoze, în timp ce un model prea mic lasă autocorelație în reziduuri, pe care diagnosticarea din cadrul următor este făcută să o detecteze.

## Diagnosticarea reziduurilor {h=48886fc8}

Dacă modelul este corect specificat, reziduurile lui $\hat\varepsilon_t$ se comportă ca un zgomot alb. Prima verificare este vizuală: graficul reziduurilor nu trebuie să arate tipare, schimbări de varianță sau valori extreme izolate. ACF a reziduurilor trebuie să rămînă în banda $\pm 1{,}96/\sqrt{T}$, cu excepția a aproximativ un lag din douăzeci.

Testul portmanteau al lui @LjungBox1978 combină primele $m$ autocorelații ale reziduurilor:

$$\begin{gathered}Q^*(m) = T(T + 2)\sum_{h=1}^{m}\frac{\hat\rho_{\hat\varepsilon}(h)^2}{T - h}, \\ Q^*(m) \approx \chi^2(m - p - q) \quad \text{în ipoteza } H_0.\end{gathered} $$ {#eq:lb}

Ipoteza nulă este că modelul este corect. Estimarea celor $p + q$ coeficienți consumă $p + q$ grade de libertate [@BoxPierce1970]. Numărul de laguri $m$ trebuie să fie mult peste $p + q$: $m = 10$ pentru date anuale sau trimestriale și $m = 2s$ pentru date sezoniere cu perioada $s$ [@HyndmanAthanasopoulos2021].

Normalitatea se testează cu statistica Jarque–Bera [@JarqueBera1987], $\mathrm{JB} = \frac{T}{6}\big(S^2 + (K - 3)^2/4\big) \approx \chi^2(2)$, unde $S$ este asimetria (*skewness*), iar $K$ kurtosis-ul reziduurilor (0 și 3 pentru distribuția Normală), împreună cu un grafic cuantilă–cuantilă (QQ, *quantile–quantile*). Normalitatea se respinge la 5% dacă $\mathrm{JB} > 5{,}99$, cuantila de 95% a lui $\chi^2(2)$. Reziduurile care nu urmează distribuția Normală nu deplasează $\hat\phi$, dar fac greșite intervalele de prognoză construite cu distribuția Normală. În sfîrșit, un test Ljung–Box pentru pătratele reziduurilor $\hat\varepsilon_t^2$ detectează volatility clustering, subiectul Capitolului 5.

## Justificarea celor m - p - q grade de libertate {h=ceb95dc3}

Experimentul aplică testul Ljung–Box la nivelul de 5% reziduurilor modelului adevărat, cu $T = 200$ și $m = 10$: 1 000 de eșantioane dintr-un AR(1) și 500 de eșantioane dintr-un ARMA(1,1). Pentru fiecare model, graficul arată rata de respingere cu $m = 10$ grade de libertate (portocaliu) și cu $m - p - q$ grade de libertate (albastru); linia roșie punctată marchează nivelul nominal de 5%.

Cum modelul estimat este chiar cel adevărat, ipoteza nulă este adevărată în fiecare eșantion. Un test cu mărimea corectă respinge în 5% dintre eșantioane, așa că o alegere bună a gradelor de libertate dă o bară care atinge linia punctată.

## Interpretarea experimentului cu gradele de libertate {h=ab51e510}

Cu $m = 10$ grade de libertate testul respinge prea rar: în 3,2% dintre eșantioane pentru AR(1) și în 1,4% pentru ARMA(1,1), în loc de 5%. Reziduurile sînt ajustate: estimarea alege coeficienții astfel încît reziduurile să pară necorelate la primele laguri. Autocorelațiile lor sînt deci mai mici decît ale unui zgomot alb adevărat, iar $Q^*$ este mai mic decît sugerează referința $\chi^2(m)$.

Cu $m - p - q$ grade de libertate, ratele de respingere sînt 4,8% și 3,4%, apropiate de nivelul nominal de 5%. Corecția contează mai ales cînd $p + q$ este mare în raport cu $m$: ea este mai mare pentru ARMA(1,1), cu doi coeficienți, decît pentru AR(1).

Funcția acorr_ljungbox din statsmodels folosește implicit $m$ grade de libertate; corecția cere argumentul model_df = p + q. Cu grade de libertate greșite, p-value-urile sînt prea mari, iar un model prost pare acceptabil.

## Recapitulare: selecția modelului și diagnosticarea {h=ad28c3e7}

Criteriile @eq:ic sînt $\mathrm{AIC} = -2\ln L + 2k$ și $\mathrm{BIC} = -2\ln L + k\ln T$; cîștigă valoarea cea mai mică, iar BIC alege modele mai mici. BIC este consistent; AIC urmărește prognoza și uneori alege modele prea mari.

Reziduurile se verifică prin grafic, ACF lor, testul Ljung–Box @eq:lb cu $m - p - q$ grade de libertate, testul Jarque–Bera și testul Ljung–Box pentru pătratele reziduurilor. Un model trece verificarea cînd reziduurile lui sînt zgomot alb; normalitatea contează pentru intervale.

## @end

Selecția și verificarea sînt pași separați. Criteriile @eq:ic ordonează candidații, BIC fiind consistent, iar AIC orientat spre prognoză; cu aproximativ 100 de observații, ambele ratează adesea ordinul adevărat. Diagnosticarea decide dacă modelul ales este adecvat, iar testul Ljung–Box @eq:lb trebuie să folosească $m - p - q$ grade de libertate pentru a avea mărimea corectă. [Secțiunea următoare](#sec-7) folosește un model adecvat pentru prognoză.

# Prognoza cu modele ARMA

Scopul celor mai multe modele ARMA este prognoza. Secțiunea derivă prognoza punctuală optimă, varianța erorii ei și intervalele de prognoză și arată proprietatea comună tuturor modelelor staționare: prognozele lor revin la medie.

## Prognoza optimă {h=6a96fd33}

Prognoza lui $X_{T+h}$ făcută la momentul $T$ este media condiționată de trecut, $\hat X_{T+h|T} = E[X_{T+h} \mid X_T, X_{T-1}, \dots]$. Dintre toate funcțiile de trecut, ea minimizează eroarea medie pătratică (MSE, *mean squared error*) $E[(X_{T+h} - \hat X_{T+h|T})^2]$.

Pentru un model ARMA, media condiționată rezultă dintr-o rețetă simplă. Scriem modelul la momentul $T + h$ și înlocuim șocurile viitoare $\varepsilon_{T+j}$, $j \ge 1$, cu 0, media lor condiționată, iar valorile viitoare $X_{T+j}$ cu prognozele lor; valorile trecute și reziduurile trecute rămîn neschimbate. Prognozele se calculează recursiv pentru $h = 1, 2, \dots$.

Pentru un AR(1), rețeta dă

$$\hat X_{T+h|T} = \mu + \phi^h(X_T - \mu). $$ {#eq:ar1-fc}

Abaterea de la medie scade cu factorul $\phi$ pe perioadă: aceasta este revenirea la medie (*mean reversion*) cu viteza $\phi$. Pentru un MA($q$), fiecare termen al modelului la $T + h$, cu $h > q$, este un șoc viitor, deci $\hat X_{T+h|T} = \mu$: după $q$ pași modelul nu mai știe nimic în afară de medie.

## Erorile de prognoză și intervalele {h=9b775789}

Eroarea de prognoză rezultă din forma cauzală $X_{T+h} = \mu + \sum_j\psi_j\varepsilon_{T+h-j}$. Prognoza păstrează termenii cu șocurile pînă la momentul $T$, așa că eroarea este suma șocurilor viitoare, ponderate cu primele ponderi $\psi$:

$$\begin{gathered}e_{T+h} = \varepsilon_{T+h} + \psi_1\varepsilon_{T+h-1} + \dots + \psi_{h-1}\varepsilon_{T+1}, \\ \sigma_h^2 = \sigma^2(1 + \psi_1^2 + \dots + \psi_{h-1}^2).\end{gathered} $$ {#eq:fvar}

Varianța erorii $\sigma_h^2$ crește cu orizontul $h$.

Cu șocuri gaussiene, intervalul de prognoză de 95% este $\hat X_{T+h|T} \pm 1{,}96\,\sigma_h$. La $h = 1$ el este $\pm 1{,}96\,\sigma$. Cînd $h \to \infty$, $\sigma_h^2 \to \sigma^2\sum_j\psi_j^2 = \gamma(0)$, varianța necondiționată: pe termen lung, intervalul de prognoză este banda distribuției necondiționate.

Intervalul lasă deoparte trei surse de incertitudine: parametrii sînt estimați ($\hat\phi$ în loc de $\phi$), modelul însuși poate fi greșit, iar șocurile pot avea cozi mai groase decît distribuția Normală. Acoperirea efectivă a intervalelor nominale de 95% este de aceea, de obicei, sub 95%. În plus, erorile prognozelor făcute în același moment pentru orizonturile $h$ și $h + 1$ sînt corelate, pentru că au în comun șocurile $\varepsilon_{T+1}, \dots, \varepsilon_{T+h}$: o traiectorie de prognoză prea ridicată la un orizont tinde să fie prea ridicată și la următorul.

## Exemplu rezolvat: prognoza unui AR(1) {h=98853127}

Revenim la $X_t = 2 + 0{,}8X_{t-1} + \varepsilon_t$, cu $\sigma^2 = 1$ și media $\mu = 10$, și presupunem că ultima valoare observată este $X_T = 12$, cu două unități peste medie. Din @eq:ar1-fc, prognozele punctuale sînt $\hat X_{T+1} = 10 + 0{,}8 \cdot 2 = 11{,}60$, $\hat X_{T+2} = 10 + 0{,}64 \cdot 2 = 11{,}28$ și $\hat X_{T+10} = 10{,}21$.

Varianțele erorilor rezultă din @eq:fvar cu $\psi_1 = 0{,}8$: $\sigma_1^2 = 1$ și $\sigma_2^2 = 1 + 0{,}8^2 = 1{,}64$. Intervalele de 95% sînt $[9{,}64; 13{,}56]$ la $h = 1$ și $[8{,}77; 13{,}79]$ la $h = 2$: intervalul se lărgește, iar centrul lui se apropie de medie.

Pe termen lung, prognoza tinde spre media 10, iar abaterea standard a erorii de prognoză tinde spre $\sqrt{\gamma(0)} = 1{,}67$, abaterea standard necondiționată calculată în primul exemplu rezolvat.

## Revenirea la medie în prognoze {h=df384392}

Graficul compară prognozele a trei modele pe orizonturile de la 0 la 20: un AR(1) cu $\phi = 0{,}9$, un AR(1) cu $\phi = 0{,}5$ și un MA(1) cu $\theta = 0{,}6$ și ultimul șoc $\varepsilon_T = 2$. În toate trei, ultima valoare este $X_T = 3$, media este $\mu = 0$, iar abaterea standard a șocului este $\sigma = 1$.

Linia continuă este prognoza punctuală, zona colorată intervalul de 95%, iar liniile roșii punctate banda necondiționată $\mu \pm 1{,}96\,\sigma_X$, unde $\sigma_X$ este abaterea standard a procesului. Distanța prognozei față de 0 măsoară informația pe care prezentul o mai poartă despre viitor; diferența dintre intervalul colorat și banda punctată măsoară cît reduce prognoza incertitudinea față de situația în care am cunoaște doar distribuția necondiționată.

## Interpretarea traiectoriilor de prognoză {h=ffc6ab38}

Cu $\phi = 0{,}9$ revenirea este lentă: după 5 perioade prognoza este încă 1,77, cu o eroare standard de 1,85. Intervalul se apropie de lățimea lui de termen lung, $\pm 1{,}96 \times 2{,}29$, abia după multe perioade.

Cu $\phi = 0{,}5$, prognoza după 5 perioade este 0,094, practic media. MA(1) are un singur pas informativ: prognoza la $h = 1$ este $\theta\varepsilon_T = 1{,}2$, iar de la $h = 2$ încolo este media 0; intervalul sare la lățimea de termen lung la $h = 2$.

Un model staționar prognozează întotdeauna revenirea la medie. Acest lucru este util pentru seriile care chiar revin, cum sînt spread-urile și ratele de creștere, și înșelător cînd media însăși se schimbă: modelul trage atunci prognozele spre un nivel care nu mai este valabil.

## Recapitulare: prognoza {h=96da6c83}

Prognoza optimă $\hat X_{T+h|T}$ înlocuiește șocurile viitoare cu 0 și valorile viitoare cu prognozele lor. Varianța erorii este $\sigma^2\sum_{j<h}\psi_j^2$, așa că intervalele se lărgesc cu orizontul pînă la $\pm 1{,}96\sqrt{\gamma(0)}$.

Prognozele revin la $\mu$; pentru un MA($q$) ele sînt egale cu $\mu$ după $q$ pași. Intervalele construite cu distribuția Normală ignoră incertitudinea parametrilor și a modelului și sînt de aceea prea înguste în practică.

## @end

Prognoza unui model ARMA staționar este o revenire la medie cu viteza dată de ponderile $\psi$, @eq:ar1-fc pentru AR(1); varianța erorii @eq:fvar cumulează aceleași ponderi și tinde spre $\gamma(0)$. [Secțiunea următoare](#sec-8) reunește identificarea, estimarea, verificarea și prognoza într-o singură metodă.

# Metoda Box–Jenkins

Secțiunile anterioare furnizează piesele: identificarea din corelograme, estimarea, selecția, diagnosticarea și prognoza. Box și Jenkins le-au reunit într-o metodă iterativă, pe care studiul de caz din [secțiunea 9](#sec-9) o aplică apoi pas cu pas.

## George Box și Gwilym Jenkins {h=ad094794}

George E. P. Box (1919–2013) a fost statistician la University of Wisconsin–Madison. Este cunoscut pentru fraza „Toate modelele sînt greșite, dar unele sînt utile” (*All models are wrong, but some are useful*). Gwilym M. Jenkins (1932–1982) a fost statistician la Lancaster University. Cartea lor *Time Series Analysis: Forecasting and Control* a apărut în 1970; a patra ediție, împreună cu Gregory Reinsel, este @BoxJenkinsReinsel2008.

Contribuția lor nu a constat în modele noi: autoregresiile provin de la @Yule1927, iar mediile mobile de la @Slutzky1937. Contribuția a fost o metodă: o buclă de identificare, estimare și verificare pe care orice analist o poate aplica unei serii. Testele pentru reziduuri ale lui @BoxPierce1970 și @LjungBox1978 provin din acest program.

Fraza despre modelele greșite rezumă atitudinea metodei. Un model nu este judecat după faptul că ar fi adevărat, lucru pe care niciun eșantion finit nu îl poate stabili, ci după faptul că reziduurile lui sînt zgomot alb și prognozele lui sînt utile.

## Bucla Box–Jenkins {h=75f925ca}

Diagrama arată metoda sub forma unei bucle. Pasul 0 pregătește seria: o reprezentăm grafic, o transformăm (de exemplu, logaritmăm) și o facem staționară. Pasul 1 identifică ordinele candidate $(p, q)$ din ACF și PACF. Pasul 2 estimează candidații prin verosimilitate maximă și îi compară cu AIC și BIC. Pasul 3 verifică modelul ales: graficele reziduurilor, testul Ljung–Box și rădăcinile ambelor polinoame. Dacă modelul trece verificarea, pasul 4 produce prognozele punctuale și intervalele; dacă nu, analistul se întoarce la identificare.

Pasul 0 ține de Capitolele 1 și 3: logaritmi, diferențe și teste de rădăcină unitară. Modelele ARMA cer o serie staționară, iar estimarea lor pe o serie cu rădăcină unitară produce estimări aproape de granița regiunii de staționaritate și prognoze înșelătoare.

Bucla se încheie cu cel mai mic model ale cărui reziduuri sînt zgomot alb și ale cărui rădăcini sînt bine în interiorul regiunii admisibile. Verificările în afara eșantionului, cu setul de antrenare și setul de test din Capitolul 0, completează metoda: un model care trece toate verificările în eșantion trebuie totuși să prognozeze mai bine decît reperele simple.

## @end

Metoda Box–Jenkins transformă rezultatele capitolului într-o procedură: pregătim seria, identificăm, estimăm, verificăm și ne întoarcem la identificare pînă cînd reziduurile sînt zgomot alb, apoi prognozăm și comparăm în afara eșantionului cu repere simple. [Secțiunea următoare](#sec-9) o aplică pentru creșterea PIB-ului României.

# Studiu de caz: creșterea PIB-ului României

Studiul de caz aplică bucla Box–Jenkins creșterii anuale a PIB-ului real al României. Fiecare pas al metodei are cadrul lui: datele, identificarea din corelograme, estimarea și criteriile informaționale, diagnosticarea modelului ales și prognozele. Structura datelor explică ea însăși modelul pe care îl aleg criteriile.

## Datele {h=d7999082}

Seria este PIB-ul real al României în volume înlănțuite (*chain-linked volumes*, prețurile din 2010), trimestrial și neajustat sezonier, publicat de Eurostat; sursa națională este Institutul Național de Statistică (INS). Volumele înlănțuite elimină efectul modificărilor de preț, așa că ratele de creștere măsoară modificări ale producției reale.

Analiza folosește rata anuală de creștere $y_t = 100\,(\ln Y_t - \ln Y_{t-4})$, în %. Comparînd fiecare trimestru cu același trimestru din anul anterior, eliminăm tiparul sezonier, așa cum a arătat Capitolul 1. Eșantionul acoperă T1 2000–T2 2026, $T = 106$ trimestre, cu media 3,29% și abaterea standard 4,35%. Valorile extreme sînt $-9{,}9\%$ în T2 2020 și $+12{,}5\%$ în T4 2004, iar ultima valoare este $-2{,}0\%$.

Rata de creștere nu are trend, ceea ce susține tratarea ei ca serie staționară; testul formal este în Capitolul 3. Fotografia de pe slide arată clădirea Institutului Național de Statistică din București.

## Pasul 1: identificarea {h=c2e8c4e4}

Panoul de sus arată rata anuală de creștere, cu media ei de 3,29% (linie punctată). Se disting mișcările mari din eșantion: recesiunea din 2009–2010, căderea la $-9{,}9\%$ din T2 2020, în timpul pandemiei, revenirea din 2021 și încetinirea de la sfîrșitul eșantionului, cu valori negative în ultimele trimestre.

Panourile de jos arată ACF și PACF de selecție pentru lagurile de la 1 la 16 trimestre, cu banda $\pm 1{,}96/\sqrt{T} = \pm 0{,}19$. Valorile din afara benzii sînt cele pe care această verificare le separă de zgomot; ele se citesc cu ajutorul tabelului de identificare din [secțiunea 4](#sec-4).

## Interpretarea corelogramelor creșterii PIB {h=5a8ca766}

ACF are valorile 0,62, 0,40 și 0,21 la lagurile 1–3 și $-0{,}01$ la lagul 4: se anulează după lagul 3. Conform tabelului de identificare, candidatul este un MA(3). PACF este 0,62 la lagul 1 și mică după aceea, cu excepția valorii 0,32 de la lagul 5. Citită singură, ea indică un AR(1); valoarea de la lagul 5 poate fi întîmplătoare sau un rest al tiparului sezonier.

Modelul MA(3) are o explicație structurală. Rata anuală de creștere este aproximativ suma a patru rate trimestriale de creștere, $y_t \approx g_t + g_{t-1} + g_{t-2} + g_{t-3}$, unde $g_t = 100\,(\ln Y_t - \ln Y_{t-1})$. Dacă ratele trimestriale $g_t$ sînt apropiate de un zgomot alb (Seminarul 2, sarcina B2), două rate anuale aflate la un trimestru distanță au în comun trei dintre cei patru termeni trimestriali, cele aflate la două trimestre au în comun doi, cele aflate la trei trimestre unul, iar cele aflate la patru trimestre niciunul. Rezultatul este un MA(3) prin construcție.

Înainte de estimarea oricărui model, statistica Ljung–Box pentru seria însăși este $Q^*(8) = 64{,}9$. Seria este puternic autocorelată, deci este nevoie de un model: media singură ar lăsa structură predictibilă în erori.

## Pasul 2: estimarea și criteriile informaționale {h=01d1fb81}

Tabelul compară șapte modele, toate estimate prin verosimilitate maximă gaussiană exactă pe aceleași 106 trimestre. Pentru fiecare, el dă numărul de parametri $k$, log-verosimilitatea maximizată $\ln L$, AIC, BIC, abaterea standard a reziduurilor $\hat\sigma$ și p-value-ul testului Ljung–Box $Q^*(8)$ pentru reziduuri, cu $8 - p - q$ grade de libertate. Căutarea completă acoperă toate perechile $p, q \le 3$.

Modelul care conține doar media, ARMA(0,0), are $\ln L = -305{,}83$ și un p-value Ljung–Box sub 0,001. AR(1) ridică $\ln L$ la $-279{,}67$ și reduce $\hat\sigma$ de la 4,33 la 3,38, dar reziduurile lui rămîn autocorelate, cu un p-value de 0,023. Adăugarea unui al doilea parametru, în AR(2) sau în ARMA(1,1), lasă $\ln L$ la $-279{,}66$: parametrul suplimentar este inutil, iar ambele teste pentru reziduuri resping în continuare la 5%.

MA(3) ajunge la $\ln L = -270{,}02$, AIC 550,04, BIC 563,36, $\hat\sigma = 3{,}07$ și un p-value Ljung–Box de 0,375. ARMA(1,3) are cel mai mic AIC, 548,97, mai mic cu doar 1,07, dar un BIC mai mare, 564,95. BIC alege deci MA(3), iar AIC alege ARMA(1,3); ARMA(2,3) nu este mai bun după niciun criteriu. AR(1), candidatul sugerat de PACF, este respins de diagnosticare.

## Interpretarea modelului MA(3) estimat {h=a78857e1}

Modelul estimat este $\hat y_t = 3{,}17 + \hat\varepsilon_t + 0{,}834\,\hat\varepsilon_{t-1} + 0{,}675\,\hat\varepsilon_{t-2} + 0{,}591\,\hat\varepsilon_{t-3}$, cu $\hat\sigma = 3{,}07$. Erorile standard OPG sînt 1,03 pentru medie și 0,083, 0,102 și 0,082 pentru cei trei coeficienți MA, deci toți cei trei termeni MA sînt semnificativi.

Coeficienții scad de la 0,834 la 0,591. Ei sînt aproape de tiparul $(1, 1, 1)$, dar sub el; acest tipar ar rezulta din suprapunerea a patru trimestre dacă creșterea trimestrială ar fi exact zgomot alb. El ar plasa rădăcinile lui $\theta(z) = 1 + z + z^2 + z^3$ pe cercul unitate; pentru modelul estimat, cel mai mic modul al rădăcinilor este 1,17, peste 1, deci modelul este invertibil.

În ARMA(1,3), primul coeficient MA este $\hat\theta_1 = 1{,}019 > 1$, și totuși toate rădăcinile lui $\theta(z)$ au modulul de cel puțin 1,10, așa că și acest model este invertibil: pentru $q > 1$, invertibilitatea se citește din rădăcini, nu din coeficienți luați separat. Coeficientul lui AR este $\hat\phi = -0{,}284$, cu eroarea standard 0,143: un parametru în plus pentru un cîștig mic, pe care BIC îl elimină. Creșterea medie este de 3,17% pe an, dar cu o eroare standard de 1,03: observațiile care se suprapun conțin mai puțină informație decît sugerează cele 106 trimestre.

## Pasul 3: diagnosticarea modelului MA(3) {h=c84471b4}

Graficul are patru panouri. Cel din stînga sus arată reziduurile modelului MA(3) în timp, iar cel din dreapta sus ACF lor, cu banda $\pm 1{,}96/\sqrt{T}$. Cel din stînga jos dă p-value-urile testului Ljung–Box pentru $m = 4, \dots, 16$, fiecare cu $m - 3$ grade de libertate, față de nivelul de 5% (linia punctată). Cel din dreapta jos este un grafic QQ Normal al reziduurilor standardizate.

Într-un grafic QQ, cuantilele de selecție ale reziduurilor sînt reprezentate în funcție de cuantilele distribuției Normale; dacă reziduurile urmează distribuția Normală, punctele se află pe linia roșie. Punctele de sub linie din colțul din stînga jos sînt reziduuri negative mai extreme decît permite distribuția Normală, adică o coadă stîngă groasă.

## Interpretarea diagnosticării MA(3) {h=b9018c33}

Reziduurile nu au autocorelație. Statistica Ljung–Box este $Q^*(8) = 5{,}35$ cu 5 grade de libertate, cu un p-value de 0,375, iar testul pentru $m = 16$ are un p-value de 0,525. Toate p-value-urile din grafic sînt peste 0,12. Cu 8 în loc de 5 grade de libertate, p-value-ul lui $Q^*(8)$ ar fi 0,720, un rezultat care pare mai liniștitor decît justifică datele.

Reziduurile nu urmează distribuția Normală: statistica Jarque–Bera este 39,0, cu un p-value sub 0,001, asimetria este $-0{,}75$, iar kurtosis-ul 5,61. Cel mai mare reziduu este $-11{,}4$ în T2 2020, adică $-3{,}89$ abateri standard: perioada de lockdown din pandemie, un șoc pe care niciun model construit pe PIB-ul trecut nu l-ar fi putut anticipa.

Testul Ljung–Box pentru pătratele reziduurilor are un p-value de 0,779, deci nu există volatility clustering la această frecvență. Verdictul este că MA(3) surprinde dependența din serie, dar intervalele construite cu distribuția Normală subestimează riscul recesiunilor.

## Pasul 4: prognozele {h=48f29db3}

Graficul arată rata de creștere observată din 2016 și prognozele pentru T3 2026–T2 2028 din două modele: MA(3), ales de BIC (roșu), și AR(1) (albastru), fiecare cu intervalul de 95%.

Ambele prognoze pornesc puțin sub zero, aproape de ultimele valori observate, și urcă spre medie. MA(3) ajunge la medie după trei trimestre și rămîne apoi constant; AR(1) se apropie treptat de media lui. Cele două intervale colorate sînt largi în comparație cu mișcările seriei din ultimii ani și se suprapun aproape complet.

## Interpretarea prognozelor PIB {h=87d0debb}

MA(3) prognozează $-0{,}14\%$, $1{,}81\%$ și $2{,}77\%$ pentru următoarele trei trimestre și apoi media, 3,17%, începînd cu al patrulea trimestru. Aceasta este exact regula pentru MA($q$): după $q = 3$ pași modelul cunoaște doar media.

AR(1) revine lin la media lui și ajunge la 2,99% după 8 trimestre, față de o medie de 3,12%. Cele două modele sînt de acord pe termen scurt și diferă prin viteza cu care uită încetinirea recentă.

Intervalele de 95% ale modelului MA(3) sînt $[-6{,}1; 5{,}9]$ pentru un trimestru înainte și $[-6{,}3; 12{,}7]$ începînd cu al patrulea trimestru, adică $\pm 6{,}0$ și $\pm 9{,}5$ puncte procentuale. Deoarece reziduurile au cozi groase, aceste intervale subestimează probabilitatea unei recesiuni adînci. Un model univariat nu poate anticipa un punct de întoarcere; el arată cît de neobișnuită ar fi următoarea valoare, dat fiind trecutul seriei.

## Recapitulare: creșterea PIB-ului României {h=586c9bd8}

Pentru creșterea anuală a PIB-ului trimestrial, ACF se anulează după lagul 3; BIC alege MA(3), iar AIC alege ARMA(1,3). MA(3) reflectă suprapunerea a patru trimestre în rata anuală de creștere: structura datelor explică modelul.

Reziduurile sînt zgomot alb, dar au cozi groase din cauza anului 2020, așa că intervalele construite cu distribuția Normală sînt prea înguste în crize. Prognozele ajung la medie după trei trimestre.

## @end

Bucla Box–Jenkins aplicată creșterii PIB-ului se încheie cu un MA(3): corelogramele îl sugerează, BIC îl alege, testele Ljung–Box cu $m - 3$ grade de libertate îl acceptă, iar construcția ratelor anuale de creștere îl explică. Diagnosticarea arată și limita lui, reziduurile cu cozi groase, care fac intervalele construite cu distribuția Normală prea înguste în recesiuni. [Secțiunea următoare](#sec-10) aplică aceleași instrumente unor serii de alte tipuri.

# Alte serii reale

Metoda se aplică aici la trei serii cu caractere diferite: inflația lunară, foarte persistentă; randamentele zilnice ale indicelui BET și ale cursului EUR/RON, aproape imprevizibile în medie; și numerele istorice de pete solare ale lui Yule. Un ultim studiu de caz rezumă ce spun competițiile de prognoză despre modelele ARMA.

## Inflația în România: o prognoză AR(2) {h=ee46560c}

Seria este rata pe 12 luni a indicelui armonizat al prețurilor de consum (HICP, *harmonised index of consumer prices*) pentru România, publicat de Eurostat, în %: 260 de luni pînă în august 2026. Ordinul AR este ales de BIC dintre $p \le 6$.

Graficul arată rata inflației (roșu), prognoza AR(2) cu intervalele de 80% și 95% (zone colorate), media estimată de 5,16% (linie portocalie punctată) și ținta de inflație a Băncii Naționale a României (BNR), 2,5% ± 1 punct procentual (banda verde). Eșantionul include perioade de inflație negativă în jurul anilor 2015–2016 și vîrful din 2022–2023.

## Interpretarea modelului pentru inflație {h=eca27e40}

BIC este 495,7 pentru AR(1), 469,7 pentru AR(2) și 473,4 pentru AR(3), deci se alege AR(2). Coeficienții lui sînt $\hat\phi_1 = 1{,}320$ (eroarea standard 0,057) și $\hat\phi_2 = -0{,}344$ (eroarea standard 0,055). Suma lor este 0,976, iar cea mai mare rădăcină inversă are modulul 0,963.

Modelul este foarte aproape de o rădăcină unitară, ceea ce ridică întrebarea dacă inflația este staționară; testul este în Capitolul 3. Media estimată, 5,16%, cu eroarea standard 1,35, este media perioadei 2005–2026 și se află peste ținta BNR de 2,5% ± 1 punct procentual. Prognoza merge de la ultima valoare, 6,09%, la 5,44% luna următoare și la 5,13% în august 2028, cu un interval de 95% de $[-0{,}6; 10{,}9]$ la acest orizont. Modelul revine la media istorică, nu la țintă: el nu conține nicio informație despre politica monetară.

Diagnosticarea eșuează: statistica Ljung–Box este $Q^*(12) = 71{,}6$, cu un p-value sub 0,001. Reziduurile sînt corelate la lagul 12, o consecință a efectelor de bază (*base effects*): un șoc de preț intră în rata pe 12 luni cînd se produce și iese din ea douăsprezece luni mai tîrziu. Modelarea acestei structuri sezoniere este o sarcină pentru Capitolul 4.

## Randamente zilnice: BET și EUR/RON {h=6c5b483f}

Panourile din stînga arată ACF a randamentelor logaritmice zilnice ale indicelui BET al Bursei de Valori București și ale cursului EUR/RON, pentru laguri de la 1 la 20 de zile. Panourile din dreapta arată ACF a pătratelor reziduurilor unui AR(1) estimat pentru fiecare serie. Banda este $\pm 1{,}96/\sqrt{T}$, foarte îngustă, pentru că ambele eșantioane au mii de observații.

În randamente, lagul 1 este clar pozitiv, iar celelalte laguri sînt mici, deși mai multe ies din banda îngustă. În pătratele reziduurilor toate lagurile sînt pozitive și mult în afara benzii: randamentele înseși sînt greu de prevăzut, dar mărimea lor nu.

## Interpretarea modelului AR(1) pentru randamentele zilnice {h=be7b8802}

Tabelul prezintă modelul AR(1) pentru ambele serii. Pentru BET (6 670 de observații), $\hat\phi = 0{,}109$, cu eroarea standard 0,005, raportul $t$ 20,9 și un $R^2$ de 1,2%. Pentru EUR/RON (5 352 de observații), $\hat\phi = 0{,}170$, cu eroarea standard 0,006, raportul $t$ 30,6 și un $R^2$ de 2,9%. Acestea sînt erori standard OPG. Cu un kurtosis al reziduurilor de 14 și 17, ele sînt prea mici: erorile standard din curbura verosimilității sau cele robuste sînt de cîteva ori mai mari, dar ambii coeficienți rămîn semnificativi. Coeficienții sînt foarte semnificativi, dar explică doar 1–3% din varianță: efectul este real din punct de vedere statistic și mic din punct de vedere economic, odată ce se iau în calcul costurile de tranzacționare.

Statisticile Ljung–Box $Q^*(10)$ pentru reziduuri, cu 9 grade de libertate, sînt 29,7 și 61,5, ambele cu p-value sub 0,001, deci AR(1) nu surprinde toată dependența din medie; BIC pe $p, q \le 2$ alege ARMA(1,0) pentru BET și ARMA(2,1) pentru EUR/RON. Valoarea pozitivă a lui $\hat\phi$ are explicații economice: ajustarea lentă a prețurilor pentru BET și un curs de schimb în regim de managed float pentru EUR/RON.

Statisticile Ljung–Box pentru pătratele reziduurilor sînt 2496 pentru BET și 2351 pentru EUR/RON, iar kurtosis-ul reziduurilor este 14,0 și 17,0. Media este modelată, varianța nu: mișcările mari se grupează în timp, iar combinația dintre modelele ARMA și GARCH din Capitolul 5 tratează acest aspect.

## Studiu de caz: Yule (1927) și petele solare {h=58717619}

@Yule1927 a estimat prima autoregresie, pentru numerele anuale de pete solare ale lui Wolfer din perioada 1749–1924. Pînă atunci, ciclurile din date erau modelate ca sume de sinusoide, analizate cu periodograma. Yule a propus un mecanism diferit, un „pendul bombardat cu mazăre”: un oscilator amortizat, lovit de șocuri aleatoare. Ciclul provine din dinamica sistemului, iar neregularitatea lui din șocuri.

Un AR(2) estimat prin cele mai mici pătrate pe aceiași ani, 176 de observații, dă $\hat\phi_1 = 1{,}336$ și $\hat\phi_2 = -0{,}650$. Rădăcinile sînt complexe, iar modulul rădăcinilor inverse este 0,806. Perioada implicită, $2\pi/\arccos\big(\hat\phi_1/(2\sqrt{-\hat\phi_2})\big)$, este de 10,57 ani: doi coeficienți regăsesc ciclul solar.

@Walker1931 a generalizat ecuațiile la AR($p$), iar de atunci ele se numesc ecuațiile Yule–Walker. Aceste două articole marchează începutul analizei parametrice a seriilor de timp.

## Petele solare: AR(2) și criteriile informaționale {h=54843ee8}

Panoul din stînga arată numerele anuale de pete solare pentru 1700–2008, 309 ani, împreună cu predicțiile cu un pas ale unui AR(2) estimat prin verosimilitate maximă pe întregul eșantion; zona colorată marchează eșantionul lui Yule, 1749–1924. Panoul din dreapta arată AIC și BIC pentru modele AR($p$) cu $p = 1, \dots, 10$, estimate pe întregul eșantion.

Predicțiile urmăresc îndeaproape ciclurile observate. În panoul din dreapta, ambele criterii scad brusc de la $p = 1$ la $p = 2$, rămîn aproximativ constante pentru cîteva ordine și ating minimul la $p = 9$.

## Interpretarea modelelor pentru petele solare {h=1dc3315b}

AR(2) estimat pe întregul eșantion are $\hat\phi = (1{,}391; -0{,}689)$, aproape de estimațiile lui Yule pe eșantionul mai scurt. El explică 83% din varianța numerelor de pete solare cu un an înainte. Predicțiile cu un pas urmăresc ciclul, dar rămîn în urmă la vîrfuri: modelul reacționează la un vîrf abia după ce l-a observat.

AIC și BIC aleg amîndouă $p = 9$ dintre ordinele pînă la 10. Lagurile suplimentare surprind o trăsătură pe care AR(2) nu o poate reproduce: ciclul solar este asimetric, cu o creștere rapidă și o scădere lentă, în timp ce un AR(2) liniar generează oscilații simetrice.

Lecția depășește cazul petelor solare. Doi parametri surprind esența seriei, ciclul ei; mai multe laguri surprind detaliile; iar trăsăturile neliniare, cum este asimetria ciclului, sînt dincolo de modelele ARMA.

## Studiu de caz: competițiile de prognoză {h=af6cac2c}

@MakridakisHibon2000 prezintă competiția M3: 3 003 serii (anuale, trimestriale, lunare și altele), prognozate cu 24 de metode. Prognozele au fost comparate în afara eșantionului, cu mai multe măsuri de acuratețe și pe mai multe orizonturi.

Competiția a confirmat concluziile competițiilor M anterioare. Metodele sofisticate din punct de vedere statistic nu prognozează neapărat mai bine decît cele simple. Clasamentul metodelor depinde de măsura de acuratețe și de orizont, iar combinațiile de metode au rezultate bune.

Pentru modelele ARMA, consecința este că bucla Box–Jenkins trebuie să se încheie cu o comparație în afara eșantionului cu repere simple: media, ultima valoare și netezirea exponențială din Capitolul 0. Un model care trece toată diagnosticarea în eșantion nu a arătat încă că prognozează mai bine. Seminarul 2, sarcina C1, face această comparație pentru inflația din România.

## Recapitulare: alte serii reale {h=6cc3c654}

Inflația este descrisă de un AR(2) aproape de o rădăcină unitară; modelul revine la media istorică și lasă corelație în reziduuri la lagul 12. Randamentele zilnice au efecte AR semnificative, dar foarte mici, iar varianța lor cere modele GARCH.

AR(2) al lui Yule regăsește ciclul solar de 11 ani din rădăcini complexe. Competițiile de prognoză arată că orice model trebuie comparat cu repere simple.

## @end

Aceleași instrumente dau verdicte diferite pe serii diferite. Pentru inflație, AR(2) este aproape de o rădăcină unitară și nu trece diagnosticarea la lagul 12; pentru randamentele zilnice, coeficienții AR sînt semnificativi, dar explică puțin, iar dependența se află în varianță; pentru petele solare, doi coeficienți produc ciclul de 11 ani, iar nouă laguri îi descriu mai bine asimetria. Competițiile de prognoză adaugă ultima verificare: comparația cu repere simple în afara eșantionului.

# Contribuția posibilă a AI

Instrumentele bazate pe inteligență artificială (AI, *artificial intelligence*) pot schița cod, pot oferi o a doua explicație a unei derivări și pot repeta bucla Box–Jenkins pe multe serii. Cele două cadre ale secțiunii descriu ce pot aduce și ce trebuie să verifice analistul înainte de a le folosi rezultatele.

## Contribuția posibilă a AI {h=c3b3430e}

Un asistent AI poate produce o primă variantă de cod care estimează modele ARMA($p,q$) pe o grilă de ordine, tabelează AIC și BIC și rulează testele pentru reziduuri. Poate oferi și o a doua derivare a unui rezultat, de exemplu ACF a unui ARMA(1,1) sau motivul pentru care PACF a unui MA descrește, pe care cititorul o poate compara rînd cu rînd cu derivarea din curs.

Poate extinde o analiză la multe serii: aceeași buclă Box–Jenkins aplicată creșterii PIB-ului tuturor țărilor din Uniunea Europeană (UE), de exemplu, pentru a compara persistența ciclurilor lor economice.

Exemplul de prompt de pe slide arată nivelul de detaliu de care are nevoie o cerere utilă: sursa datelor, grila de ordine, criteriile, gradele de libertate ale testului Ljung–Box (model_df = p + q) și orizontul de prognoză. O cerere vagă lasă aceste alegeri pe seama instrumentului, iar valorile implicite sînt adesea chiar cele împotriva cărora avertizează cadrul următor.

## Verificări necesare {h=f0951282}

Primele verificări privesc convențiile. Programele scriu polinomul MA fie ca $\theta(L) = 1 + \theta L$ (așa face statsmodels), fie ca $1 - \theta L$ (așa fac unele manuale), așa că semnul fiecărui $\theta$ trebuie verificat. Constanta raportată de clasa ARIMA din statsmodels este media $\mu$, nu termenul liber $c$.

Următoarele verificări privesc inferența. Testul Ljung–Box pentru reziduuri trebuie să folosească $m - p - q$ grade de libertate, iar toate modelele comparate prin criterii informaționale trebuie estimate pe același eșantion. Staționaritatea și invertibilitatea se citesc din rădăcinile ambelor polinoame, nu din mărimea coeficienților luați separat.

Ultimele verificări privesc metoda. Seria trebuie să fie staționară înainte de estimarea unui model ARMA (Capitolul 3), iar prognozele trebuie comparate cu repere simple. Orice referință citată de un instrument AI trebuie să existe; DOI-ul ei trebuie verificat.

## @end

Instrumentele AI scurtează scrierea codului și pot repeta analiza pe multe serii, dar deciziile care fac o analiză ARMA corectă rămîn ale analistului: convențiile de semn, semnificația constantei, gradele de libertate ale testului pentru reziduuri, rădăcinile, staționaritatea seriei și comparația cu reperele simple.

# Rezumat

Rezumatul adună rezultatele principale, formulele de reținut și trei întrebări de autoevaluare.

## Idei de reținut {h=983b67dc}

Modelul ARMA($p,q$) înlocuiește suma Wold infinită cu un raport a două polinoame scurte, $\theta(L)/\phi(L)$. El este staționar cînd rădăcinile lui $\phi(z)$ sînt în afara cercului unitate, invertibil cînd rădăcinile lui $\theta(z)$ sînt în afara lui și bine identificat cînd cele două polinoame nu au factori comuni. În date, o ACF care se anulează indică un MA, o PACF care se anulează un AR, iar două funcții descrescătoare un ARMA.

Modelele se estimează prin verosimilitate maximă, se aleg cu AIC sau BIC și se verifică cu testul Ljung–Box cu $m - p - q$ grade de libertate. Prognozele lor revin la medie, iar intervalele se lărgesc pînă la $\pm 1{,}96\sqrt{\gamma(0)}$, ignorînd incertitudinea parametrilor.

Pe datele reale, creșterea PIB-ului României este un MA(3) prin construcție, inflația este aproape de o rădăcină unitară, iar randamentele zilnice au nevoie de modele GARCH pentru varianța lor.

## Formule de reținut {h=3673ea8c}

Tabelul adună formulele capitolului. Primele rînduri descriu modelele: ecuația ARMA și forma ei cauzală $\psi(L) = \theta(L)/\phi(L)$; media, varianța și ACF a modelului AR(1); recurența Yule–Walker a modelului AR(2); prima autocorelație și condiția de invertibilitate a modelului MA(1); ponderile $\psi$ și recurența ACF a modelului ARMA(1,1).

Ultimele rînduri descriu metoda: estimatorul Yule–Walker, criteriile informaționale, distribuția de referință a statisticii Ljung–Box și intervalul de prognoză $\hat X_{T+h|T} \pm 1{,}96\,\sigma\sqrt{\sum_{j=0}^{h-1}\psi_j^2}$, care rezultă din varianța erorii @eq:fvar.

## Autoevaluare {h=472bef13}

Prima întrebare cere să stabilim dacă $X_t = 1{,}2X_{t-1} - 0{,}32X_{t-2} + \varepsilon_t$ este staționar. Polinomul caracteristic se descompune ca $1 - 1{,}2z + 0{,}32z^2 = (1 - 0{,}8z)(1 - 0{,}4z)$, deci rădăcinile sînt 1,25 și 2,5, ambele în afara cercului unitate, iar procesul este staționar. Rădăcinile inverse, 0,8 și 0,4, dau același răspuns, citit din interiorul cercului.

A doua întrebare descrie o serie a cărei ACF are o singură valoare semnificativă, $\hat\rho(1) = -0{,}45$, și a cărei PACF descrește. Tiparul este cel al unui MA(1) cu $\theta < 0$. Valoarea lui $\theta$ rezolvă ecuația $\theta/(1 + \theta^2) = -0{,}45$, o ecuație de gradul al doilea cu două rădăcini reale al căror produs este 1; se păstrează cea invertibilă, cu $|\theta| < 1$.

A treia întrebare dă $Q^*(10) = 15{,}2$ pentru reziduurile unui ARMA(2,1). Gradele de libertate sînt $10 - 3 = 7$, iar cuantila de 95% a lui $\chi^2(7)$ este 14,07, sub 15,2, deci testul respinge la 5%. Cu 10 grade de libertate, valoarea critică ar fi 18,31, iar modelul ar fi acceptat greșit. Capitolul 3 continuă cu rădăcinile unitare și modelele ARIMA, pentru cazul în care o rădăcină se află pe cercul unitate.

## @end

Capitolul a construit modelul ARMA staționar pornind de la descompunerea Wold, i-a derivat proprietățile, l-a estimat, l-a verificat și l-a folosit pentru prognoză. Toate aplicațiile au presupus o serie staționară. Capitolul 3 tratează cazul exclus aici, o rădăcină pe cercul unitate, și modelele ARIMA care îl rezolvă prin diferențiere.
