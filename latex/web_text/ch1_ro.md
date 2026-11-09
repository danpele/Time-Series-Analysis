<!--
Textul de curs al versiunii web, capitolul 1 (RO). Inserat în pagină de latex/build_web_lecture.py.
Format: # titlul secțiunii (introducere), ## titlul slide-ului {h=hash} (text), @after (text după blocul interactiv), ## @end (sinteză).
Ecuații numerotate: $$ ... $$ {#eq:eticheta}, trimitere @eq:eticheta. Citări: @Cheie -> Autor (an), [@Cheie] -> (Autor, an).
Hash-urile curente: python3 build_web_lecture.py 1 --hashes
-->

## @intro

@epigraph Slutzky1937 | The summation of random causes as the source of cyclic processes
> [I]s it possible that a definite structure of a connection between random fluctuations could form them into a system of more or less regular waves?

@text
O serie de timp se observă o singură dată. PIB-ul României din trimestrul al doilea al anului 2020, randamentul BET din 7 ianuarie 2009 sau cursul EUR/RON dintr-o anumită zi au fiecare o singură valoare înregistrată, iar istoria nu poate fi reluată pentru a vedea ce altceva s-ar fi putut întîmpla. Statistica se bazează însă pe repetiție: o medie sau o varianță este o proprietate a unei distribuții, iar o distribuție se cunoaște din multe extrageri. Întrebarea acestui capitol este ce poate dezvălui o singură traiectorie a unei serii despre mecanismul care a generat-o.

Răspunsul se sprijină pe două ipoteze. **Staționaritatea** (*stationarity*) cere ca distribuția seriei, sau măcar media și autocovarianțele ei, să nu se schimbe în timp, astfel încît observațiile de la date diferite să informeze despre aceiași parametri. **Ergodicitatea** (*ergodicity*) cere ca dependența dintre observațiile îndepărtate să se stingă, astfel încît mediile calculate de-a lungul unei singure traiectorii lungi să conveargă la mediile calculate pe toate traiectoriile posibile. În aceste ipoteze, instrumentul descriptiv principal este funcția de autocorelație (ACF, *autocorrelation function*), corelația seriei cu propriul trecut la fiecare lag.

Capitolul construiește aceste noțiuni de la definiții pînă la date. Deduce momentele zgomotului alb, ale mediei mobile, ale autoregresiei și ale mersului aleator; dă distribuția ACF de selecție și testele portmanteau ale lui @BoxPierce1970 și @LjungBox1978; arată cum logaritmul, diferențierea și transformarea Box–Cox apropie de staționaritate PIB-ul României, randamentele BET, indicele S&P 500 și două serii clasice, Nilul și petele solare.

@chart tsa_ch1_four_series
Patru serii românești ale cursului: PIB-ul real (trimestrial, T1 1995–T2 2026), inflația IAPC (lunară), cursul de referință EUR/RON al BNR (zilnic) și randamentele logaritmice zilnice ale indicelui BET din 2000; fiecare panou arată un alt mod în care o serie se abate de la staționaritate sau se apropie de ea.

@goals
- **Definiți** un proces stochastic și funcțiile lui de medie, autocovarianță și autocorelație.
- **Deosebiți** staționaritatea strictă de cea slabă și verificați staționaritatea slabă pentru procese simple.
- **Deduceți** momentele zgomotului alb, ale unei medii mobile, ale unei autoregresii și ale unui mers aleator.
- **Calculați** și interpretați ACF și PACF de selecție cu benzile $\pm 1{,}96/\sqrt{T}$.
- **Testați** autocorelația cu statisticile Box–Pierce și Ljung–Box.
- **Alegeți** o transformare (logaritm, diferență, Box–Cox) care apropie o serie reală de staționaritate.

@plan
- Serii de timp și procese stochastice | Seria ca traiectorie a unui proces; funcțiile de medie, autocovarianță și autocorelație.
- Staționaritate | Staționaritatea strictă și cea slabă, un contraexemplu, proprietățile autocovarianței, MA(1) și AR(1).
- Zgomotul alb și mersul aleator | Zgomotul alb slab, i.i.d. și gaussian; mersul aleator cu și fără derivă; indicele BET.
- Operatorul lag, diferențierea și descompunerea Wold | Polinoame în lag, procese integrate, teorema lui Wold și ergodicitatea.
- ACF și PACF de selecție | Estimatori, benzile lui Bartlett, corelogramele proceselor simulate și ale randamentelor BET.
- Teste portmanteau: Box–Pierce și Ljung–Box | Testarea simultană a mai multor autocorelații, mărimea testelor în eșantioane mici și serii reale.
- Transformări | Logaritmul și ratele de creștere ale PIB-ului României, transformarea Box–Cox și supradiferențierea.
- Două serii clasice | Yule și Slutzky, Nilul și petele solare.
- Contribuția posibilă a AI | Unde ajută instrumentele AI în analizele capitolului și ce verificări cer rezultatele lor.
- Rezumat | Ideile, formulele și întrebările de autoevaluare.

# Serii de timp și procese stochastice

Secțiunea stabilește limbajul cursului. O serie de timp este o înregistrare finită de numere, dar obiectul inferenței este mecanismul aleator din spatele ei, un proces stochastic. Secțiunea pornește de la patru serii românești care se comportă în patru moduri diferite, apoi definește procesul, arată diferența dintre o singură traiectorie și ansamblul tuturor traiectoriilor și introduce cele trei funcții care descriu un proces pînă la ordinul doi: media, autocovarianța și autocorelația.

## Patru serii ale acestui curs {h=67f31e2a}

Graficul arată patru serii care revin pe tot parcursul cursului, fiecare cu altă frecvență. PIB-ul real al României este trimestrial, din T1 1995 pînă în T2 2026, în prețuri constante (Eurostat). Inflația este rata anuală a indicelui armonizat al prețurilor de consum (IAPC), lunară (Eurostat). Cursul EUR/RON este cursul de referință zilnic publicat de Banca Națională a României (BNR). A patra serie este randamentul logaritmic zilnic al indicelui BET al Bursei de Valori București (BVB), din 2000.

Fiecare panou se citește urmărind două lucruri: dacă seria are un nivel fix în jurul căruia oscilează și dacă amplitudinea oscilațiilor este stabilă. Cele două întrebări anticipează cele două condiții ale staționarității slabe, definite în secțiunea următoare: o medie constantă și o autocovarianță care nu depinde de dată. Cele patru panouri răspund diferit, de aceea aceleași instrumente nu pot fi aplicate tuturor fără ca unele serii să fie mai întîi transformate.

## Interpretarea celor patru serii {h=1348a772}

PIB-ul real crește de la 16,5 la 45,9 mld. EUR pe trimestru. Are un trend și o sezonalitate a cărei amplitudine crește odată cu nivelul, deci media lui se schimbă în timp, iar varianța crește odată cu nivelul. Inflația se mișcă lent și persistent: a fost 102% în iunie 1997, −3,0% în mai 2016 și 6,1% în august 2026. O valoare ridicată în luna aceasta face probabilă o valoare ridicată și în luna următoare, ceea ce înseamnă o dependență puternică de trecut, nu neapărat o medie care se schimbă.

Cursul EUR/RON evoluează de la 3,60 la 5,26 lei și nu revine la niciun nivel fix; este un candidat pentru mersul aleator studiat în secțiunea a treia. Randamentele BET, dimpotrivă, oscilează în jurul unui nivel constant, apropiat de zero, cu extreme de −13,1% pe 7 ianuarie 2009 și +10,6% pe 10 mai 2010. Media lor pare stabilă, dar amplitudinea nu este: abaterea standard a randamentelor zilnice este 3,62% între septembrie 2008 și martie 2009 și 0,66% în 2017.

Cele patru serii ilustrează astfel patru abateri de la cazul din manual al datelor independente și identic distribuite: un trend cu sezonalitate, persistența, un nivel care rătăcește și o varianță care se schimbă în timp. Capitolul oferă conceptele care numesc fiecare dintre aceste comportamente și instrumentele care le măsoară.

## Seria de timp ca o realizare {h=e58d3bf8}

O serie de timp (*time series*) este un șir de observații $x_1, x_2, \dots, x_T$ ordonate în timp și înregistrate la intervale egale; $T$ este volumul eșantionului, iar $t$ indicele de timp, care numără zile, luni sau trimestre. Pasul esențial de modelare este să tratăm fiecare observație $x_t$ ca valoarea realizată a unei variabile aleatoare $X_t$. Randamentul BET de mîine este necunoscut azi: înainte de a fi observat, are o întreagă distribuție de valori posibile.

Dificultatea este că istoria se desfășoară o singură dată. Pentru fiecare dată vedem o singură extragere din distribuția lui $X_t$, niciodată o a doua. Dacă distribuțiile de la date diferite nu ar avea nicio legătură între ele, cele $T$ observații ar trebui să estimeze $T$ medii diferite, cîte una pentru fiecare dată, fiecare dintr-un singur număr, ceea ce este imposibil. Inferența are deci nevoie de ipoteze care leagă distribuțiile de la date diferite. Staționaritatea, care face distribuțiile asemănătoare în timp, și ergodicitatea, care face să se stingă dependența dintre datele îndepărtate, sînt cele două ipoteze dezvoltate în acest capitol.

## Procesul stochastic {h=a07e32cc}

Un proces stochastic (*stochastic process*) este o familie de variabile aleatoare $\{X_t : t \in \mathcal{T}\}$ definite pe același spațiu de probabilitate $(\Omega, \mathcal{F}, P)$. În acest curs, mulțimea de indici $\mathcal{T}$ este mulțimea numerelor întregi $\mathbb{Z}$ sau mulțimea finită $\{1, \dots, T\}$, adică timpul este discret. Un element $\omega$ din $\Omega$ poate fi privit ca o istorie completă posibilă a lumii, iar $P$ atribuie probabilități mulțimilor de istorii.

Valoarea $X_t(\omega)$ depinde de două argumente, iar fixarea fiecăruia dă un alt obiect. Cu data $t$ fixată, $X_t$ este o variabilă aleatoare: mulțimea valorilor pe care seria le-ar putea lua la acea dată, pe toate istoriile. Cu istoria $\omega$ fixată, funcția $t \mapsto X_t(\omega)$ este o **traiectorie** (*path*), numită și realizare. O serie de timp observată este o traiectorie.

Un proces este caracterizat de **distribuțiile finit-dimensionale** (*finite-dimensional distributions*), adică distribuțiile comune ale vectorilor $(X_{t_1}, \dots, X_{t_k})$ pentru orice mulțime finită de date $t_1, \dots, t_k$. Ele descriu tot ce se poate spune despre proces pe baza unui număr finit de observații. În practică, mulțimea completă a distribuțiilor comune este rar necesară; primele două momente, definite la sfîrșitul secțiunii, conțin deja cea mai mare parte din ce folosesc modelele liniare de serii de timp.

## Un ansamblu de traiectorii {h=822dfdc8}

Graficul compară două procese desenînd multe traiectorii ale fiecăruia. În stînga sînt 40 de traiectorii ale autoregresiei $X_t = 0{,}7X_{t-1} + \varepsilon_t$, unde șocurile $\varepsilon_t$ sînt independente și identic distribuite (i.i.d.), Normale standard. În dreapta sînt 40 de traiectorii ale mersului aleator $X_t = X_{t-1} + \varepsilon_t$, pornit din $X_0 = 0$ și generat de același tip de șocuri.

Banda colorată din fiecare panou marchează $\pm 1{,}96$ abateri standard ale lui $X_t$ la fiecare dată, intervalul care conține circa 95% din valorile posibile ale lui $X_t$ dacă acesta urmează distribuția Normală. Graficul se citește în două direcții: pe orizontală, fiecare linie este o istorie posibilă; pe verticală, norul de linii de deasupra unei date arată distribuția lui $X_t$ la acea dată. Întrebarea pusă de grafic este dacă această distribuție verticală arată la fel la orice dată.

## Interpretarea ansamblului de traiectorii {h=59a73437}

O secțiune verticală prin ansamblu la data $t$ arată distribuția lui $X_t$; o singură traiectorie arată o singură istorie. Momentele procesului, de exemplu media și varianța, sînt medii pe traiectorii la o dată fixată, nu medii de-a lungul unei traiectorii. Această distincție este miezul capitolului: datele dau medii de-a lungul unei traiectorii, în timp ce teoria este formulată în termeni de medii pe traiectorii.

În stînga, banda are aceeași lățime la orice dată. Pentru autoregresia cu coeficientul 0,7 și varianța șocurilor egală cu 1, varianța este $\mathrm{Var}(X_t) = 1/(1 - 0{,}7^2) = 1{,}96$, deci banda este $\pm 1{,}96\sqrt{1{,}96} = \pm 2{,}74$ la orice $t$. Distribuția lui $X_t$ nu depinde de $t$, o primă imagine a staționarității.

În dreapta, banda se lărgește în timp. Pe 5000 de traiectorii simulate ale mersului aleator, varianța lui $X_{100}$ este 100,8, față de valoarea teoretică 100, deoarece fiecare pas adaugă o unitate de varianță. Distribuția lui $X_t$ se schimbă cu $t$: mersul aleator nu este staționar și nicio varianță unică nu îl descrie.

## Funcțiile de medie, autocovarianță și autocorelație {h=ad04bf39}

Pentru un proces cu momente de ordinul doi finite, $E[X_t^2] < \infty$ pentru orice $t$, trei funcții descriu primele două momente. **Funcția de medie** (*mean function*) este $\mu_t = E[X_t]$. **Funcția de autocovarianță** (*autocovariance function*) este

$$ \gamma(t, s) = \mathrm{Cov}(X_t, X_s) = E[(X_t - \mu_t)(X_s - \mu_s)], $$ {#eq:acvf}

iar **funcția de autocorelație** este versiunea ei normalizată, $\rho(t, s) = \gamma(t, s)/\sqrt{\gamma(t, t)\,\gamma(s, s)}$. Prefixul „auto” arată că este vorba de covarianța seriei cu ea însăși, la două date diferite.

Trei proprietăți decurg direct din definiții. Pe diagonală, $\gamma(t, t) = \mathrm{Var}(X_t)$ și $\rho(t, t) = 1$. Din inegalitatea Cauchy–Schwarz, $|\rho(t, s)| \le 1$ pentru orice date. În general, ambele funcții depind separat de cele două date, deci un proces observat la $T$ date are $T$ medii și $T(T+1)/2$ autocovarianțe distincte, mult mai multe decît cele $T$ observații disponibile.

Exemplul $X_t = a + bt + \varepsilon_t$, cu $\varepsilon_t$ i.i.d. de medie 0 și varianță $\sigma^2$, arată funcțiile la lucru. Funcția de medie $\mu_t = a + bt$ depinde de $t$; varianța este $\gamma(t, t) = \sigma^2$ la orice dată, iar autocovarianțele sînt $\gamma(t, s) = 0$ pentru $t \ne s$. Dependența de timp stă în întregime în medie, astfel că procesul nu este staționar, deși abaterile lui de la dreaptă sînt zgomot pur.

## Recapitulare: serii de timp și procese stochastice {h=491ae790}

O serie de timp este o traiectorie a unui proces stochastic $\{X_t\}$. Momentele procesului, $\mu_t$, $\gamma(t, s)$ și $\rho(t, s)$, sînt definite ca medii pe traiectorii, la date fixate. Cu o singură traiectorie observată, aceste funcții pot fi estimate doar dacă nu se schimbă în timp, ipoteză pe care secțiunea următoare o formulează precis.

## @end

Secțiunea a separat seria observată de procesul care o generează. Imaginea ansamblului arată cele două direcții posibile de mediere, pe traiectorii la o dată fixată și de-a lungul unei traiectorii în timp, iar cele trei funcții $\mu_t$, $\gamma(t, s)$ și $\rho(t, s)$ din @eq:acvf sînt definite în prima direcție. Cele patru serii românești au arătat că datele reale încalcă în moduri diferite ipotezele cele mai simple. Secțiunea următoare definește staționaritatea, proprietatea care face aceste funcții estimabile dintr-o singură traiectorie.

# Staționaritate

Staționaritatea este ipoteza că structura probabilistică a unei serii nu depinde de data calendaristică. Ea are două forme. Staționaritatea strictă privește distribuții comune întregi; staționaritatea slabă privește doar media și autocovarianțele. Secțiunea le definește pe amîndouă, le compară, dă un proces care satisface forma slabă, dar nu și pe cea strictă, deduce proprietățile pe care trebuie să le aibă orice funcție de autocovarianță și calculează momentele celor două modele liniare de bază, media mobilă și autoregresia de ordinul 1.

## Staționaritatea strictă {h=440befb5}

Un proces $\{X_t\}$ este **strict staționar** (*strictly stationary*) dacă, pentru orice număr de date $k \ge 1$, orice date $t_1, \dots, t_k$ și orice deplasare $h$,

$$ (X_{t_1}, \dots, X_{t_k}) \overset{d}{=} (X_{t_1 + h}, \dots, X_{t_k + h}), $$ {#eq:strict}

unde $\overset{d}{=}$ înseamnă că cei doi vectori au aceeași distribuție comună. Deplasarea în timp a întregii ferestre de observație îi lasă neschimbat conținutul probabilistic.

Două consecințe arată ce impune definiția. Cu $k = 1$, toate variabilele $X_t$ au aceeași distribuție marginală. Cu $k = 2$, distribuția comună a perechii $(X_t, X_{t+h})$ depinde doar de lagul $h$, nu de data $t$. Un șir i.i.d. este cel mai simplu proces strict staționar.

Definiția privește distribuții întregi, deci nu cere existența momentelor: un șir i.i.d. de variabile Cauchy este strict staționar, deși nu are medie. Prețul acestei generalități este că staționaritatea strictă este greu de verificat din date, deoarece ar trebui comparate toate distribuțiile comune, de orice ordin.

## Staționaritatea slabă {h=e35c0f5d}

Noțiunea mai slabă, datorată lui @Khintchine1934, restricționează doar primele două momente. Un proces este **slab staționar** (*weakly stationary*), numit și staționar în covarianță sau staționar de ordinul doi, dacă sînt îndeplinite trei condiții: (i) $E[X_t^2] < \infty$ pentru orice $t$; (ii) $E[X_t] = \mu$, aceeași pentru orice $t$; (iii) $\mathrm{Cov}(X_t, X_{t+h}) = \gamma(h)$ depinde doar de lagul $h$, nu de $t$.

În aceste condiții, funcțiile cu două argumente din secțiunea precedentă devin funcții de un singur argument, lagul:

$$ \begin{gathered} \gamma(h) = \mathrm{Cov}(X_t, X_{t+h}) \\ \rho(h) = \frac{\gamma(h)}{\gamma(0)} \end{gathered} $$ {#eq:acf}

cu $\gamma(0) = \mathrm{Var}(X_t)$. Șirul $\rho(h)$ este funcția de autocorelație (ACF). În locul celor $T$ medii și $T(T+1)/2$ autocovarianțe rămîn o singură medie și cîte o autocovarianță pentru fiecare lag, iar fiecare pereche de observații aflate la distanța $h$ aduce informație despre același număr $\gamma(h)$.

În acest curs, „staționar” înseamnă slab staționar, dacă nu se precizează altfel. Forma slabă este cea de care au nevoie modelele liniare, de exemplu ARMA, și poate fi verificată, cel puțin informal, din media de selecție și din ACF de selecție.

## Staționaritatea strictă și cea slabă {h=223d1b5c}

Niciuna dintre definiții nu o implică pe cealaltă fără o condiție suplimentară. Dacă un proces este strict staționar și are varianță finită, el este slab staționar: distribuțiile marginale identice dau medii identice, iar distribuțiile comune identice ale lui $(X_t, X_{t+h})$ dau covarianțe identice. Varianța finită este esențială. Un șir i.i.d. de variabile Student-t cu 2 grade de libertate este strict staționar, dar are varianță infinită, deci condiția (i) nu este îndeplinită și șirul nu este slab staționar.

În sens invers, staționaritatea slabă fixează doar primele două momente, deci forma distribuției se poate schimba în timp, în timp ce media și autocovarianțele rămîn fixe; slide-ul următor construiește un astfel de proces. Excepția este clasa **proceselor gaussiene** (*Gaussian processes*), ale căror distribuții finit-dimensionale sînt toate Normale multivariate. O distribuție Normală multivariată este determinată complet de medii și covarianțe, deci pentru un proces gaussian staționaritatea slabă și cea strictă coincid.

## Slab, dar nu strict staționar {h=6e81a68a}

Graficul arată un proces construit pentru a fi slab, dar nu strict staționar. Variabilele $X_t$ sînt independente; la datele pare $X_t$ urmează distribuția Normală standard, $N(0, 1)$, iar la datele impare este o variabilă hi-pătrat standardizată, $(\chi^2_5 - 5)/\sqrt{10}$. Distribuția hi-pătrat cu 5 grade de libertate are media 5 și varianța 10, deci ambele distribuții au media 0 și varianța 1.

Panoul din stînga arată primele 100 de observații: nimic din traiectorie nu sugerează că alternează două distribuții. Panoul din dreapta suprapune histogramele valorilor de la datele pare și de la cele impare, într-un eșantion simulat de 4000 de observații. Varianțele de selecție sînt 0,99 la datele pare și 1,00 la cele impare, așa cum cere construcția. Formele diferă însă: asimetria (*skewness*) de selecție este −0,02 la datele pare și 1,30 la cele impare, față de valorile teoretice 0 și 1,26. Cele două histograme se citesc după simetrie și după coada din stînga: histograma datelor impare se oprește brusc în stînga și are o coadă mai lungă în dreapta.

## Interpretarea contraexemplului {h=55d5a73f}

Procesul este slab staționar. La orice dată $E[X_t] = 0$ și $\gamma(0) = 1$, iar independența dă $\gamma(h) = 0$ pentru $h \ne 0$. Media și ACF sînt exact cele ale zgomotului alb gaussian, deci niciun instrument bazat pe momentele de ordinul doi nu le poate deosebi.

Procesul nu este strict staționar, deoarece distribuția lui $X_t$ alternează între una simetrică și una asimetrică la dreapta. Coada din stînga arată cel mai clar diferența. Probabilitatea $P(X_t < -1{,}6)$ este circa 5% la datele pare, unde $X_t$ are distribuția Normală, și exact 0 la datele impare, deoarece o variabilă hi-pătrat este nenegativă și deci $X_t \ge -5/\sqrt{10} = -1{,}58$.

Lecția este că momentele de ordinul doi rezumă dependența liniară, dar nu întreaga distribuție. Întrebările despre pierderile extreme, care depind de cozi, cer mai mult decît media și ACF; capitolul 5 revine la ele cu modelele de volatilitate condiționată.

## Proprietățile funcției de autocovarianță {h=928da813}

Funcția de autocovarianță a unui proces slab staționar are trei proprietăți. Este **simetrică**, $\gamma(-h) = \gamma(h)$, deoarece $\mathrm{Cov}(X_t, X_{t-h}) = \mathrm{Cov}(X_{t-h}, X_t)$; de aceea corelogramele arată doar lagurile nenegative. Este **mărginită** de valoarea din zero, $|\gamma(h)| \le \gamma(0)$, deci $|\rho(h)| \le 1$. Și este **pozitiv semidefinită** (*non-negative definite*): pentru orice $n$ și orice numere reale $a_1, \dots, a_n$,

$$ \sum_{i,j} a_i a_j\, \gamma(i - j) \ge 0. $$ {#eq:nnd}

Demonstrația lui @eq:nnd are un singur rînd: membrul stîng este varianța combinației liniare $\sum_i a_i X_i$, deoarece

$$ \begin{gathered} \mathrm{Var}\Big(\sum_i a_i X_i\Big) = \sum_{i,j} a_i a_j \mathrm{Cov}(X_i, X_j) \\ = \sum_{i,j} a_i a_j \gamma(i - j), \end{gathered} $$

iar o varianță nu poate fi negativă.

Consecința este că nu orice șir de numere cuprinse între −1 și 1 poate fi o ACF. Să luăm $\rho(1) = 0{,}9$ și $\rho(2) = -0{,}9$: observațiile vecine ar fi aproape perfect corelate pozitiv, iar cele aflate la doi pași distanță aproape perfect corelate negativ. Combinația $X_1 - X_2 + X_3$ scoate la iveală contradicția: varianța ei ar fi

$$ \begin{gathered} \mathrm{Var}(X_1 - X_2 + X_3) = \gamma(0)\,(3 - 4\rho(1) + 2\rho(2)) \\ = \gamma(0)\,(3 - 3{,}6 - 1{,}8) = -2{,}4\,\gamma(0), \end{gathered} $$

adică negativă. Pozitiv semidefinirea este și proprietatea pe care trebuie să o păstreze estimatorii de selecție ai lui $\gamma(h)$, după cum arată secțiunea despre ACF de selecție.

## Exemplu rezolvat: o medie mobilă de ordinul 1 {h=fb732e07}

Media mobilă de ordinul 1, MA(1) (*moving average*), este $X_t = \varepsilon_t + \theta\varepsilon_{t-1}$, unde $\{\varepsilon_t\}$ sînt șocuri necorelate cu media 0 și varianța $\sigma^2$, iar $\theta$ este ponderea șocului din perioada anterioară. Fiecare observație combină șocul de azi cu cel de ieri. Media este $E[X_t] = 0$, deoarece ambele șocuri au media 0.

Varianța se obține dezvoltînd pătratul: $\gamma(0) = E[(\varepsilon_t + \theta\varepsilon_{t-1})^2] = \sigma^2(1 + \theta^2)$, deoarece termenul încrucișat $2\theta\varepsilon_t\varepsilon_{t-1}$ are media 0. La lagul 1, produsul $(\varepsilon_t + \theta\varepsilon_{t-1})(\varepsilon_{t+1} + \theta\varepsilon_t)$ conține un singur șoc comun ambilor factori, $\varepsilon_t$, care contribuie cu $\theta\sigma^2$, deci $\gamma(1) = \theta\sigma^2$. La laguri $|h| \ge 2$, cele două observații nu au niciun șoc comun și $\gamma(h) = 0$. Prin urmare,

$$ \begin{gathered} \rho(1) = \frac{\theta}{1 + \theta^2} \\ \rho(h) = 0 \ \text{pentru } |h| \ge 2. \end{gathered} $$ {#eq:ma1}

Niciunul dintre aceste momente nu depinde de $t$, deci MA(1) este slab staționar pentru orice valoare a lui $\theta$. Cu $\theta = 0{,}6$ și $\sigma^2 = 1$, varianța este $\gamma(0) = 1{,}36$, iar prima autocorelație este $\rho(1) = 0{,}441$. Funcția $\theta/(1 + \theta^2)$ își atinge maximul 0,5 la $\theta = 1$, deci un MA(1) nu poate avea niciodată $|\rho(1)| > 0{,}5$; o primă autocorelație de selecție mult peste 0,5, fără nimic după ea, exclude acest model.

## Exemplu rezolvat: o autoregresie de ordinul 1 {h=854bbb74}

Autoregresia de ordinul 1, AR(1) (*autoregression*), este $X_t = \phi X_{t-1} + \varepsilon_t$ cu $|\phi| < 1$, unde $\phi$ este ponderea valorii din perioada anterioară, iar $\varepsilon_t$ sînt șocuri necorelate cu media 0 și varianța $\sigma^2$, necorelate cu trecutul lui $X$. Metoda este să presupunem că procesul este staționar și să aflăm momentele pe care le implică staționaritatea. Aplicînd media, obținem $\mu = \phi\mu$, deci $\mu = 0$. Aplicînd varianța, obținem $\gamma(0) = \phi^2\gamma(0) + \sigma^2$, deci $\gamma(0) = \sigma^2/(1 - \phi^2)$.

Pentru autocovarianțe, înmulțim ecuația cu $X_{t-h}$, $h \ge 1$, și aplicăm media. Șocul $\varepsilon_t$ este necorelat cu $X_{t-h}$, deci $\gamma(h) = \phi\gamma(h - 1)$, iar prin iterare

$$ \begin{gathered} \gamma(0) = \frac{\sigma^2}{1 - \phi^2} \\ \rho(h) = \phi^{|h|}. \end{gathered} $$ {#eq:ar1}

Cu $\phi = 0{,}8$ și $\sigma^2 = 1$, varianța este $\gamma(0) = 2{,}78$, iar autocorelațiile sînt $\rho(1) = 0{,}8$, $\rho(5) = 0{,}328$ și $\rho(10) = 0{,}107$: ACF descrește geometric, fără să ajungă vreodată exact la zero. Această descreștere geometrică este semnătura dependenței autoregresive într-o corelogramă.

Deducția arată și unde se termină staționaritatea. Cu $|\phi| \ge 1$, formula lui $\gamma(0)$ ar da o varianță negativă sau infinită, deci nu există o soluție staționară de această formă. Cazul $\phi = 1$ este mersul aleator din secțiunea următoare; capitolele 2 și 3 tratează condițiile generale de staționaritate și testele de rădăcină unitară (*unit root*).

## Întrebare pentru sală: care serii sînt staționare? {h=1cb29347}

Graficul arată patru serii simulate, cu cîte 300 de observații. La baza tuturor stă același zgomot AR(1) cu $\phi = 0{,}5$, astfel că oscilațiile lor pe termen scurt arată la fel; ceea ce diferă este ce s-a adăugat acestui zgomot sau cu ce a fost înmulțit.

Întrebarea este care dintre cele două condiții privind momentele, din definiția staționarității slabe, nu este îndeplinită în fiecare panou: condiția (ii), o medie constantă, sau condiția (iii), o autocovarianță care depinde doar de lag, ceea ce la lagul 0 înseamnă o varianță constantă. Înainte de a citi răspunsul, căutați în fiecare panou un centru care se deplasează, o împrăștiere care se schimbă de-a lungul eșantionului și o schimbare bruscă de nivel.

## Răspuns: doar seria A este staționară {h=579a141c}

Seria A este chiar procesul AR(1) staționar: media și varianța ei sînt constante, iar ACF depinde doar de lag. Seria B adaugă un trend determinist, $\mu_t = 0{,}04\,t$, deci media se schimbă în timp și condiția (ii) nu este îndeplinită. O astfel de serie se numește **staționară în jurul trendului** (*trend-stationary*): după scăderea dreptei, ceea ce rămîne este staționar.

Seria C înmulțește zgomotul cu un factor care crește de-a lungul eșantionului, astfel încît abaterea standard este la sfîrșit de 7,5 ori mai mare decît la început. Media rămîne constantă, dar varianța nu, deci condiția (iii) nu este îndeplinită la $h = 0$. Seria D are un salt de nivel de $+4$ la $t = 150$: media are o valoare înainte de ruptură și alta după, deci condiția (ii) nu este îndeplinită. În datele reale, o astfel de ruptură structurală (*structural break*) corespunde unei schimbări de regim sau de politică; seria Nilului din ultima secțiune principală este un exemplu.

În practică, graficul seriei este prima verificare pentru toate cele trei tipuri de nestaționaritate. Testele formale, care deosebesc un trend determinist de unul stochastic, apar în capitolul 3.

## Recapitulare: staționaritate {h=ae7e0534}

Staționaritatea strictă @eq:strict cere ca toate distribuțiile comune să fie invariante la deplasări în timp; staționaritatea slabă cere o medie constantă și o autocovarianță care depinde doar de lag @eq:acf. Staționaritatea strictă împreună cu varianța finită implică staționaritatea slabă, iar pentru procesele gaussiene cele două coincid. MA(1) este staționar pentru orice $\theta$; AR(1) este staționar pentru $|\phi| < 1$, cu $\rho(h) = \phi^{|h|}$. Trendurile, varianța variabilă și rupturile structurale încalcă toate staționaritatea.

## @end

Staționaritatea slabă reduce descrierea unui proces la o singură medie și cîte o autocovarianță pentru fiecare lag, ceea ce face posibilă estimarea dintr-o singură traiectorie. Funcția de autocovarianță este simetrică, mărginită și pozitiv semidefinită @eq:nnd, deci nu orice șir este o ACF. Cele două exemple rezolvate au dat cele două forme care revin în toate capitolele următoare: ACF a procesului MA(1) @eq:ma1 se anulează după lagul 1, iar ACF a procesului AR(1) @eq:ar1 descrește geometric. Secțiunea următoare introduce piesa de bază a ambelor modele, zgomotul alb, și cel mai simplu proces nestaționar construit din el, mersul aleator.

# Zgomotul alb și mersul aleator

Zgomotul alb este șirul de șocuri din care este construit fiecare model al cursului, iar mersul aleator este ceea ce se obține prin acumularea acestor șocuri. Secțiunea definește trei variante de zgomot alb, de tărie crescătoare, și arată prin simulare că varianta cea mai slabă poate ascunde o dependență puternică în pătrate. Apoi deduce momentele mersului aleator, arată de ce nu este staționar, adaugă o derivă și întreabă dacă indicele BET poate fi deosebit cu ochiul liber de un mers aleator.

## Zgomotul alb {h=44fecf71}

Un proces $\{\varepsilon_t\}$ este **zgomot alb** (*white noise*), notat $\varepsilon_t \sim \mathrm{WN}(0, \sigma^2)$, dacă are media zero, $E[\varepsilon_t] = 0$, varianță constantă, $\mathrm{Var}(\varepsilon_t) = \sigma^2$, și nicio corelație între date diferite, $\mathrm{Cov}(\varepsilon_t, \varepsilon_s) = 0$ pentru $t \ne s$. ACF lui este deci $\rho(0) = 1$ și $\rho(h) = 0$ pentru orice $h \ne 0$. Numele vine de la spectru: asemenea luminii albe, procesul are aceeași putere la toate frecvențele.

Se folosesc trei variante, de la cea mai slabă la cea mai tare. Zgomotul alb **slab** este doar necorelat; dependența neliniară este permisă, de exemplu între pătratele $\varepsilon_t^2$ și $\varepsilon_{t-1}^2$. Zgomotul alb **i.i.d.**, sau tare, cere ca șocurile să fie independente și identic distribuite, ceea ce exclude orice dependență. Zgomotul alb **gaussian** este i.i.d. $N(0, \sigma^2)$; pentru variabile comun Normale, corelația nulă implică independența, deci în cazul gaussian varianta slabă și cea tare coincid.

Zgomotul alb este piesa de bază a fiecărui model din acest curs. Modelele ARMA îl filtrează liniar, modelele ARIMA îl acumulează, modelele GARCH îl scalează cu o volatilitate variabilă în timp, iar modelele VAR îl extind la vectori. Contează ce variantă presupune un model: prognozele mediei au nevoie doar de zgomot alb slab, în timp ce afirmațiile despre întreaga distribuție cer mai mult.

## Trei tipuri de zgomot alb {h=b8cce2fb}

Graficul simulează cîte 1000 de valori din trei zgomote albe, fiecare cu varianța 1, și arată 500 dintre ele pe rîndul de sus. Primul este gaussian. Al doilea este Laplace, o distribuție i.i.d. cu cozi mai groase decît distribuția Normală. Al treilea este un proces GARCH(1,1), modelul de volatilitate din capitolul 5: $\varepsilon_t = \sigma_t z_t$, cu $z_t$ i.i.d. $N(0, 1)$ și

$$ \sigma_t^2 = 0{,}05 + 0{,}10\,\varepsilon_{t-1}^2 + 0{,}85\,\sigma_{t-1}^2, $$

unde $\sigma_t^2$ este varianța zilei $t$ condiționată de trecut. Varianța necondiționată a acestui proces GARCH este $0{,}05/(1 - 0{,}10 - 0{,}85) = 1$, deci toate cele trei serii au varianța 1.

Rîndul de jos arată ACF de selecție a pătratelor $X_t^2$, nu a seriei, pentru lagurile de la 1 la 20, cu banda $\pm 1{,}96/\sqrt{T}$ marcată prin linii întrerupte. Pe rîndul de sus, seria GARCH iese în evidență prin alternanța perioadelor liniștite și agitate; pe rîndul de jos, aceasta apare ca autocorelație pozitivă a pătratelor. Toate cele trei serii sînt necorelate prin construcție; întrebarea pusă de grafic este dacă sînt și independente.

## Interpretarea celor trei zgomote albe {h=814fd0c1}

Statistica Ljung–Box $Q^*(10)$, definită în secțiunea despre testele portmanteau, testează simultan primele 10 autocorelații ale fiecărei serii. P-value-ul ei este 0,278 pentru seria gaussiană și 0,986 pentru seria Laplace, deci niciuna nu este separată de zgomotul alb de acest test. Pentru seria GARCH, p-value-ul este 0,049, chiar sub 5%, deci testul respinge ipoteza de zgomot alb la nivelul de 5%, deși seria este necorelată prin construcție. Cînd varianța se schimbă în timp, testul obișnuit, ale cărui valori critice presupun date i.i.d., respinge prea des.

Seria Laplace are cozi mai groase decît distribuția Normală, cu un kurtosis de selecție de 5,2, față de 3,0 pentru distribuția Normală, dar este tot i.i.d.: testul Ljung–Box pentru pătratele ei dă un p-value de 0,680. Cozile groase singure nu creează dependență. Pentru seria GARCH, în schimb, pătratele sînt puternic corelate, cu $Q^*(10) = 73{,}2$ și un p-value sub 0,001.

Seria GARCH este deci un zgomot alb slab care nu este i.i.d.: semnul și nivelul valorii de mîine nu pot fi anticipate din trecut, dar mărimea ei poate fi. Acesta este volatility clustering, mișcările mari urmate de mișcări mari, iar randamentele zilnice ale acțiunilor se comportă exact așa, după cum arată corelogramele BET din secțiunea despre ACF de selecție.

## Mersul aleator {h=8fe7d34c}

**Mersul aleator** (*random walk*) este definit prin $X_t = X_{t-1} + \varepsilon_t$ pentru $t \ge 1$, cu $X_0 = 0$ și $\varepsilon_t \sim \mathrm{WN}(0, \sigma^2)$. Prin substituție înapoi se obține $X_t = \sum_{i=1}^{t}\varepsilon_i$: nivelul este suma tuturor șocurilor trecute. Fiecare șoc rămîne pentru totdeauna în nivel, deoarece $\partial X_{t+h}/\partial\varepsilon_t = 1$ pentru orice $h \ge 0$; nimic nu readuce seria acolo unde a fost.

Șocurile necorelate sînt suficiente pentru deducerea momentelor. Media este $E[X_t] = 0$. Varianța unei sume de termeni necorelați este suma varianțelor lor, deci $\mathrm{Var}(X_t) = t\sigma^2$. Pentru $t \le s$, sumele $\sum_{i \le t}\varepsilon_i$ și $\sum_{j \le s}\varepsilon_j$ au în comun exact primele $t$ șocuri, deci

$$ \begin{gathered} \mathrm{Var}(X_t) = t\sigma^2 \\ \mathrm{Cov}(X_t, X_s) = \min(t, s)\,\sigma^2 \\ \rho(t, s) = \sqrt{t/s}\ \ (t \le s). \end{gathered} $$ {#eq:rw}

Mersul aleator nu este staționar din două motive: varianța lui crește cu $t$, iar autocorelația depinde de datele însele, nu doar de distanța $s - t$ dintre ele. Două observații aflate la zece perioade distanță au corelația $\rho(X_{100}, X_{110}) = \sqrt{100/110} = 0{,}953$ tîrziu în eșantion, dar $\rho(X_{10}, X_{20}) = \sqrt{0{,}5} = 0{,}707$ la începutul lui. Același lag dă corelații diferite, ceea ce condiția (iii) interzice.

## Mersuri aleatoare și varianța lor {h=39fb63db}

Panoul din stînga desenează 60 de traiectorii ale mersului aleator cu $\sigma = 1$, împreună cu banda $\pm 1{,}96\sqrt{t}$, care se lărgește ca rădăcina pătrată a timpului, deoarece abaterea standard a lui $X_t$ este $\sqrt{t}\,\sigma$. Panoul din dreapta reprezintă, în funcție de $t$, varianța lui $X_t$ calculată pe 5000 de traiectorii simulate, împreună cu dreapta teoretică $t\sigma^2$.

La $t = 250$, varianța simulată este 249,1, față de valoarea teoretică 250, iar 5,0% dintre traiectorii se află în afara benzii, cum implică o bandă de 95% pentru o variabilă cu distribuția Normală. Dreapta din panoul din dreapta este forma vizuală a lui @eq:rw: varianța se acumulează liniar cu numărul de șocuri.

## Interpretarea mersurilor aleatoare {h=7f3b4eeb}

Împrăștierea traiectoriilor crește ca $\sqrt{t}$, deci incertitudinea privind nivelul se acumulează fără limită. O prognoză a nivelului peste 100 de zile, făcută azi, are varianța erorii $100\sigma^2$, deoarece cele 100 de șocuri viitoare sînt toate necunoscute și toate permanente. Pentru procesele staționare, în schimb, varianța erorii de prognoză converge la varianța finită a seriei.

Traiectoriile par să aibă trenduri și cicluri: creșteri lungi, scăderi lungi, puncte de întoarcere aparente. Nimic în afară de hazard nu le generează. Aceasta este lecția lui @Slutzky1937: sumele de șocuri aleatoare pot semăna cu ciclurile economice, iar un tipar văzut pe o singură traiectorie nu este o dovadă a unui mecanism sistematic.

Pe traiectoriile simulate, corelația dintre $X_{100}$ și $X_{110}$ este 0,953, egală cu valoarea teoretică 0,953. Nivelurile depărtate în timp rămîn puternic corelate, deoarece au în comun majoritatea șocurilor, astfel că ACF de selecție a unui mers aleator scade foarte încet; acesta este tiparul de urmărit în corelogramele de mai departe din capitol.

## Mersul aleator cu derivă {h=ab536443}

Adăugarea unei constante $c$ la fiecare pas dă **mersul aleator cu derivă** (*random walk with drift*), $X_t = c + X_{t-1} + \varepsilon_t$ cu $X_0 = 0$, astfel încît $X_t = ct + \sum_{i=1}^{t}\varepsilon_i$. Deriva $c$ produce un trend liniar în medie, $E[X_t] = ct$, în timp ce varianța rămîne $\mathrm{Var}(X_t) = t\sigma^2$. Prima diferență, $\Delta X_t = X_t - X_{t-1} = c + \varepsilon_t$, este staționară: zgomot alb plus o constantă. O serie care devine staționară după diferențiere se numește **staționară în diferențe** (*difference-stationary*), iar șocurile acumulate formează un **trend stochastic** (*stochastic trend*).

Comparația se face cu seria staționară în jurul trendului $Y_t = a + bt + u_t$, unde $u_t$ este staționar, $a$ este termenul liber, iar $b$ panta trendului. Ambele serii au o medie liniară și, pe un eșantion finit, pot arăta la fel. Ele reacționează însă diferit la șocuri: în $Y_t$, un șoc asupra lui $u_t$ se stinge și seria revine la dreapta ei, în timp ce în $X_t$ fiecare șoc deplasează întreaga traiectorie viitoare. Prognozele lui $Y_t$ au varianța erorii mărginită; prognozele lui $X_t$ nu.

A decide care dintre cele două descrie o serie dată este problema rădăcinii unitare din capitolul 3. Pentru datele financiare, punctul de plecare obișnuit este mersul aleator cu derivă: logaritmii prețurilor acțiunilor și ai indicilor sînt adesea apropiați de un mers aleator cu o derivă mică și pozitivă, motiv pentru care obiectul analizei sînt variațiile lor, adică randamentele.

## Bursa de Valori București {h=c179cdc6}

Indicele BET este indicele de referință al Bursei de Valori București (BVB), calculat din 19 septembrie 1997. Analiza din acest capitol folosește nivelurile lui zilnice de închidere din 2000, care dau 6 670 de randamente logaritmice $r_t = 100\,\Delta\ln P_t$, unde $P_t$ este nivelul de închidere din ziua $t$; înmulțirea cu 100 exprimă randamentele în procente. Abaterea lor standard este 1,40% pe zi.

Dacă logaritmul prețului, $\ln P_t$, ar fi un mers aleator cu derivă, randamentele $r_t$ ar fi zgomot alb în jurul unei medii constante. Slide-ul următor supune această ipoteză unui prim test, vizual, ascunzînd indicele BET printre mersuri aleatoare simulate cu aceeași derivă și aceeași volatilitate zilnică.

## Întrebare pentru sală: care serie este cea reală? {h=8bc6cf4f}

Graficul are șase panouri, notate de la A la F. Unul este logaritmul prețului BET în ultimele 500 de zile de tranzacționare; celelalte cinci sînt mersuri aleatoare simulate cu aceeași derivă și aceeași abatere standard zilnică pe care le-a avut BET în acea perioadă. Fiecare panou reprezintă variația logaritmului prețului față de prima zi, în procente, pe propria scară verticală.

Întrebarea este care panou este BET. Căutați trăsături pe care un mers aleator nu le-ar putea produce: un nivel la care seria revine, porțiuni mai netede sau mai agitate decît ar da hazardul, sau un trend prea regulat pentru a fi zgomot acumulat. Ce credeți: ce ar însemna dacă niciun panou nu ar ieși în evidență?

## Răspuns: seria D {h=12c26ada}

Seria D este BET, între 11 septembrie 2024 și 18 septembrie 2026; în această perioadă, variația zilnică medie este 0,13%, iar abaterea standard 0,99%. Cu ochiul liber, seria nu poate fi deosebită de cele cinci mersuri aleatoare: creșterile lungi, pauzele și reculurile de felul celor din indice sînt exact ceea ce produc șocurile aleatoare acumulate.

Datele nu sînt totuși exact un mers aleator. Autocorelația de ordinul 1 a variațiilor zilnice este 0,12, iar testul Ljung–Box pentru primele 10 autocorelații, $Q^*(10)$, are un p-value de 0,018, deci ipoteza de zgomot alb este respinsă la nivelul de 5%. Aceste abateri sînt mici și invizibile în nivel, unde șocurile acumulate domină imaginea, dar apar în ACF a diferențelor.

Lecția este una de metodă: nivelul unei serii de prețuri îi ascunde dependența, iar variațiile o dezvăluie. Restul capitolului studiază de aceea variațiile, cu ACF și testele portmanteau ca instrumente principale.

## Recapitulare: zgomotul alb și mersul aleator {h=535e1a57}

Zgomotul alb are media 0, varianța constantă și nicio autocorelație; are o variantă slabă, una i.i.d. și una gaussiană. Necorelat nu înseamnă independent: pătratele unui zgomot alb slab pot fi puternic corelate. Mersul aleator are $\mathrm{Var}(X_t) = t\sigma^2$ și $\rho(t, s) = \sqrt{t/s}$, deci nu este staționar, în timp ce prima lui diferență este zgomot alb. Seriile staționare în jurul trendului și cele staționare în diferențe pot arăta la fel, dar reacționează diferit la șocuri.

## @end

Zgomotul alb este unitatea elementară a cursului, iar distincția dintre varianta lui slabă și cea i.i.d. este distincția dintre niveluri imprevizibile și mărimi imprevizibile. Mersul aleator @eq:rw acumulează zgomotul alb într-o serie a cărei varianță crește fără limită și ale cărei corelații depind de dată; adăugarea unei derive dă trendul stochastic tipic pentru logaritmii prețurilor. Exemplul BET a arătat că nivelul unei serii de prețuri ascunde o dependență mică, dar semnificativă, care apare doar în variații. Secțiunea următoare dă algebra trecerii de la niveluri la variații, operatorul lag și diferențierea, și teorema care reprezintă orice proces staționar prin zgomot alb.

# Operatorul lag, diferențierea și descompunerea Wold

Secțiunea oferă notația și cele două rezultate teoretice pe care se sprijină capitolele următoare. Operatorul lag transformă modelele de serii de timp în algebră de polinoame, iar diferențierea, scrisă cu operatorul lag, este principalul mod de a transforma o serie nestaționară într-una staționară. Descompunerea Wold arată că orice proces staționar este o sumă ponderată de șocuri de tip zgomot alb, iar ergodicitatea precizează cînd mediile calculate de-a lungul unei singure traiectorii regăsesc momentele procesului.

## Operatorul lag {h=fe07f62d}

**Operatorul lag** (*lag operator*) $L$ deplasează o serie cu o perioadă înapoi, $LX_t = X_{t-1}$; puterile lui deplasează cu mai multe perioade, $L^kX_t = X_{t-k}$, iar $L^0 = 1$ lasă seria neschimbată. Unele texte îl notează $B$, de la *backshift*. O constantă nu este modificată de operator, $Lc = c$, deoarece are aceeași valoare la orice dată.

Un **polinom în lag** (*lag polynomial*) de grad $p$ este $\phi(L) = 1 - \phi_1L - \dots - \phi_pL^p$, cu coeficienții reali $\phi_1, \dots, \phi_p$; aplicat unei serii, el dă $\phi(L)X_t = X_t - \phi_1X_{t-1} - \dots - \phi_pX_{t-p}$. Cele două modele din secțiunea precedentă devin compacte: AR(1) este $(1 - \phi L)X_t = \varepsilon_t$, iar MA(1) este $X_t = (1 + \theta L)\varepsilon_t$. Polinoamele în lag se înmulțesc ca polinoamele obișnuite, de exemplu $(1 - L)(1 + L) = 1 - L^2$.

Operatorul poate fi și inversat. Pentru $|\phi| < 1$, seria geometrică dă

$$ \begin{gathered} (1 - \phi L)^{-1} = 1 + \phi L + \phi^2L^2 + \dots \\ X_t = \sum_{j \ge 0}\phi^j\varepsilon_{t-j} \end{gathered} $$ {#eq:ar1ma}

deci AR(1) este o medie mobilă infinită a șocurilor trecute, cu ponderi care scad geometric. Condiția $|\phi| < 1$ face ponderile sumabile; este aceeași condiție care a făcut finită varianța din @eq:ar1. Modelele ARMA, ARIMA și SARIMA din capitolele 2–4 se scriu toate cu polinoame în lag.

## Diferențierea {h=5d7431fe}

**Prima diferență** (*first difference*) este $\Delta X_t = (1 - L)X_t = X_t - X_{t-1}$, iar **a doua diferență** este $\Delta^2X_t = (1 - L)^2X_t = X_t - 2X_{t-1} + X_{t-2}$. **Diferența sezonieră** (*seasonal difference*) cu perioada $s$ este $\Delta_sX_t = (1 - L^s)X_t = X_t - X_{t-s}$, care compară fiecare observație cu același sezon din anul anterior: $s = 4$ pentru date trimestriale și $s = 12$ pentru date lunare.

Un exemplu mic arată mecanismul. Pentru $x = (10, 12, 15, 14, 18)$, primele diferențe sînt $\Delta x = (2, 3, -1, 4)$, iar a doua diferență este $\Delta^2x = (1, -4, 5)$. Fiecare diferențiere pierde o observație la începutul eșantionului, iar o diferență sezonieră pierde $s$ observații.

Diferențierea elimină anumite tipuri de nestaționaritate. Un trend liniar devine o constantă, $\Delta(a + bt) = b$, iar un trend pătratic necesită $\Delta^2$. Un mers aleator devine zgomot alb, $\Delta X_t = \varepsilon_t$, iar un tipar sezonier care se repetă exact la fiecare $s$ perioade este eliminat de $\Delta_s$. Un proces este **integrat de ordinul $d$** (*integrated of order d*), notat $X_t \sim I(d)$, dacă $\Delta^dX_t$ este staționar, iar $\Delta^{d-1}X_t$ nu este. Zgomotul alb este $I(0)$, iar mersul aleator este $I(1)$; testarea ordinului $d$ este subiectul capitolului 3.

## S&P 500: nivelul și diferența lui {h=3adafbf0}

Graficul aplică prima diferență unei serii reale de prețuri. Panoul de sus reprezintă logaritmul indicelui S&P 500 din 2000; panoul de jos reprezintă prima lui diferență, randamentul logaritmic zilnic, pe 6 714 zile de tranzacționare pînă la 18 septembrie 2026.

Cele două panouri se citesc prin proprietățile din secțiunile precedente. În panoul de sus, urmăriți dacă există un nivel fix la care seria revine și cît de repede sînt uitate valorile trecute. În panoul de jos, urmăriți dacă media este constantă și dacă amplitudinea randamentelor rămîne aceeași în timp; anii 2008 și 2020 merită o atenție specială.

## Interpretarea transformării S&P 500 {h=03d08d56}

Logaritmul prețului se comportă ca un mers aleator. Autocorelația lui de selecție este 0,999 la lagul 1 și încă 0,973 la lagul 50: nivelul rămîne corelat cu întregul său trecut, cum prevede @eq:rw pentru un proces care acumulează șocuri.

Randamentele logaritmice au un nivel constant: media lor este 0,025% pe zi, abaterea standard 1,21%, iar autocorelația de ordinul 1 este −0,099. O singură diferențiere a eliminat trendul stochastic și a produs o serie care oscilează în jurul unei medii fixe, cum implică mersul aleator cu derivă.

Randamentele nu sînt încă zgomot alb în sens strict. Perioadele liniștite alternează cu cele agitate, cu grupări de mișcări mari în 2008 și 2020, deci varianța este previzibilă, chiar dacă media aproape că nu este. Randamentele sînt apropiate de un zgomot alb slab; modelarea varianței lor este subiectul capitolului 5.

## Descompunerea Wold {h=72a2eb6f}

Teorema centrală de reprezentare pentru procesele staționare îi aparține lui @Wold1938. Orice proces slab staționar cu media 0 se poate scrie

$$ X_t = \sum_{j=0}^{\infty}\psi_j\varepsilon_{t-j} + \eta_t, \qquad \psi_0 = 1, \quad \sum_{j}\psi_j^2 < \infty. $$ {#eq:wold}

Aici $\varepsilon_t$ este zgomot alb și are un sens precis: este eroarea celei mai bune prognoze liniare a lui $X_t$ din propriul trecut, numită **inovație** (*innovation*). Termenul $\eta_t$ este o parte deterministă, perfect previzibilă din trecutul îndepărtat, de exemplu un ciclu fix; pentru majoritatea seriilor din practică el lipsește.

Teorema spune că orice serie staționară este o sumă ponderată de șocuri trecute, o medie mobilă de ordin infinit, MA($\infty$). Ponderea $\psi_j$ măsoară efectul asupra lui $X_{t+j}$ al unui șoc unitar din momentul $t$, deci șirul $\psi_0, \psi_1, \dots$ este **răspunsul la impuls** (*impulse response*) al procesului. Condiția $\sum_j\psi_j^2 < \infty$ face varianța finită.

Descompunerea explică de ce funcționează modelele din capitolele următoare. Ponderile $\psi_j$ formează un șir infinit și nu pot fi estimate una cîte una, dar modelele ARMA (capitolul 2) le aproximează cu puțini parametri. AR(1) din @eq:ar1ma, cu $\psi_j = \phi^j$, este cazul cel mai simplu: un singur parametru generează întregul șir.

## Ponderile Wold ale a patru procese {h=83348c01}

Graficul reprezintă ponderile Wold $\psi_j$, pentru $j = 0, \dots, 15$, ale a patru procese: AR(1) cu $\phi = 0{,}8$, AR(1) cu $\phi = -0{,}6$, MA(1) cu $\theta = 0{,}6$ și mersul aleator. Fiecare bară este efectul unui șoc după $j$ perioade.

Pentru AR(1) cu $\phi = 0{,}8$, ponderile sînt $\psi_j = 0{,}8^j$, deci $\psi_5 = 0{,}328$ și $\psi_{10} = 0{,}107$, aceleași numere ca autocorelațiile din @eq:ar1. Varianța rezultă din ponderi: $\mathrm{Var}(X_t) = \sigma^2\sum_j\psi_j^2 = 2{,}78\,\sigma^2$, adică $\sigma^2/(1 - 0{,}8^2)$. Comparați cele patru forme: cît de repede scad barele, dacă își schimbă semnul și dacă scad deloc.

## Interpretarea ponderilor Wold {h=5e341f2a}

Pentru cele trei procese staționare, ponderile tind la zero, $\psi_j \to 0$: șocurile se sting, iar varianța $\sigma^2\sum_j\psi_j^2$ este finită. Cu $\phi = -0{,}6$, efectul își alternează semnul: un șoc pozitiv azi are un efect negativ în perioada următoare, unul pozitiv după două perioade și așa mai departe, cu amplitudine descrescătoare. Pentru MA(1), efectul se oprește după o perioadă: $\psi_1 = \theta$ și $\psi_j = 0$ pentru $j \ge 2$.

Pentru mersul aleator, $\psi_j = 1$ pentru orice $j$, deci $\sum_j\psi_j^2 = \infty$: nu există reprezentare Wold și nici staționaritate. Economic, este vorba de un șoc permanent, de exemplu un șoc tehnologic care ridică definitiv PIB-ul sau o știre care schimbă o dată pentru totdeauna un preț.

Ponderile determină și autocovarianțele. Deoarece șocurile sînt necorelate, la $\mathrm{Cov}(X_t, X_{t+h})$ contribuie doar produsele șocurilor de la aceeași dată, iar

$$ \gamma(h) = \sigma^2\sum_{j}\psi_j\psi_{j+h}. $$ {#eq:woldacf}

ACF este deci un rezumat al răspunsului la impuls, iar cele două forme din secțiunea precedentă rezultă din @eq:woldacf: o singură pondere nenulă după $\psi_0$ dă o ACF care se anulează după lagul 1, iar ponderile geometrice dau o ACF geometrică.

## Ergodicitatea {h=f7dac575}

Staționaritatea spune că momentele sînt constante în timp. Ea nu spune încă dacă pot fi estimate dintr-o singură traiectorie. Trebuie deosebite două medii: **media pe ansamblu** (*ensemble mean*), $\mu = E[X_t]$, o medie pe toate traiectoriile posibile la o dată fixată, și **media în timp** (*time mean*), $\bar X_T = \frac1T\sum_{t=1}^{T}X_t$, o medie de-a lungul singurei traiectorii observate.

Un proces staționar este **ergodic pentru medie** (*ergodic for the mean*) dacă $\bar X_T \to \mu$ în medie pătratică atunci cînd $T \to \infty$. O condiție suficientă este ca autocovarianța să se stingă, $\gamma(h) \to 0$ cînd $h \to \infty$: observațiile depărtate aduc atunci informație aproape independentă. Dacă autocovarianțele sînt absolut sumabile, $\sum_h|\gamma(h)| < \infty$, este cunoscută și viteza:

$$ T\,\mathrm{Var}(\bar X_T) \to \sum_{h=-\infty}^{\infty}\gamma(h). $$ {#eq:lrv}

Membrul drept din @eq:lrv, varianța pe termen lung (*long-run variance*), înlocuiește $\gamma(0)$ în formula obișnuită a varianței unei medii; autocorelația pozitivă o mărește.

Staționaritatea singură nu este suficientă. Fie $X_t = Z + \varepsilon_t$, unde $Z \sim N(0, 1)$ este extras o singură dată și rămîne fix de-a lungul traiectoriei, iar $\varepsilon_t$ este zgomot alb independent de $Z$. Procesul este staționar, cu media 0 și $\gamma(h) = \mathrm{Var}(Z) = 1$ pentru orice $h \ne 0$, o autocovarianță care nu se stinge niciodată. De-a lungul unei traiectorii, zgomotul alb se compensează și $\bar X_T \to Z$, nu $\mu = 0$.

## Mediile în timp: proces ergodic și neergodic {h=fb27f553}

Graficul reprezintă media cumulată $\bar X_T$ de-a lungul a 6 traiectorii, pentru $T$ pînă la 2 000, pe o axă orizontală logaritmică, pentru două procese staționare cu media 0. În stînga este AR(1) cu $\phi = 0{,}7$, a cărui autocovarianță scade geometric; în dreapta este procesul $X_t = Z + \varepsilon_t$ de pe slide-ul precedent.

Fiecare linie se citește ca estimația mediei pe care ar obține-o un cercetător care dispune doar de acea istorie, pe măsură ce sosesc date noi. Valorile finale sînt între −0,13 și 0,09 în stînga, toate aproape de media adevărată 0, și între −0,95 și 1,16 în dreapta, fiecare aproape de valoarea lui $Z$ extrasă pentru acea traiectorie.

## Interpretarea ergodicității {h=deecb8f7}

În stînga, fiecare traiectorie „învață” media adevărată 0: o singură istorie lungă este la fel de informativă ca multe istorii independente. Convergența este lentă cînd corelația este puternică. Pentru un AR(1), @eq:lrv dă aproximativ

$$ \mathrm{Var}(\bar X_T) \approx \frac{\gamma(0)}{T}\cdot\frac{1 + \phi}{1 - \phi}, $$

deci cu $\phi = 0{,}7$ varianța mediei în timp este de $1{,}7/0{,}3 \approx 5{,}7$ ori mai mare decît ar fi pentru date independente cu aceeași varianță.

În dreapta, fiecare traiectorie converge la propriul $Z$, iar oricîte date dintr-o singură traiectorie nu dezvăluie că media pe ansamblu este 0. O economie cu un „tip” permanent și neobservat, de exemplu un nivel fix al calității instituțiilor sau al productivității, se comportă astfel: istoria ei lungă îi dezvăluie propriul tip, nu media peste toate economiile care ar fi putut exista.

Ergodicitatea nu poate fi testată dintr-o singură traiectorie, deoarece traiectoria singură nu poate arăta cum ar fi arătat celelalte traiectorii. Este o ipoteză făcută, explicit sau nu, de fiecare dată cînd momentele sînt estimate dintr-o singură serie.

## Recapitulare: operatorul lag, diferențierea și descompunerea Wold {h=cca05445}

Operatorul lag deplasează o serie, $L^kX_t = X_{t-k}$; diferențele sînt $\Delta = 1 - L$, iar diferențele sezoniere $\Delta_s = 1 - L^s$; un proces $I(d)$ devine staționar după $d$ diferențieri. Conform teoremei lui Wold @eq:wold, un proces staționar este un MA($\infty$) al inovațiilor sale, cu ponderi de pătrat sumabil. Ergodicitatea permite ca mediile în timp să estimeze momentele pe ansamblu și cere corelații care se sting.

## @end

Polinoamele în lag transformă modelele în algebră, iar inversarea lor @eq:ar1ma leagă autoregresiile de mediile mobile. Diferențierea elimină trendurile liniare, trendurile stochastice și sezonalitatea exactă, iar S&P 500 a arătat cum o singură diferență transformă un nivel asemănător unui mers aleator în randamente cu medie stabilă. Descompunerea Wold @eq:wold reprezintă orice proces staționar prin inovațiile lui, iar @eq:woldacf arată că ACF rezumă răspunsul la impuls. Ergodicitatea @eq:lrv este ultima legătură dintre teorie și date: ea permite ca o singură serie lungă să estimeze media și autocovarianțele, sarcina secțiunii următoare.

# ACF și PACF de selecție

Cu staționaritatea și ergodicitatea presupuse, media și ACF pot fi estimate dintr-o singură traiectorie. Secțiunea definește autocovarianța și autocorelația de selecție, deduce benzile de încredere în ipoteza de zgomot alb și le verifică prin simulare. Apoi compară ACF de selecție cu cea teoretică pentru patru procese, introduce funcția de autocorelație parțială (PACF), care izolează efectul direct al fiecărui lag, și aplică ambele instrumente randamentelor zilnice ale BET.

## Autocovarianța și autocorelația de selecție {h=270c65ba}

Din observațiile $x_1, \dots, x_T$, cu media de selecție $\bar x = \frac1T\sum_t x_t$, **autocovarianța de selecție** (*sample autocovariance*) la lagul $h$ este

$$ \hat\gamma(h) = \frac1T\sum_{t=1}^{T-h}(x_t - \bar x)(x_{t+h} - \bar x), \qquad 0 \le h < T, $$ {#eq:sacvf}

iar **ACF de selecție** (*sample ACF*) este $\hat\rho(h) = \hat\gamma(h)/\hat\gamma(0)$. Graficul lui $\hat\rho(h)$ în funcție de $h$ este **corelograma** (*correlogram*). Fiecare termen al sumei pune în pereche o observație cu cea aflată $h$ perioade mai tîrziu; există $T - h$ astfel de perechi.

Împărțitorul din @eq:sacvf este $T$, nu numărul de perechi $T - h$. Cu împărțitorul $T$, șirul $\hat\gamma(h)$ este pozitiv semidefinit, ca o autocovarianță adevărată [@BrockwellDavis2016], deci ACF estimată satisface întotdeauna @eq:nnd. Prețul este o mică deplasare spre zero la laguri mari, unde se mediază puține perechi, dar suma este împărțită tot la $T$. De aceea corelogramele se citesc de obicei doar pînă la $h \le T/4$.

Un exemplu rezolvat fixează calculele. Pentru $x = (3, 5, 4, 6, 7)$, media este $\bar x = 5$, iar abaterile sînt $(-2, 0, -1, 1, 2)$. Atunci $\hat\gamma(0) = 10/5 = 2{,}0$, $\hat\gamma(1) = (0 + 0 - 1 + 2)/5 = 0{,}2$ și $\hat\gamma(2) = (2 + 0 - 2)/5 = 0{,}0$, deci $\hat\rho(1) = 0{,}10$ și $\hat\rho(2) = 0{,}00$. Cu $T = 5$, banda de 95% pentru zgomot alb, definită în continuare, este $\pm 0{,}88$: mult prea largă pentru ca cinci observații să spună ceva despre dependență.

## Benzi de încredere pentru ACF de selecție {h=ba8a6d6f}

Benzile desenate pe corelograme provin dintr-un rezultat asimptotic al lui @Bartlett1946. Dacă $X_t$ este i.i.d. cu varianță finită, atunci, pentru $T$ mare, autocorelațiile de selecție $\hat\rho(1), \dots, \hat\rho(m)$ sînt aproximativ independente și urmează distribuția Normală cu media 0 și varianța $1/T$. O bandă de 95% este deci $\pm 1{,}96/\sqrt{T}$, iar o bară în afara ei respinge $\rho(h) = 0$ la nivelul de 5%, pentru acel lag luat separat.

Cînd procesul nu este zgomot alb, varianța lui $\hat\rho(h)$ se schimbă. Pentru un MA($q$), la lagurile de după anulare, $h > q$, formula lui Bartlett dă

$$ \mathrm{Var}(\hat\rho(h)) \approx \frac1T\Big(1 + 2\sum_{k=1}^{q}\rho(k)^2\Big). $$ {#eq:bartlett}

Acestea sînt benzile mai largi, crescătoare cu lagul, pe care plot_acf din statsmodels le desenează implicit; ele testează dacă ACF s-a anulat după lagul $h - 1$, întrebarea identificării modelului din capitolul 2.

Din caracterul „pe lag” al benzii decurg două capcane. Prima: cu 20 de laguri și nicio corelație, ne așteptăm ca aproximativ o bară să iasă din bandă din întîmplare, deci o singură depășire este o dovadă slabă. A doua: banda presupune date i.i.d.; pentru un zgomot alb slab cu varianță variabilă, de exemplu randamentele GARCH, variabilitatea de selecție reală a lui $\hat\rho(h)$ este mai mare, iar banda corectă este mai largă.

## Distribuția de selecție a ACF {h=c80cc074}

Graficul verifică prin simulare rezultatul lui Bartlett. Se extrag 10 000 de eșantioane de zgomot alb gaussian de lungime $T = 100$, pentru care banda este $\pm 1{,}96/\sqrt{100} = \pm 0{,}196$. Panoul din stînga este histograma lui $\hat\rho(1)$ pe cele 10 000 de eșantioane, cu aproximarea Normală $N(0, 1/T)$ pentru comparație.

Panoul din dreapta numără, în fiecare eșantion, cîte dintre primele 20 de autocorelații de selecție cad în afara lui $\pm 0{,}196$ și compară proporțiile simulate cu distribuția Binomial(20; 0,05), pe care ar da-o 20 de teste independente de mărime 5%. Deoarece toate autocorelațiile adevărate sînt zero, fiecare număr este un număr de alarme false. Panoul din stînga se citește după centrul și împrăștierea lui $\hat\rho(1)$, iar cel din dreapta după cît de des o corelogramă a unui zgomot pur arată cel puțin o bară în afara benzii.

## Interpretarea simulării {h=2f403728}

Pe eșantioane, $\hat\rho(1)$ are media −0,008, apropiată de media exactă în eșantion mic, $-1/T = -0{,}01$, și abaterea standard 0,098, apropiată de $1/\sqrt{T} = 0{,}1$. Aproximarea lui Bartlett este deja bună la $T = 100$. Media ușor negativă provine din estimarea lui $\bar x$ din aceleași date, ceea ce obligă abaterile să aibă suma zero.

În 50% dintre eșantioane, cel puțin una dintre cele 20 de bare se află în afara benzii, deși nu există nicio corelație. Dacă cele 20 de teste ar fi independente și fiecare ar avea exact mărimea 5%, proporția ar fi $1 - 0{,}95^{20} = 64\%$. Ea este ceva mai mică deoarece la laguri depărtate varianța lui $\hat\rho(h)$ este sub $1/T$, un efect al împărțitorului $T$ din @eq:sacvf, deci ies mai puține bare.

Regula practică este să nu supra-interpretăm barele izolate. O corelogramă se judecă după tipar, de exemplu o descreștere, o anulare sau un vîrf sezonier, sau printr-un test comun pentru mai multe laguri, subiectul secțiunii următoare.

## ACF teoretică și de selecție pentru patru procese {h=185890cd}

Graficul compară, pentru patru procese, ACF de selecție a $T = 500$ de valori simulate (bare) cu ACF teoretică (puncte). Banda este $\pm 1{,}96/\sqrt{500} = \pm 0{,}088$. Cele patru procese sînt zgomotul alb, AR(1) cu $\phi = 0{,}8$, MA(1) cu $\theta = 0{,}6$ și mersul aleator.

În fiecare panou, comparați barele cu punctele: cît de aproape sînt estimațiile de adevăr și cum diferă forma barelor de la un proces la altul. Panoul mersului aleator nu are puncte, dintr-un motiv explicat mai jos.

## Interpretarea celor patru corelograme {h=4393878b}

Pentru zgomotul alb, toate barele sînt mici. Cea mai mare, $\hat\rho(2) = 0{,}08$, se află în interiorul benzii și este zgomot. Pentru AR(1) cu $\phi = 0{,}8$, barele descresc geometric, cu $\hat\rho(1) = 0{,}81$ și $\hat\rho(2) = 0{,}65$, față de valorile teoretice 0,80 și 0,64 din @eq:ar1.

Pentru MA(1) cu $\theta = 0{,}6$ există un singur vîrf, $\hat\rho(1) = 0{,}44$, egal cu valoarea teoretică 0,44 din @eq:ma1, și nimic după el: ACF **se anulează** (*cuts off*) după lagul 1. Descreșterea și anularea sînt prima distincție folosită pentru a identifica un model din corelogramă.

Pentru mersul aleator, ACF de selecție este $\hat\rho(1) = 0{,}95$ și încă 0,55 la lagul 10, o descreștere lentă, aproape liniară. Aceasta este corelograma tipică a unei serii nestaționare. Ea nu are corespondent teoretic, deoarece $\rho(h)$ nu este definit pentru un proces a cărui autocorelație depinde de dată; barele descriu doar eșantionul și semnalează că seria trebuie diferențiată înainte ca ACF să fie interpretată.

## Autocorelația parțială {h=d5a440c6}

ACF la lagul $h$ amestecă dependența directă cu cea indirectă: într-un AR(1), $X_t$ este corelat cu $X_{t-2}$ doar pentru că ambele sînt legate prin $X_{t-1}$. **Funcția de autocorelație parțială** (PACF, *partial autocorrelation function*) elimină partea indirectă. Valoarea ei la lagul $h$, $\phi_{hh}$, este ultimul coeficient al celei mai bune prognoze liniare a lui $X_t$ din $X_{t-1}, \dots, X_{t-h}$:

$$ X_t = \phi_{h1}X_{t-1} + \dots + \phi_{hh}X_{t-h} + e_t, $$

unde $\phi_{h1}, \dots, \phi_{hh}$ sînt coeficienții regresiei pe $h$ laguri, iar $e_t$ este eroarea de prognoză. Echivalent, $\phi_{hh}$ este corelația dintre $X_t$ și $X_{t-h}$ după eliminarea efectului liniar al lagurilor intermediare.

Primele două valori au forme închise:

$$ \begin{gathered} \phi_{11} = \rho(1) \\ \phi_{22} = \frac{\rho(2) - \rho(1)^2}{1 - \rho(1)^2}. \end{gathered} $$ {#eq:pacf}

În general, PACF se calculează din ACF prin recursia Durbin–Levinson; PACF de selecție folosește $\hat\rho(h)$ în locul lui $\rho(h)$. În ipoteza de zgomot alb, benzile ei sînt aceleași ca ale ACF, $\pm 1{,}96/\sqrt{T}$.

Două exemple arată ce surprinde @eq:pacf. Pentru un AR(1), $\rho(2) = \rho(1)^2$, deci $\phi_{22} = 0$: odată cunoscut $X_{t-1}$, $X_{t-2}$ nu mai aduce nimic. Pentru un AR(2) cu $\phi_1 = 0{,}5$ și $\phi_2 = 0{,}3$, autocorelațiile sînt $\rho(1) = 0{,}714$ și $\rho(2) = 0{,}657$, iar $\phi_{22} = (0{,}657 - 0{,}510)/(1 - 0{,}510) = 0{,}300$, exact al doilea coeficient autoregresiv, $\phi_2$. PACF la lagul $p$ regăsește ultimul coeficient al unui AR($p$) și este zero după $p$.

## ACF și PACF pentru trei procese {h=487ab9af}

Graficul arată ACF de selecție (rîndul de sus) și PACF de selecție (rîndul de jos) pentru $T = 500$ de valori simulate din trei procese: un AR(1) cu $\phi = 0{,}7$, un AR(2) cu $\phi_1 = 0{,}5$ și $\phi_2 = 0{,}3$ și un MA(1) cu $\theta = 0{,}7$. Fiecare coloană este un proces, astfel că cele două funcții pot fi comparate pe aceleași date.

Coloanele se citesc după locul în care fiecare funcție se anulează și locul în care descrește. Pentru fiecare proces, una dintre cele două funcții ar trebui să arate cîteva vîrfuri clare și apoi nimic, iar cealaltă o descreștere treptată, eventual cu semne alternante.

## Interpretarea ACF și PACF {h=86d9a489}

Pentru AR(1) cu $\phi = 0{,}7$, ACF descrește, în timp ce PACF are un singur vîrf, $\hat\phi_{11} = 0{,}73$, urmat de $\hat\phi_{22} = 0{,}03$, în interiorul benzii. Pentru AR(2), PACF are două vîrfuri, cu $\hat\phi_{22} = 0{,}24$ față de valoarea teoretică 0,3, apoi $\hat\phi_{33} = 0{,}01$. Numărul vîrfurilor din PACF dă ordinul autoregresiei.

MA(1) este imaginea în oglindă. ACF are un singur vîrf, $\hat\rho(1) = 0{,}46$, iar PACF își alternează semnul și descrește, cu valorile −0,34 la lagul 2 și 0,26 la lagul 3. Un MA(1) poate fi inversat într-un AR($\infty$), deci autocorelațiile lui parțiale nu se anulează niciodată.

Aceste tipare dau regula de identificare dezvoltată în capitolul 2 [@HuangPetukhina2022]: o PACF care se anulează după lagul $p$ indică un AR($p$), o ACF care se anulează după lagul $q$ indică un MA($q$), iar două funcții care descresc amîndouă indică un model mixt ARMA.

## Date reale: trei piețe {h=b84b3357}

Instrumentele trec acum de la date simulate la date reale. Slide-urile următoare arată ACF a randamentelor logaritmice zilnice ale BET din 2000, a valorilor lor absolute și a pătratelor lor. Aceeași analiză pentru cursul de referință EUR/RON publicat de Banca Națională a României și pentru S&P 500 este cerința B2 din Seminarul 1.

Întrebarea este dacă randamentele BET sînt zgomot alb și, dacă da, de ce tip. ACF a randamentelor răspunde la prima parte; ACF a valorilor absolute și a pătratelor răspunde la a doua, deoarece un zgomot alb slab poate avea pătrate corelate, iar unul i.i.d. nu.

## BET: ACF a randamentelor, a valorilor absolute și a pătratelor {h=7a28a9b3}

Graficul arată trei corelograme ale celor 6 670 de randamente logaritmice zilnice ale BET, pentru lagurile de la 1 la 50: ale randamentelor $r_t$, ale valorilor absolute $|r_t|$ și ale pătratelor $r_t^2$. Cu $T = 6\,670$, banda pentru zgomot alb este $\pm 1{,}96/\sqrt{6670} = \pm 0{,}024$, foarte îngustă, astfel că și corelațiile mici ies din ea.

Comparați înălțimea barelor din cele trei panouri și persistența lor: dacă corelația dispare după cîteva laguri sau rămîne pozitivă pînă la lagul 50.

## Interpretarea corelogramelor BET {h=0260b286}

Pentru randamente, $\hat\rho(1) = 0{,}109$, clar în afara benzii, iar statistica Ljung–Box pentru primele 10 laguri este $Q^*(10) = 108$. Dependența este semnificativă statistic, dar mică din punct de vedere economic: pătratul autocorelației de ordinul 1, $\hat\rho(1)^2 = 1{,}2\%$, este partea din varianța randamentului de azi pe care o explică o prognoză liniară din randamentul de ieri. Autocorelația pozitivă de ordinul 1 a randamentelor unui indice este un semn tipic al unei piețe mai puțin lichide, în care unele acțiuni din indice nu se tranzacționează în fiecare zi, iar prețurile lor reacționează cu întîrziere la știri (tranzacționare nesincronă, *non-synchronous trading*).

Pentru valorile absolute, $\hat\rho(1) = 0{,}40$, iar corelația este încă 0,11 la lagul 50; pentru pătrate, $\hat\rho(1) = 0{,}33$. Mișcările mari urmează după mișcări mari, în orice direcție: acesta este volatility clustering, cu o memorie lungă a mărimii randamentelor.

Randamentele BET sînt deci apropiate de un zgomot alb slab, ca seria GARCH din secțiunea despre zgomotul alb și spre deosebire de seria Laplace i.i.d. Media lor este greu de anticipat, dar varianța lor poate fi anticipată, iar capitolul 5 construiește modelele care folosesc acest lucru.

## Recapitulare: ACF și PACF de selecție {h=b8681d21}

ACF de selecție este $\hat\rho(h) = \hat\gamma(h)/\hat\gamma(0)$, cu împărțitorul $T$, citită pînă la aproximativ lagul $T/4$. Pentru date i.i.d., $\hat\rho(h)$ urmează aproximativ distribuția $N(0, 1/T)$, de unde banda $\pm 1{,}96/\sqrt{T}$, iar aproximativ o bară din 20 o depășește din întîmplare. PACF măsoară efectul direct al lagului $h$: un AR($p$) se anulează în PACF, un MA($q$) în ACF. Randamentele zilnice au o dependență liniară slabă și o dependență neliniară puternică.

## @end

ACF de selecție @eq:sacvf estimează autocorelația unei singure traiectorii, iar rezultatul lui Bartlett îi dă variabilitatea de selecție: $1/T$ pentru date i.i.d., mai mult în cazul dependenței MA @eq:bartlett sau al varianței variabile. Simularea a arătat că jumătate dintre corelogramele de 20 de laguri ale unui zgomot pur conțin o alarmă falsă. Compararea ACF cu PACF @eq:pacf deosebește dependența autoregresivă de cea de tip medie mobilă, iar corelogramele BET au arătat profilul caracteristic al randamentelor zilnice: corelație mică în randamente, corelație puternică și persistentă în mărimea lor. Deoarece barele izolate nu sînt de încredere, secțiunea următoare trece la testele care combină mai multe laguri într-o singură statistică.

# Teste portmanteau: Box–Pierce și Ljung–Box

O corelogramă testează fiecare lag separat, ceea ce înmulțește alarmele false și irosește informația din multe corelații mici. Testele portmanteau combină primele $m$ autocorelații de selecție într-o singură statistică. Secțiunea definește statisticile Box–Pierce și Ljung–Box, calculează de mînă un exemplu, compară prin simulare comportamentul lor în eșantioane mici și aplică testul Ljung–Box seriilor reale ale capitolului.

## Testarea simultană a mai multor autocorelații {h=c20fb8cb}

Ipoteza nulă este că seria este zgomot alb pînă la lagul $m$, $H_0: \rho(1) = \dots = \rho(m) = 0$, cu alternativa $H_1$ că cel puțin un $\rho(h) \ne 0$. Un test de acest tip se numește **portmanteau**, după geanta de voiaj care poartă multe obiecte: o singură statistică „poartă” $m$ autocorelații.

**Statistica Box–Pierce** a lui @BoxPierce1970 este $Q(m) = T\sum_{h=1}^{m}\hat\rho(h)^2$. Distribuția ei rezultă din rezultatul lui Bartlett: în ipoteza $H_0$, fiecare $\sqrt{T}\hat\rho(h)$ urmează aproximativ distribuția Normală standard, iar cele $m$ variabile sînt aproximativ independente, deci suma pătratelor lor urmează aproximativ distribuția hi-pătrat cu $m$ grade de libertate. **Statistica Ljung–Box** a lui @LjungBox1978 reponderează termenii:

$$ Q^*(m) = T(T + 2)\sum_{h=1}^{m}\frac{\hat\rho(h)^2}{T - h} \ \overset{a}{\sim}\ \chi^2(m). $$ {#eq:lb}

Ponderea $(T + 2)/(T - h)$ este mai mare decît 1 și crește cu $h$. Ea corectează faptul, observat în secțiunea precedentă, că varianța lui $\hat\rho(h)$ este mai mică decît $1/T$ la laguri depărtate; ambele statistici au aceeași limită $\chi^2(m)$.

Regula de decizie compară $Q^*(m)$ cu $\chi^2_{0{,}95}(m)$, cuantila de 95% a distribuției hi-pătrat cu $m$ grade de libertate, și respinge $H_0$ la 5% dacă statistica este mai mare; de exemplu, $\chi^2_{0{,}95}(10) = 18{,}31$. Echivalent, p-value-ul, probabilitatea în ipoteza $H_0$ de a obține o statistică cel puțin la fel de mare ca cea observată, se compară cu 0,05. Cînd testul se aplică reziduurilor unui model ARMA($p, q$) estimat, distribuția de referință devine $\chi^2(m - p - q)$, deoarece estimarea consumă $p + q$ grade de libertate (capitolul 2).

## Exemplu rezolvat: statistica Ljung–Box {h=dcb4191c}

Să luăm $T = 100$ de observații, cu autocorelațiile de selecție $\hat\rho(1) = 0{,}25$, $\hat\rho(2) = 0{,}12$ și $\hat\rho(3) = -0{,}08$. Banda este $\pm 0{,}196$, deci doar $\hat\rho(1)$ se află în afara ei. Pătratele autocorelațiilor sînt 0,0625, 0,0144 și 0,0064, deci statistica Box–Pierce este $Q(3) = 100 \cdot 0{,}0833 = 8{,}33$, iar statistica Ljung–Box, cu ponderile din @eq:lb, este

$$ \begin{gathered} Q^*(3) = 100 \cdot 102\,\Big(\frac{0{,}0625}{99} + \frac{0{,}0144}{98} \\ + \frac{0{,}0064}{97}\Big) = 8{,}61. \end{gathered} $$

Valoarea critică este $\chi^2_{0{,}95}(3) = 7{,}81$, deci $H_0$ este respinsă; p-value-ul lui $Q^*(3)$ este 0,035. Seria nu este zgomot alb, iar corelograma arată că dependența este concentrată la lagul 1. Testul spune doar că există o anumită corelație; ACF spune unde.

Alegerea lui $m$ presupune un compromis. Un $m$ prea mic ratează dependența la laguri depărtate, de exemplu corelația sezonieră; un $m$ prea mare diluează cîteva corelații puternice printre multe nule și scade puterea testului. Alegerile uzuale sînt $m = 10$ pentru date nesezoniere și $m = 2s$ pentru date sezoniere cu perioada $s$ [@HyndmanAthanasopoulos2021].

## Box–Pierce și Ljung–Box în eșantioane mici {h=ef96d5e3}

Graficul prezintă o simulare a mărimii testelor (*size*): proporția din 5000 de eșantioane de zgomot alb gaussian în care fiecare test respinge ipoteza nulă adevărată la nivelul de 5%. Un test cu mărimea corectă respinge o ipoteză nulă adevărată în 5% dintre eșantioane; o proporție mai mare înseamnă alarme false, una mai mică înseamnă un test prea conservator, care pierde din putere.

Barele sînt grupate în șase perechi, pentru volumele de eșantion $T = 50$, 100 și 500 și pentru $m = 10$ și $m = 20$ de laguri; linia întreruptă marchează nivelul nominal de 5%. Fiecare pereche de bare se citește față de această linie: care statistică este mai aproape de ea și cum depinde abaterea de $T$ și de cît de mare este $m$ față de $T$.

## Interpretarea simulării mărimii testelor {h=ce6b4ab5}

Cu $T = 50$ și $m = 20$, testul Box–Pierce respinge în 2,4% dintre eșantioane, iar testul Ljung–Box în 9,4%. Box–Pierce respinge prea rar: cu puține observații, statistica lui este prea mică, deci poate rata o dependență reală. Ljung–Box greșește în sens opus cînd $m$ este mare față de $T$, deoarece ponderile lui $(T + 2)/(T - h)$ devin mari la ultimele laguri.

Cu $T = 500$ și același $m = 20$, ratele de respingere sînt 5,0% pentru Box–Pierce și 5,9% pentru Ljung–Box, ambele aproape de nivelul nominal de 5%. Cu suficiente date, cele două statistici concordă, iar aproximarea hi-pătrat este de încredere.

În practică se folosește testul Ljung–Box, care este și cel implicit în programele statistice. Numărul de laguri trebuie să rămînă mult sub volumul eșantionului, de exemplu $m \le T/5$, ceea ce pentru $T = 50$ înseamnă cel mult 10 laguri.

## Statisticile Ljung–Box pentru serii reale {h=1db03112}

Graficul reprezintă, pentru șase serii reale, statistica Ljung–Box $Q^*(m)$ în funcție de $m = 1, \dots, 30$: randamentele zilnice ale S&P 500, ale BET și ale cursului EUR/RON, pătratele randamentelor S&P 500 și BET și creșterea trimestrială a PIB-ului României. Statisticile sînt desenate pe o scară logaritmică, împreună cu valoarea critică de 5%, $\chi^2_{0{,}95}(m)$, ca linie întreruptă. O serie este compatibilă cu zgomotul alb pînă la lagul $m$ cînd curba ei se află sub linia întreruptă la acel $m$.

Scara logaritmică este necesară deoarece statisticile diferă de la o serie la alta prin ordine de mărime. Distanța verticală dintre fiecare curbă și linia întreruptă se citește ca tăria dovezilor împotriva zgomotului alb, iar panta fiecărei curbe arată dacă dependența este concentrată la primele laguri sau răspîndită pe multe.

## Interpretarea statisticilor Ljung–Box {h=9c361414}

Tabelul dă, pentru fiecare serie, volumul eșantionului $T$, autocorelațiile de selecție la lagurile 1 și 4, statistica $Q^*(10)$ și p-value-ul ei. Toate seriile resping ipoteza de zgomot alb, cu p-value-uri sub 0,001. Pentru randamentele zilnice S&P 500, $Q^*(10) = 96{,}3$, cu $\hat\rho(1) = -0{,}10$; pentru randamentele BET, $Q^*(10) = 108{,}0$, cu $\hat\rho(1) = 0{,}11$; pentru randamentele EUR/RON, $Q^*(10) = 221{,}7$, cu $\hat\rho(1) = 0{,}17$.

Pentru pătratele randamentelor, statisticile sînt de zeci de ori mai mari decît pentru randamentele înseși: 6132,8 pentru S&P 500, față de 96,3 pentru randamentele lui, și 2440,0 pentru BET, față de 108,0. Dependența randamentelor financiare zilnice stă mai ales în mărimea lor, nu în semnul lor, după cum au arătat deja corelogramele BET.

O respingere spune doar că există o anumită corelație, nu unde se află sau ce formă are. ACF o localizează. Pentru creșterea trimestrială a PIB-ului României față de trimestrul anterior, $\hat\rho(4) = 0{,}96$ și $Q^*(10) = 364{,}8$: dependența este sezonul, la lagul 4. Pentru creșterea anuală, $\hat\rho(1) = 0{,}63$ și $\hat\rho(4) = 0{,}03$, cu $Q^*(10) = 75{,}0$: corelația sezonieră a dispărut, iar dependența rămasă este persistența ciclului economic. Inflația lunară IAPC are $\hat\rho(1) = 0{,}40$ și $Q^*(10) = 126{,}7$. Nilul ($\hat\rho(1) = 0{,}50$, $Q^*(10) = 88{,}1$) și petele solare ($\hat\rho(1) = 0{,}82$, $\hat\rho(4) = -0{,}28$, $Q^*(10) = 627{,}4$) sînt discutate în secțiunea despre cele două serii clasice.

## Recapitulare: teste portmanteau {h=e2fd50f4}

Statistica Box–Pierce $Q(m)$ și statistica Ljung–Box $Q^*(m)$ însumează pătratele autocorelațiilor de selecție pentru primele $m$ laguri și urmează ambele aproximativ distribuția $\chi^2(m)$ pentru zgomot alb. Ljung–Box are o mărime mai bună în eșantioane mici, cu condiția ca $m$ să rămînă mult sub $T$. O respingere este un semnal pentru modelarea dependenței, nu un model; iar în prezența efectelor GARCH, testul aplicat randamentelor respinge prea des.

## @end

Testele portmanteau transformă o corelogramă într-o singură decizie. Statistica Ljung–Box @eq:lb are o mărime apropiată de 5% în eșantioane de cîteva sute de observații, în timp ce statistica Box–Pierce respinge prea rar în eșantioane mici. Pe seriile reale, toate testele resping ipoteza de zgomot alb, dar din motive diferite: sezonalitatea PIB-ului trimestrial, persistența inflației, dependența liniară mică a randamentelor și dependența puternică a pătratelor lor. Mai multe dintre aceste serii resping deoarece nu au fost încă transformate corespunzător; secțiunea următoare arată cum logaritmul, diferențierea și transformarea Box–Cox pregătesc o serie pentru analiză.

# Transformări

Puține serii economice sînt staționare în forma în care sînt publicate. Seriile pozitive cu creștere exponențială au varianțe care cresc odată cu nivelul, iar trendurile și tiparele sezoniere deplasează media. Secțiunea explică de ce și în ce ordine se transformă o serie, aplică logaritmul și diferențele PIB-ului real al României, alege o transformare Box–Cox pentru PIB-ul nominal al României și arată ce strică supradiferențierea (*over-differencing*), adică diferențierea unei serii care nu avea nevoie de ea.

## Motivele transformării unei serii {h=9be9d02d}

O transformare are trei scopuri posibile. Primul este stabilizarea varianței, de exemplu cînd oscilațiile sezoniere cresc odată cu nivelul seriei. Al doilea este transformarea creșterii exponențiale într-un trend liniar. Al treilea este eliminarea unui trend stochastic sau a unui tipar sezonier, ceea ce face diferențierea. Primele două privesc scara seriei, al treilea dinamica ei.

Logaritmul servește primelor două scopuri. Diferența lui este rata de creștere în termeni continui, iar pentru variații mici $100\,\Delta\ln Y_t$ este aproximativ variația procentuală: $\ln 1{,}03 = 0{,}0296$, circa 3%. Aproximarea nu mai funcționează pentru variații mari: $\ln 1{,}30 = 0{,}262$ pentru o creștere de 30% și $\ln 0{,}70 = -0{,}357$ pentru o scădere de 30%. Variația logaritmică are o proprietate pe care procentele nu o au: variațiile logaritmice se adună în timp, deci variația logaritmică pe un an este suma celor douăsprezece variații logaritmice lunare.

Pentru o serie pozitivă, cu trend și sezonalitate, ordinea operațiilor contează: întîi logaritmul, care stabilizează varianța, apoi diferențele, care elimină trendul și sezonul. Logaritmul unei diferențe nu este definit cînd diferența este negativă, iar diferențierea făcută prima ar lăsa varianța să crească odată cu nivelul.

## PIB-ul real al României: logaritm, creștere trimestrială și anuală {h=bca97b70}

Graficul aplică aceste transformări PIB-ului real al României: date trimestriale, neajustate sezonier, în prețuri din 2010 (Eurostat), din T1 1995 pînă în T2 2026. Rîndul de sus arată trei serii: logaritmul nivelului, $\ln Y_t$, creșterea trimestrială, $\Delta\ln Y_t$, și creșterea anuală, $\Delta_4\ln Y_t$, care compară fiecare trimestru cu același trimestru din anul anterior. Rîndul de jos arată ACF de selecție a fiecăreia.

Fiecare coloană se citește după cele două trăsături pe care diferențierea trebuie să le elimine: trendul, care apare în corelogramă ca o descreștere lentă, și sezonul, care apare ca vîrfuri la lagurile 4, 8, 12 și așa mai departe. O transformare bună lasă o corelogramă care se stinge după cîteva laguri.

## Interpretarea transformărilor PIB {h=7d34e7e5}

Logaritmul nivelului, $\ln Y_t$, are o ACF de selecție de 0,68 la lagul 1 și 0,87 la lagul 8, cu vîrfuri la fiecare 4 trimestre. Trendul face mari toate autocorelațiile, iar sezonul le face și mai mari pe cele de la multiplii lui 4, astfel că seria este la fel de puternic corelată cu valoarea de acum doi ani ca și cu trimestrul precedent.

Creșterea trimestrială, $\Delta\ln Y_t$, nu mai are trend, dar sezonul domină: $\hat\rho(4) = 0{,}96$ și $\hat\rho(2) = -0{,}49$, iar abaterea standard este 27,5%. O rată trimestru față de trimestru calculată din date neajustate măsoară mai ales tiparul sezonier, diferența regulată dintre, de exemplu, un trimestru de iarnă și unul de vară, nu știrile economice.

Creșterea anuală, $\Delta_4\ln Y_t$, are media 2,7%, iar ACF ei se stinge după 2–3 trimestre, cu $\hat\rho(1) = 0{,}63$ și $\hat\rho(4) = 0{,}03$. Seria este apropiată de staționaritate, cu persistența ciclului economic. Valorile ei extreme sînt evenimente economice, nu artefacte sezoniere: −9,9% în T2 2020 și −2,0% în ultimul trimestru al eșantionului.

## Transformarea Box–Cox {h=da31f33d}

Logaritmul face parte dintr-o familie. @BoxCox1964 au definit, pentru $y > 0$,

$$ w = \begin{cases} \dfrac{y^\lambda - 1}{\lambda}, & \lambda \ne 0, \\[4pt] \ln y, & \lambda = 0. \end{cases} $$ {#eq:boxcox}

Parametrul $\lambda$ controlează tăria transformării: $\lambda = 1$ lasă forma neschimbată (doar deplasează seria cu 1), $\lambda = 0{,}5$ este în esență radicalul, $\lambda = 0$ este logaritmul, iar $\lambda = -1$ inversa. Familia este continuă în $\lambda$, deoarece $(y^\lambda - 1)/\lambda \to \ln y$ cînd $\lambda \to 0$, motiv pentru care logaritmul ia locul formulei în zero.

Un exemplu rezolvat arată cum comprimă $\lambda$ valorile mari. Pentru $y = 100$ și $y = 400$, $\lambda = 0{,}5$ dă $w = 18$ și $w = 38$, deci un raport de patru devine aproximativ un raport de doi; $\lambda = 0$ dă $w = 4{,}605$ și $w = 5{,}991$. Cu cît $\lambda$ este mai mic, cu atît valorile mari sînt comprimate mai mult față de cele mici.

Parametrul $\lambda$ se alege prin metoda lui @Guerrero1993, ca în @HyndmanAthanasopoulos2021. Seria se împarte în blocuri, de exemplu ani, iar $\lambda$ se alege astfel încît raportul $\mathrm{sd}_i/\mathrm{medie}_i^{1-\lambda}$, unde $\mathrm{sd}_i$ și $\mathrm{medie}_i$ sînt abaterea standard și media blocului $i$, să fie cît mai constant de la un bloc la altul. Prognozele se fac pentru $w$ și se transformă înapoi cu $y = (\lambda w + 1)^{1/\lambda}$; deoarece transformarea este neliniară, prognoza transformată înapoi este mediana distribuției de prognoză, nu media ei.

## Box–Cox pentru PIB-ul nominal al României {h=5c699c1f}

Graficul aplică metoda lui Guerrero PIB-ului nominal trimestrial al României, neajustat sezonier, care crește de la 6,4 la 95,2 mld. EUR de-a lungul eșantionului. Blocurile sînt ani de cîte 4 trimestre. Primele două panouri arată seria înainte și după transformare; al treilea reprezintă criteriul, variabilitatea lui $\mathrm{sd}_i/\mathrm{medie}_i^{1-\lambda}$ de la un an la altul, în funcție de $\lambda$, iar minimul lui dă alegerea lui Guerrero.

Curba criteriului se citește după minim și după cît de plată este în jurul minimului: o curbă plată înseamnă că valorile apropiate ale lui $\lambda$, inclusiv cele rotunde, sînt aproape la fel de bune.

## Interpretarea alegerii Box–Cox {h=84ebd04f}

Estimația lui Guerrero este $\hat\lambda = 0{,}02$, practic logaritmul. Criteriul este 0,243 la $\hat\lambda$, tot 0,243 la $\lambda = 0$ și 0,68 la $\lambda = 1$, seria netransformată. Deoarece $\lambda = 0$ este la fel de bun ca estimația, se alege valoarea rotundă și ușor de interpretat: logaritmul, ale cărui diferențe sînt rate de creștere.

Logaritmul se potrivește deoarece oscilația sezonieră este o proporție stabilă din nivel: circa 39% în primii cinci ani ai eșantionului și 41% în ultimii cinci. Cînd un tipar sezonier este proporțional cu nivelul, el este multiplicativ, $Y_t = T_t \times S_t$, iar logaritmul îl face aditiv, $\ln Y_t = \ln T_t + \ln S_t$, cu o oscilație de mărime constantă.

Box–Cox corectează doar varianța. Seria transformată are în continuare trend și sezon, care trebuie eliminate prin diferențiere, în ordinea stabilită la începutul secțiunii.

## Supradiferențierea {h=f9cbd8f3}

Diferențierea elimină nestaționaritatea, dar aplicată unei serii deja staționare creează o dependență proprie. Graficul arată ACF de selecție a 500 de valori de zgomot alb în stînga și ACF de selecție a primei lui diferențe în dreapta.

Zgomotul alb nu are corelație, deci barele din stînga sînt mici, iar una sau două care ating banda sînt depășirile întîmplătoare așteptate pe 10 laguri. În dreapta, punctele marchează ACF teoretică a seriei diferențiate. Comparați panoul din dreapta cu corelograma MA(1) din secțiunea despre ACF de selecție și urmăriți semnul și mărimea primei bare.

## Interpretarea supradiferențierii {h=d67fb639}

Diferența unui zgomot alb, $\Delta\varepsilon_t = \varepsilon_t - \varepsilon_{t-1}$, este un MA(1) cu $\theta = -1$, deci, conform @eq:ma1, prima ei autocorelație este $\rho(1) = -1/(1 + 1) = -1/2$; în eșantionul simulat, $\hat\rho(1) = -0{,}54$. Varianța se dublează, deoarece $\mathrm{Var}(\varepsilon_t - \varepsilon_{t-1}) = 2\sigma^2$: de la 0,95 în eșantion la 2,01 după diferențiere. Diferențierea unei serii staționare adaugă zgomot și creează o corelație care nu exista în date.

Un $\hat\rho(1)$ puternic negativ, apropiat de −0,5, imediat după diferențiere este semnul de diagnostic că seria nu avea nevoie de diferență. La fel se întîmplă cu o serie staționară în jurul trendului, deoarece $\Delta(a + bt + \varepsilon_t) = b + \varepsilon_t - \varepsilon_{t-1}$: eliminarea unui trend determinist prin diferențiere lasă același MA(1) cu $\theta = -1$.

Regula este să diferențiem cît este nevoie, nu mai mult. Capitolul 3 dă testele formale pentru numărul de diferențieri de care are nevoie o serie.

## Recapitulare: transformări {h=baafb235}

Pentru seriile pozitive ale căror oscilații cresc cu nivelul, se aplică întîi logaritmul, apoi diferențele. Mărimea $100\,\Delta\ln Y_t$ este creșterea în procente; $\Delta_4$ și $\Delta_{12}$ dau creșterea anuală pentru date trimestriale și lunare. Parametrul Box–Cox se alege prin metoda lui Guerrero, iar pentru PIB-ul României $\hat\lambda \approx 0$, adică logaritmul. Supradiferențierea creează $\rho(1) \approx -0{,}5$ și mărește varianța.

## @end

Transformările pregătesc o serie pentru modelele staționare ale cursului. Logaritmul, o transformare Box–Cox @eq:boxcox cu $\lambda = 0$, stabilizează varianța cînd oscilațiile sezoniere sînt proporționale cu nivelul, ca pentru PIB-ul nominal al României; diferența sezonieră transformă apoi PIB-ul real al României într-o creștere anuală cu o corelogramă care se stinge după cîteva trimestre. Diferențierea are un cost cînd nu este necesară: produce un MA(1) cu $\rho(1) = -0{,}5$ și o varianță dublă. Ultima secțiune principală revine la istorie, la două serii care au modelat felul în care se citește ACF.

# Două serii clasice

Instrumentele acestui capitol s-au născut dintr-o dezbatere din anii 1920 și 1930 despre originea ciclurilor regulate din date. Secțiunea prezintă răspunsurile lui Yule și Slutzky și citește apoi ACF a două serii clasice, debitul anual al Nilului și numărul anual de pete solare. Fiecare oferă o lecție despre ce poate și ce nu poate însemna o corelogramă care scade lent sau oscilează.

## Studiu de caz: Yule (1927) și Slutzky (1937) {h=ce0b884c}

În anii 1920, ciclurile regulate ale petelor solare și ale activității economice erau explicate de regulă prin componente periodice ascunse: seria era tratată ca o sumă de sinusoide plus zgomot, iar periodograma servea la găsirea perioadelor lor. În această viziune, un ciclu în date cerea o cauză periodică.

@Yule1927 a propus un alt mecanism pentru numărul de pete solare. El a lăsat fiecare valoare să depindă de propriile două valori anterioare plus o perturbație aleatoare,

$$ X_t = \phi_1X_{t-1} + \phi_2X_{t-2} + \varepsilon_t, $$ {#eq:yule}

prima autoregresie. Imaginea lui era un pendul amortizat lovit de șocuri aleatoare: lăsat în pace, pendulul s-ar opri, dar șocurile îl țin în mișcare, iar dinamica proprie dă mișcării o perioadă caracteristică. Rezultatul este un ciclu fără nicio sinusoidă, neregulat ca amplitudine și fază, dar cu o lungime recognoscibilă.

@Slutzky1937 a ajuns la o concluzie înrudită pornind din cealaltă parte. A luat numere pur aleatoare, dintr-o serie de loterie, a format sume mobile ale lor și a obținut unde netede care semănau cu ciclurile economice. Acesta este procesul de medie mobilă, iar el vine cu un avertisment: ciclurile din date pot fi create de medierea care produce datele. Cele două viziuni au fondat împreună abordarea modernă, în care procesele staționare sînt construite din zgomot alb; @Wold1938 le-a unificat în descompunerea din secțiunea despre operatorul lag, iar @HuangPetukhina2022, capitolul 3, dezvoltă modelele care rezultă.

## Nilul și petele solare {h=8337e8d0}

Prima serie este debitul anual al Nilului la Aswan, între 1871 și 1970, 100 de observații. Nivelul fluviului se înregistrează la Nilometrul de pe insula Roda din Cairo de peste o mie de ani. Seria de la Aswan are o scădere a debitului mediu în jurul anului 1898, studiată ca problemă de punct de schimbare (*change-point*) de @Cobb1978.

A doua serie este numărul anual de pete solare între 1700 și 2008, 309 observații, seria care l-a condus pe @Yule1927 la autoregresie. Ambele serii sînt incluse în biblioteca statsmodels, și fiecare oferă o lecție despre interpretarea ACF.

## Nilul și petele solare: seriile și ACF {h=0d79bcbf}

Jumătatea din stînga a graficului arată debitul Nilului, în $10^8$ metri cubi, cu media înainte și după 1898 desenată prin linii întrerupte, iar dedesubt două corelograme pînă la lagul 20: a debitului brut și a debitului după scăderea mediei perioadei căreia îi aparține fiecare observație. Jumătatea din dreapta arată numărul de pete solare și ACF lui pînă la lagul 40.

Pentru Nil, comparați cele două corelograme: cît din dependența seriei brute supraviețuiește după eliminarea celor două medii. Pentru petele solare, urmăriți forma ACF, lagurile la care atinge minimele și maximele și dacă oscilația se micșorează cu lagul.

## Interpretarea celor două serii clasice {h=9aafca72}

Seria brută a Nilului are $\hat\rho(1) = 0{,}50$ și o descreștere lentă, iar statistica Ljung–Box este $Q^*(10) = 88{,}1$, cu un p-value sub 0,001. Citită singură, această corelogramă sugerează o persistență puternică. După scăderea celor două medii, 1098 înainte de ruptură și 850 după ea, prima autocorelație scade la $\hat\rho(1) = 0{,}16$, iar $Q^*(10) = 13{,}0$, cu un p-value de 0,225, deci seria rămasă nu este separată de zgomotul alb de acest test. O ACF care scade lent poate proveni deci dintr-o **ruptură** a mediei, nu din persistență sau dintr-o rădăcină unitară.

ACF a petelor solare este o undă amortizată, cu un minim de −0,43 la lagul 5 și un vîrf de 0,66 la lagul 10. Acesta este ciclul solar de circa 11 ani, care apare ca o oscilație în corelogramă. Seria este staționară și are un ciclu, iar @Yule1927 a modelat-o cu procesul AR(2) din @eq:yule: cu coeficienți potriviți, autoregresia produce exact o astfel de oscilație amortizată în ACF.

Lecția generală este că ACF se citește împreună cu graficul seriei și cu ceea ce se știe despre modul în care au fost produse datele. Aceeași formă a corelogramei poate proveni din mecanisme diferite, iar doar contextul le deosebește.

## @end

Autoregresia lui Yule și sumele mobile ale lui Slutzky au arătat că ciclurile aparent regulate pot apărea din zgomot alb filtrat printr-o dinamică simplă, iar aceasta este fundația modelelor liniare din restul cursului. Nilul a arătat că o ruptură structurală poate imita persistența în corelogramă, iar petele solare au arătat oscilația amortizată pe care o produce un AR(2) cu ciclu. Ambele lecții duc în aceeași direcție: o corelogramă este o dovadă despre un mecanism doar împreună cu graficul seriei și cu cunoașterea datelor.

# Contribuția posibilă a AI

Asistenții AI pot accelera părțile de rutină ale analizelor din acest capitol: scrierea codului, explicarea definițiilor și repetarea diagnosticelor pe multe serii. Rezultatele lor trebuie verificate față de definițiile capitolului, deoarece erorile cele mai frecvente sînt convenții care par plauzibile și sînt greșite. Secțiunea arată unde ajută astfel de instrumente și ce verificări cer rezultatele lor.

## Contribuția posibilă a AI {h=3fa3f290}

Trei utilizări se potrivesc materialului acestui capitol. Un instrument AI poate scrie o primă versiune a unui script care descarcă o serie, o reprezintă grafic împreună cu ACF și PACF și aplică testul Ljung–Box pentru mai multe valori ale lui $m$. Poate da o a doua explicație a unei definiții, de exemplu diferența dintre staționaritatea slabă și cea strictă, sau a motivului pentru care o corelogramă are o anumită formă. Și poate repeta aceleași diagnostice pe multe serii deodată, de exemplu toate acțiunile BVB sau toate ratele inflației din UE, o sarcină anevoioasă de mînă.

Exemplul de prompt de pe slide este precis în privința sursei datelor (Eurostat), a seriei (IAPC lunar al României), a transformării (de 100 de ori diferența logaritmică lunară, din 2005), a graficelor (ACF și PACF pînă la lagul 36) și a testelor (Ljung–Box pentru $m = 12$ și $m = 24$). Precizia contează: fiecare alegere corespunde unei decizii discutate în acest capitol, iar un prompt vag lasă aceste decizii pe seama instrumentului. Alegerea $m = 12$ și $m = 24$ urmează regula $m = 2s$ pentru date lunare, cu $s = 12$.

## Verificări necesare {h=e63e0fe9}

Mai multe convenții diferă între programe și între manuale, iar codul generat le poate amesteca. Verificați împărțitorul lui $\hat\gamma(h)$, care trebuie să fie $T$, nu $T - h$, și dacă lagul 0, mereu egal cu 1, este inclus în grafic. Verificați ce benzi sînt desenate: banda de zgomot alb $\pm 1{,}96/\sqrt{T}$ testează fiecare lag față de zero, în timp ce benzile Bartlett pentru MA @eq:bartlett testează dacă ACF s-a anulat; ele răspund la întrebări diferite.

Verificați că testul Ljung–Box se aplică transformării staționare a seriei, nu nivelului ei, unde respinge trivial, și că pentru reziduurile unui model ARMA($p, q$) numărul gradelor de libertate este $m - p - q$. Verificați că „semnificativ” nu este citit ca „mare” sau „profitabil”: cu 6 670 de observații BET, o autocorelație de 0,11 este puternic semnificativă și explică circa 1% din varianță.

În sfîrșit, verificați datele și sursele: datele calendaristice și frecvența seriilor, dacă sînt ajustate sezonier, baza unui indice și dacă ratele sînt în procente sau în zecimale. Fiecare referință citată de un instrument AI trebuie să existe; verificați-i DOI-ul.

## @end

Instrumentele AI ajută cel mai mult la pașii mecanici și cel mai puțin la alegerile care cer înțelegere: împărțitorul, benzile, transformarea și gradele de libertate. Fiecare dintre verificările de mai sus corespunde unei definiții din acest capitol, motiv pentru care definițiile rămîn necesare chiar și atunci cînd codul este scris de un instrument.

# Rezumat

Rezumatul adună ideile și formulele capitolului și se încheie cu trei întrebări de autoevaluare. Fiecare punct trimite la secțiunea în care a fost dedus.

## Idei de reținut {h=9f5adac1}

O serie de timp este o traiectorie a unui proces stochastic, iar inferența dintr-o singură traiectorie cere două ipoteze: staționaritatea, pentru ca observațiile de la date diferite să informeze despre aceiași parametri, și ergodicitatea, pentru ca mediile în timp să conveargă la momentele pe ansamblu. Staționaritatea slabă cere o medie constantă și o autocovarianță care depinde doar de lag.

Zgomotul alb, în variantele slabă, i.i.d. și gaussiană, este piesa de bază a modelelor cursului; mersul aleator este suma lui cumulată și nu este staționar. ACF și PACF de selecție, citite față de benzile $\pm 1{,}96/\sqrt{T}$, descriu dependența unei serii lag cu lag, iar testul Ljung–Box examinează simultan mai multe laguri.

Seriile reale au nevoie de transformări înainte ca aceste instrumente să poată fi aplicate: logaritmul sau o transformare Box–Cox pentru varianță, diferențele pentru trend și sezon. Randamentele zilnice sînt principalul exemplu al distincției dintre zgomotul alb slab și cel tare: sînt aproape necorelate, în timp ce pătratele lor sînt puternic corelate.

## Formule de reținut {h=12e9e21b}

Tabelul adună formulele capitolului. Autocovarianța și ACF ale unui proces staționar, @eq:acf, sînt obiectele pe care restul cursului le estimează și le modelează. Momentele procesului MA(1), @eq:ma1, și ale procesului AR(1), @eq:ar1, sînt cele două forme de referință: o anulare după lagul 1 și o descreștere geometrică. Momentele mersului aleator, @eq:rw, arată o varianță care crește liniar în timp și o covarianță $\min(t, s)\,\sigma^2$ care depinde de date.

Descompunerea Wold, @eq:wold, reprezintă orice proces staționar prin inovațiile lui. ACF de selecție, @eq:sacvf, folosește împărțitorul $T$ și se citește față de banda $\pm 1{,}96/\sqrt{T}$; primele două valori ale PACF, @eq:pacf, separă dependența directă de cea indirectă. Statistica Ljung–Box, @eq:lb, testează simultan primele $m$ autocorelații, iar familia Box–Cox, @eq:boxcox, conține logaritmul drept cazul $\lambda = 0$.

## Autoevaluare {h=fd2ff22d}

Prima întrebare aplică @eq:ma1 cu un coeficient negativ. Pentru $X_t = \varepsilon_t - 0{,}5\varepsilon_{t-1}$ cu $\sigma^2 = 4$, varianța este $\gamma(0) = 4 \times (1 + 0{,}25) = 4 \times 1{,}25 = 5$, iar prima autocorelație este $\rho(1) = -0{,}5/1{,}25 = -0{,}4$. Semnul lui $\theta$ se transmite lui $\rho(1)$, iar limita $|\rho(1)| \le 0{,}5$ este respectată.

A doua întrebare privește testarea multiplă într-o corelogramă. Cu $T = 400$, banda este $\pm 1{,}96/\sqrt{400} = \pm 0{,}098$, iar pentru 40 de laguri ale unei serii necorelate ne așteptăm ca aproximativ 2 bare să iasă din bandă din întîmplare. Două bare aflate chiar în afara benzii nu sînt deci o dovadă de corelație; testul Ljung–Box este verificarea comună potrivită.

A treia întrebare este un diagnostic. Dacă, după diferențiere, $\hat\rho(1) = -0{,}48$ și niciun alt lag nu este semnificativ, corelograma corespunde procesului MA(1) cu $\theta = -1$ pe care îl produce supradiferențierea. Seria inițială era probabil deja staționară. Capitolul următor, capitolul 2, construiește pe aceste fundații modelele ARMA: identificarea cu ACF și PACF, estimarea, diagnosticul și prognoza.

## @end

Capitolul a răspuns întrebării de la început în doi pași. În ipotezele de staționaritate și ergodicitate, o singură traiectorie lungă dezvăluie media și autocovarianțele procesului care a generat-o; ACF și PACF de selecție și testul Ljung–Box transformă această posibilitate în dovezi practice. Înainte ca aceste instrumente să poată fi aplicate, o serie reală trebuie adusă aproape de staționaritate, cu logaritmul pentru varianță și cu diferențele pentru trend și sezon, fără mai multe diferențieri decît este nevoie.
