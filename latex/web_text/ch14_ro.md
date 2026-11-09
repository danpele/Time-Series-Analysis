<!--
Textul de curs al versiunii web, capitolul 14 (RO). Inserat în pagină de latex/build_web_lecture.py.
Format: # titlul secțiunii (introducere), ## titlul frame-ului {h=hash} (text), @after (text după blocul live), ## @end (sinteză).
Ecuații numerotate: $$ ... $$ {#eq:label}, referință @eq:label. Citări: @Key -> Autor (an), [@Key] -> (Autor, an).
Hash-uri curente: python3 build_web_lecture.py 14 --hashes
-->

## @intro

@text
[Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf) a modelat volatilitatea cîte unei singure serii de randamente cu modele de heteroscedasticitate condiționată autoregresivă generalizată (*generalised autoregressive conditional heteroskedasticity*, GARCH). Cele mai multe întrebări ale managementului riscului privesc însă mai multe active deodată. Riscul unui portofoliu depinde de felul în care componentele lui se mișcă împreună, o acoperire funcționează doar atît de bine cît de strînsă este legătura dintre activ și instrumentul de acoperire, iar propagarea unei crize de la o piață la alta este o afirmație despre mișcarea comună. Toate trei întrebările au ca răspuns covarianțe și corelații, iar acestea se schimbă în timp.

Capitolul extinde GARCH la $N$ serii. Obiectul de studiu este matricea de covarianță condiționată $\mathbf{H}_t$, care conține pe diagonală varianțele tuturor activelor și în afara diagonalei covarianțele lor. Capitolul stabilește mai întîi condițiile pe care orice model pentru $\mathbf{H}_t$ trebuie să le respecte, apoi prezintă modelele generale VEC și BEKK și motivul pentru care ele devin inutilizabile dincolo de cîteva active. Modelele cu corelație condiționată constantă (*constant conditional correlation*, CCC) și cu corelație condiționată dinamică (*dynamic conditional correlation*, DCC) rezolvă problema separînd volatilitățile de corelații și estimîndu-le în doi pași.

Aplicațiile folosesc date zilnice pentru S&P 500, indicele german DAX, indicele românesc BET, cursul EUR/RON și Bitcoin. Corelația dintre S&P 500 și DAX nu este constantă: testul raportului de verosimilitate față de corelația constantă dă 131,8, mult peste valoarea critică de 5,99 la pragul de 5%. Corelațiile cresc în crize, cel mai mult pentru activele slab legate de piețele mari în perioadele calme. Aceleași covarianțe condiționate dau o valoare la risc (*Value at Risk*, VaR) de 1% a portofoliului care se adaptează stării pieței și rapoarte dinamice de acoperire care elimină în afara eșantionului mai mult risc decît un raport static.

@chart tsa_ch14_dcc_sp_dax
Corelația DCC prognozată cu un pas înainte pentru randamentele zilnice ale S&P 500 și DAX, corelația pe ferestre mobile de 250 de zile și valoarea constantă CCC, 4 ianuarie 2000 – 18 septembrie 2026.

@goals
- **Explicați** de ce riscul unui portofoliu, acoperirea riscului și contagiunea cer un model pentru covarianțe variabile în timp.
- **Scrieți** modelele VEC, BEKK, CCC și DCC și numărați-le parametrii.
- **Calculați** de mînă un pas al recurenței DCC și interpretați $a$, $b$ și $a+b$.
- **Estimați** un model DCC în doi pași și testați-l față de corelațiile constante.
- **Calculați** VaR 1% al unui portofoliu pe baza lui $\mathbf{H}_t$ și verificați-l prin backtesting cu testele Kupiec și Christoffersen.
- **Calculați** rapoarte dinamice de acoperire cu varianță minimă și eficiența acoperirii.

@plan
- Rolul volatilității multivariate | Varianța portofoliului, acoperirea riscului și contagiunea; corelațiile pe ferestre mobile și tranzacționarea asincronă.
- Cadrul GARCH multivariat | Matricea de covarianță condiționată, simetria, caracterul pozitiv definit și verosimilitatea.
- Modelele VEC și BEKK | Modelele generale, transmiterea volatilității și numărul de parametri.
- Corelații condiționate constante și dinamice | Descompunerea în volatilități și corelații, recurența DCC și estimarea în doi pași.
- DCC pentru S&P 500 și DAX | Două modele GARCH, modelul DCC estimat și corelațiile în crizele din 2008 și 2020.
- Cinci piețe: S&P 500, DAX, BET, EUR/RON, Bitcoin | Un DCC cu cinci active și corelațiile în perioada calmă și în crahul din 2020.
- VaR 1% al unui portofoliu cu DCC | De la $\mathbf{H}_t$ la VaR-ul unui portofoliu și backtesting-ul lui în afara eșantionului.
- Rapoarte dinamice de acoperire | Raportul de acoperire cu varianță minimă și acoperirea BET cu DAX.
- Legături cu alte capitole și limite | GARCH, VAR și cointegrare; limitele modelului DCC și extensiile lui.
- Contribuția posibilă a AI | Ce poate scrie un asistent și ce trebuie verificat.
- Rezumat | Ideile principale, formulele și întrebările de autoevaluare.

## Motivația capitolului {h=56897175}

Întrebarea capitolului este cum se mișcă împreună riscurile mai multor active. În [Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf) varianța condiționată a unei singure serii de randamente era suficientă pentru a-i descrie riscul. Pentru un portofoliu cu două sau mai multe active acest lucru nu mai ajunge: două portofolii cu aceleași volatilități individuale pot avea riscuri foarte diferite, după cum componentele lor tind să scadă în aceleași zile sau în zile diferite. Mărimea care surprinde acest lucru este covarianța sau versiunea ei standardizată, corelația, iar capitolul le tratează pe amîndouă ca variabile care se schimbă de la o zi la alta.

Traseul merge de la general la practic. Pornește de la matricea de covarianță condiționată $\mathbf{H}_t$ și de la cele două condiții pe care trebuie să le îndeplinească, continuă cu modelele VEC și BEKK, care descriu toate elementele lui $\mathbf{H}_t$ deodată, și ajunge la modelele CCC și DCC, care estimează separat volatilitățile și corelațiile. Ultimele secțiuni folosesc aceste modele în trei aplicații: corelațiile în crize, VaR-ul unui portofoliu și rapoartele de acoperire.

Capitolul se sprijină direct pe trei capitole anterioare. Modelul GARCH univariat din [Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf) este piesa de bază a oricărui model multivariat; modelul vector autoregresiv (*vector autoregression*, VAR) din [Capitolul 6](../RO/Cursuri/capitol6_modele_var_cauzalitate_granger.pdf) aduce notația multivariată și ideea de transmitere între serii; cointegrarea, din [Capitolul 7](../RO/Cursuri/capitol7_cointegrare_vecm.pdf), este perechea pe termen lung a legăturilor pe termen scurt studiate aici.

## Rezultatele învățării {h=491042b4}

Primele rezultate sînt conceptuale. Cititorul trebuie să poată explica de ce varianța unui portofoliu, un raport de acoperire și întrebarea contagiunii depind toate de covarianțe care se schimbă în timp și să poată scrie principalele modele GARCH multivariate în formă matriceală. Numărarea parametrilor fiecărui model face parte din înțelegerea lui: numărul arată imediat ce modele pot fi estimate pentru cinci active și ce modele nu pot.

Celelalte rezultate sînt de calcul și empirice. Un pas al recurenței DCC se poate calcula de mînă, iar parametrii $a$, $b$ și $a+b$ au o interpretare directă: reacția la știri, persistența și viteza de revenire la corelația pe termen lung. Un model DCC se estimează în doi pași și se compară cu corelațiile constante printr-un test al raportului de verosimilitate. În sfîrșit, $\mathbf{H}_t$ se transformă în două numere practice: VaR 1% al unui portofoliu, evaluat prin backtesting, și raportul de acoperire cu varianță minimă, evaluat prin eficiența acoperirii în afara eșantionului.

## Bibliografie și instrumente {h=c1e50b2e}

@HuangPetukhina2022 dau fundamentele modelelor GARCH și implementarea lor în Python. Două sinteze tratează mai pe larg modelele multivariate: @BauwensLaurentRombouts2006 le clasifică și discută estimarea și testarea, iar @SilvennoinenTerasvirta2009 dau o prezentare compactă, cu condițiile pentru caracterul pozitiv definit. Fiecare model al capitolului are și lucrarea lui originală: @BollerslevEngleWooldridge1988 pentru VEC, @EngleKroner1995 pentru BEKK, @Bollerslev1990 pentru CCC și @Engle2002 pentru DCC.

Codul capitolului folosește pachetul arch pentru modelele GARCH univariate din pasul 1. Pachetul nu are un model DCC, așa că pasul 2 este scris explicit în circa 40 de linii numpy: recurența pentru $\mathbf{Q}_t$, rescalarea la corelații și partea de corelație a verosimilității. Scrierea explicită a recurenței are un avantaj didactic, deoarece fiecare element al modelului este vizibil în cod. Notebook-ul cursului reproduce calculele și graficele capitolului.

# Rolul volatilității multivariate

Secțiunea arată de ce un model pentru o singură volatilitate nu este suficient. Riscul portofoliului, acoperirea riscului și contagiunea depind toate de corelații, iar corelațiile observate pe piețele reale se schimbă mult în timp. Secțiunea se încheie cu o capcană practică a datelor zilnice: piețele care se închid la ore diferite.

## Trei aplicații care necesită covarianțe {h=404204e5}

Prima întrebare este riscul unui portofoliu. Dacă vectorul $\mathbf{w}$ conține ponderile activelor și $\mathbf{H}$ este matricea de covarianță a randamentelor lor, varianța randamentului portofoliului este o formă pătratică, ca la @Markowitz1952:

$$ \begin{gathered} \sigma_p^2 = \mathbf{w}^\top\mathbf{H}\mathbf{w}, \\ \text{pentru două active: } \sigma_p^2 = w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2w_1w_2\rho\,\sigma_1\sigma_2. \end{gathered} $$ {#eq:pvar}

Primii doi termeni provin din volatilitățile individuale; al treilea este singurul loc în care intră comportamentul comun. Cu ponderi pozitive, o corelație $\rho$ mai mică înseamnă o varianță mai mică a portofoliului, iar acesta este întregul mecanism al diversificării.

A doua întrebare este acoperirea riscului (*hedging*): cîte unități dintr-un activ trebuie vîndute pentru a compensa riscul altuia. Secțiunea 7 arată că răspunsul este raportul dintre o covarianță și o varianță, deci un raport de acoperire se mișcă ori de cîte ori se mișcă covarianța. A treia întrebare este contagiunea (*contagion*): se mișcă piețele mai strîns împreună într-o criză? @LonginSolnik2001 au găsit că corelațiile dintre piețele internaționale de acțiuni sînt mai mari pe piețele în scădere. Dacă corelațiile cresc tocmai cînd crește volatilitatea, diversificarea slăbește exact în momentul în care este mai necesară.

Un model GARCH univariat dă volatilitățile $\sigma_{1,t}$ și $\sigma_{2,t}$ zi de zi. Formula @eq:pvar are nevoie și de corelație, iar scopul acestui capitol este să adauge o corelație $\rho_t$ care se schimbă în timp în același fel.

## Exemplu rezolvat: corelația și riscul portofoliului {h=05bf9ce2}

Considerăm două active, fiecare cu volatilitatea anuală de 20%, și un portofoliu împărțit 50/50. Cu volatilitățile în procente, $\sigma_1^2 = \sigma_2^2 = 400$ și $w_1^2 = w_2^2 = 0{,}25$, deci primii doi termeni din @eq:pvar însumează 200, oricare ar fi corelația. Cu $\rho = 0{,}2$ termenul de covarianță este $2 \cdot 0{,}25 \cdot 0{,}2 \cdot 400 = 40$, varianța este 240, iar volatilitatea $\sqrt{240} = 15{,}5\%$. Cu $\rho = 0{,}8$ termenul de covarianță este 160, varianța 360, iar volatilitatea 19,0%.

Nu s-a schimbat nimic în afară de corelație: aceleași active, aceleași ponderi, aceleași volatilități individuale. Totuși riscul portofoliului crește cu mai mult de o cincime. Consecința practică este că un model de risc care ține $\rho$ la valoarea din perioadele calme subestimează riscul atunci cînd corelațiile cresc, iar datele de mai jos arată că ele cresc în crize.

## Volatilitatea portofoliului în funcție de corelație {h=b5cb9eac}

Graficul arată volatilitatea anuală a portofoliului 50/50, în procente, pe axa verticală, în funcție de corelația $\rho$ pe axa orizontală, de la $-1$ la $1$, cu ambele active la volatilitatea de 20%. Cele două puncte marcate sînt cazurile din exemplul rezolvat: 15,5% la $\rho = 0{,}2$ și 19,0% la $\rho = 0{,}8$.

Curba este rădăcina pătrată a unei funcții liniare de $\rho$: pentru acest portofoliu $\sigma_p^2 = 200 + 200\rho$. Pornește de la zero pentru $\rho = -1$, urcă la început abrupt, apoi mai lent, și ajunge la 20% pentru $\rho = 1$. Citit de la dreapta la stînga, graficul arată cîștigul din diversificare: distanța verticală dintre curbă și nivelul de 20% la fiecare corelație.

## Interpretarea curbei de diversificare {h=7a4a44dd}

Trei puncte ale curbei merită reținute. La $\rho = 1$ cele două active se mișcă la fel, nu există diversificare, iar portofoliul are volatilitatea de 20% a fiecărui activ. La $\rho = 0$ termenul de covarianță dispare, iar volatilitatea este $20/\sqrt{2} \approx 14{,}1\%$. La $\rho = -1$ mișcările celor două active se anulează exact și riscul dispare, un caz care nu apare la indicii bursieri reali.

Deoarece curba este concavă, aceeași creștere a corelației adaugă mai multă volatilitate la corelații mici decît la corelații mari. Pentru portofoliile care combină active slab corelate, tocmai portofoliile construite pentru diversificare, o creștere a corelației este deci cea mai costisitoare. De aici vine întrebarea empirică a capitolului: rămîne $\rho$ constant pe piețele reale sau se mișcă odată cu starea pieței?

## Corelații variabile în timp {h=79126fd5}

Un prim răspuns vine din corelațiile pe ferestre mobile (*rolling windows*): corelația de selecție a randamentelor logaritmice zilnice calculată pe ultimele 250 de zile de tranzacționare, aproximativ un an, și datată la sfîrșitul ferestrei. Graficul arată această corelație pe axa verticală, în funcție de timp pe axa orizontală, pentru trei perechi: S&P 500 și DAX, din 2001; BET și DAX, din 2001; S&P 500 și Bitcoin, din 2015. Benzile colorate marchează septembrie 2008 – martie 2009 și februarie – aprilie 2020, iar o linie punctată marchează zero.

Fiecare corelație folosește doar zilele în care se tranzacționează ambele piețe, astfel încît fiecare pereche de randamente să se refere la aceeași zi calendaristică. O corelație pe ferestre mobile este un instrument descriptiv, nu un model: dă aceeași pondere fiecărei zile din fereastră și uită tot ce este mai vechi de 250 de zile. Este utilă aici pentru că arată, fără nicio ipoteză despre dinamică, dacă corelația este stabilă.

## Interpretarea corelațiilor pe ferestre mobile {h=a2a99fbf}

Niciuna dintre cele trei corelații nu este stabilă. Pentru S&P 500 și DAX ea variază între 0,21 în aprilie 2026 și 0,80 în octombrie 2012, în jurul unei valori pe tot eșantionul de 0,60. Valorile cele mai mari nu apar în 2008, ci în timpul crizei datoriilor din zona euro din 2010–2012, care a legat cele două piețe și mai strîns.

BET și DAX arată o schimbare mai lentă. Înainte de 2007 corelația era aproape de zero, cu un minimum de $-0{,}08$; după aderarea României la Uniunea Europeană (UE) a crescut în trepte, pînă la 0,70 în august 2020. Piața românească s-a integrat cu piețele europene, iar valoarea ei pentru diversificarea unui portofoliu european a scăzut în consecință. S&P 500 și Bitcoin au trecut de la $-0{,}16$ în decembrie 2019 la 0,57 în februarie 2023: după 2020 Bitcoin s-a comportat mai mult ca un activ riscant, corelat cu acțiunile.

Concluzia este că corelațiile se îndepărtează mult de orice valoare constantă, pe parcursul anilor prin integrare și în cîteva luni prin crize. Un model trebuie să le permită să se miște.

## Tranzacționarea asincronă și deplasarea corelațiilor zilnice {h=faeeb1ad}

Randamentele zilnice ale unor piețe diferite nu acoperă aceleași ore. Bursa din București se închide la 16:45, ora Europei Centrale (*Central European Time*, CET), adică 17:45 ora locală, cea din Frankfurt la 17:30 CET, iar cea din New York la 22:00 CET, în timp ce Bitcoin se tranzacționează non-stop. O știre care apare după închiderea piețelor europene intră în prețurile europene abia în ziua următoare de tranzacționare. Acesta este fenomenul de tranzacționare asincronă (*non-synchronous trading*), care împarte reacția la un singur eveniment între două randamente zilnice.

Anunțul din 9 aprilie 2025 arată efectul. Suspendarea tarifelor vamale ale Statelor Unite ale Americii (SUA) a fost anunțată după-amiaza la New York, după închiderea piețelor europene: pe 9 aprilie S&P 500 a crescut cu 9,1%, iar DAX a scăzut cu 3,1%; pe 10 aprilie S&P 500 a scăzut cu 3,5%, iar DAX a crescut cu 4,4%. Aceeași știre a produs în aceeași zi calendaristică randamente de semne opuse, iar acest lucru trage corelația zilnică spre zero.

Randamentele săptămînale sînt mult mai puțin afectate, deoarece o întîrziere de o zi contează puțin pe cinci zile. Din 2015, corelația randamentelor zilnice față de cea a randamentelor săptămînale este 0,32 față de 0,46 pentru S&P 500 și BET, 0,56 față de 0,68 pentru S&P 500 și DAX și 0,42 față de 0,47 pentru BET și DAX. Diferența este mică pentru ultima pereche, ale cărei piețe se închid aproape simultan. Remediile sînt randamentele săptămînale sau randamentele măsurate între momente sincrone; modelele zilnice ale acestui capitol țin cont de această deplasare atunci cînd citesc corelațiile transatlantice.

## Recapitulare: rolul volatilității multivariate {h=90c4cac5}

Varianța portofoliului $\mathbf{w}^\top\mathbf{H}\mathbf{w}$, raportul de acoperire și întrebarea contagiunii depind toate de covarianțe. Corelațiile pe ferestre mobile ale piețelor reale se mișcă mult, prin integrarea piețelor, prin crize și prin apariția unor clase noi de active, cum este Bitcoin. Înainte de a calcula orice corelație zilnică trebuie verificate orele de tranzacționare, deoarece închiderile asincrone deplasează corelațiile zilnice spre zero.

## @end

Secțiunea a motivat un model în care corelațiile se schimbă în timp; ferestrele mobile arată schimbarea, dar nu o explică și depind de o lungime arbitrară a ferestrei. Secțiunea următoare construiește cadrul general, o matrice de covarianță condiționată actualizată zilnic pe baza șocurilor trecute, și condițiile pe care trebuie să le îndeplinească.

# Cadrul GARCH multivariat

Secțiunea definește modelul GARCH multivariat în formă generală: vectorul randamentelor, matricea de covarianță condiționată $\mathbf{H}_t$, condițiile de simetrie și de caracter pozitiv definit și estimarea prin verosimilitate maximă.

## Modelul pentru N serii de randamente {h=76c0fef1}

Modelul pentru $N$ randamente are aceeași structură ca modelul GARCH univariat, scrisă cu vectori și matrice:

$$ \begin{gathered} \mathbf{r}_t = \boldsymbol{\mu} + \boldsymbol{\varepsilon}_t, \qquad \boldsymbol{\varepsilon}_t = \mathbf{H}_t^{1/2}\boldsymbol{\eta}_t, \\ \mathbf{H}_t = \Var(\mathbf{r}_t \mid \mathcal{F}_{t-1}). \end{gathered} $$ {#eq:mgarch}

Aici $\mathbf{r}_t$ este vectorul celor $N$ randamente din ziua $t$, $\boldsymbol{\mu}$ vectorul mediilor (sau prognoza unui model VAR, ca în [Capitolul 6](../RO/Cursuri/capitol6_modele_var_cauzalitate_granger.pdf)), iar $\boldsymbol{\eta}_t$ un vector de inovații independente și identic distribuite (i.i.d.), cu media zero și matricea de covarianță $\mathbf{I}_N$. Matricea $\mathbf{H}_t^{1/2}$ este o rădăcină pătrată a lui $\mathbf{H}_t$, de exemplu factorul Cholesky $\mathbf{L}_t$, cu $\mathbf{L}_t\mathbf{L}_t^\top = \mathbf{H}_t$; orice astfel de rădăcină dă șocurilor $\boldsymbol{\varepsilon}_t$ matricea de covarianță condiționată $\mathbf{H}_t$.

Matricea de covarianță condiționată (*conditional covariance matrix*) $\mathbf{H}_t$ se calculează din informația $\mathcal{F}_{t-1}$ disponibilă pînă în ziua $t-1$. Pe diagonala ei se află varianțele condiționate $h_{ii,t} = \sigma_{i,t}^2$, exact mărimile din [Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf). În afara diagonalei se află covarianțele condiționate $h_{ij,t}$, din care rezultă corelația condiționată:

$$ \rho_{ij,t} = \frac{h_{ij,t}}{\sqrt{h_{ii,t}\,h_{jj,t}}}. $$ {#eq:rho}

Un model GARCH multivariat (*multivariate GARCH*, MGARCH) este o regulă care calculează $\mathbf{H}_t$ din șocurile trecute $\boldsymbol{\varepsilon}_{t-1}$ și din $\mathbf{H}_{t-1}$. Modelele din secțiunile următoare diferă doar prin această regulă.

## Două condiții pentru Hₜ {h=297f608d}

Prima condiție este simetria: covarianța activului $i$ cu activul $j$ este covarianța lui $j$ cu $i$, deci $h_{ij,t} = h_{ji,t}$ și doar cele $N(N+1)/2$ elemente de pe diagonală și de sub ea sînt distincte. Operatorul $\mathrm{vech}$ așază aceste elemente, coloană cu coloană, într-un vector; pentru două active, $\mathrm{vech}(\mathbf{H}) = (h_{11}, h_{12}, h_{22})^\top$. Numărul de elemente distincte crește repede: 3 pentru $N = 2$, 55 pentru $N = 10$ și 1 275 pentru $N = 50$.

A doua condiție este caracterul pozitiv definit (*positive definiteness*): $\mathbf{w}^\top\mathbf{H}_t\mathbf{w} > 0$ pentru orice vector nenul $\mathbf{w}$. Conform @eq:pvar, membrul stîng este varianța portofoliului cu ponderile $\mathbf{w}$, deci condiția spune că orice portofoliu are o varianță pozitivă. Pentru două active ea se reduce la $h_{11,t} > 0$ și $h_{11,t}h_{22,t} - h_{12,t}^2 > 0$, ceea ce este echivalent cu $|\rho_{12,t}| < 1$.

Exemplul de pe slide arată ce se întîmplă fără această condiție. Cu $h_{11} = 4$, $h_{22} = 1$ și $h_{12} = 2{,}5$ determinantul este $4 - 6{,}25 < 0$, iar @eq:rho dă o „corelație” de 1,25, pe care nicio pereche de variabile aleatoare nu o poate avea. Un model care calculează separat fiecare element al lui $\mathbf{H}_t$ poate produce astfel de matrice. Un model MGARCH bun garantează ambele condiții pentru orice $t$, oricare ar fi șocurile, și face acest lucru cu puțini parametri.

## Estimarea prin verosimilitate maximă {h=bfddae03}

Dacă inovațiile $\boldsymbol{\eta}_t$ urmează distribuția Normală multivariată, șocurile $\boldsymbol{\varepsilon}_t$ sînt condiționat normale, cu matricea de covarianță $\mathbf{H}_t$, iar logaritmul verosimilității eșantionului este

$$ \ell(\theta) = -\frac12\sum_{t=1}^{T}\left(N\log 2\pi + \log|\mathbf{H}_t| + \boldsymbol{\varepsilon}_t^\top\mathbf{H}_t^{-1}\boldsymbol{\varepsilon}_t\right), $$ {#eq:loglik}

unde $|\mathbf{H}_t|$ este determinantul, iar $\theta$ reunește toți parametrii. Pentru $N = 1$ determinantul este $\sigma_t^2$, forma pătratică este $\varepsilon_t^2/\sigma_t^2$, iar @eq:loglik devine verosimilitatea GARCH din [Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf). Cei doi termeni au aceleași roluri ca acolo: determinantul penalizează varianțele mari, iar forma pătratică penalizează șocurile mari în raport cu covarianța prognozată.

Randamentele zilnice au cozi groase, așa că ipoteza distribuției Normale nu este literal adevărată. Verosimilitatea Normală este atunci folosită ca cvasi-verosimilitate maximă (*quasi-maximum likelihood*, QML): estimatorii parametrilor lui $\mathbf{H}_t$ rămîn consistenți atîta timp cît $\mathbf{H}_t$ este corect specificată, dar erorile standard obișnuite sînt greșite și au nevoie de o corecție robustă (*sandwich*).

Dificultatea practică este de calcul. Fiecare evaluare a lui $\ell(\theta)$ cere inversa și determinantul lui $\mathbf{H}_t$ pentru fiecare zi a eșantionului, iar un algoritm de optimizare numerică evaluează $\ell(\theta)$ de foarte multe ori. Costul crește repede cu $N$ și, mai ales, cu numărul de parametri, subiectul secțiunii următoare.

## Recapitulare: cadrul GARCH multivariat {h=9a9b0e4f}

Matricea de covarianță condiționată $\mathbf{H}_t$ are varianțele condiționate pe diagonală și covarianțele în afara ei, iar corelația condiționată rezultă din @eq:rho. Un model valid trebuie să păstreze $\mathbf{H}_t$ simetrică, cu $N(N+1)/2$ elemente distincte, și pozitiv definită în fiecare zi. Estimarea folosește verosimilitatea Normală multivariată @eq:loglik, ca cvasi-verosimilitate atunci cînd randamentele au cozi groase.

## @end

Cadrul lasă deschisă regula care actualizează $\mathbf{H}_t$. Secțiunea următoare prezintă cele două răspunsuri clasice, VEC și BEKK, care modelează toate elementele matricei deodată, și arată prețul pe care îl plătesc în parametri.

# Modelele VEC și BEKK

Secțiunea prezintă cel mai general model MGARCH liniar, VEC, și modelul BEKK, care garantează prin construcție caracterul pozitiv definit. Numărarea parametrilor lor arată de ce niciunul nu poate fi folosit pentru mai mult de cîteva active.

## Modelul VEC {h=29949df3}

@BollerslevEngleWooldridge1988 au scris cea mai generală extensie liniară a modelului GARCH(1,1): fiecare element distinct al lui $\mathbf{H}_t$ depinde de toate pătratele și produsele încrucișate ale șocurilor trecute și de toate elementele lui $\mathbf{H}_{t-1}$. Cu operatorul $\mathrm{vech}$, modelul VEC(1,1) este

$$ \begin{aligned} \mathrm{vech}(\mathbf{H}_t) = \mathbf{c} &+ \mathbf{A}\,\mathrm{vech}(\boldsymbol{\varepsilon}_{t-1}\boldsymbol{\varepsilon}_{t-1}^\top) \\ &+ \mathbf{B}\,\mathrm{vech}(\mathbf{H}_{t-1}). \end{aligned} $$ {#eq:vec}

Matricea $\boldsymbol{\varepsilon}_{t-1}\boldsymbol{\varepsilon}_{t-1}^\top$ conține pe diagonală pătratele șocurilor de ieri și în afara diagonalei produsele lor încrucișate. Vectorul $\mathbf{c}$ conține constante, $\mathbf{A}$ măsoară reacția la șocuri, iar $\mathbf{B}$ persistența, așa cum fac $\omega$, $\alpha$ și $\beta$ într-un GARCH univariat.

Generalitatea costă. Cu $k = N(N+1)/2$ elemente distincte, $\mathbf{c}$ are $k$ componente, iar $\mathbf{A}$ și $\mathbf{B}$ sînt matrice $k \times k$, deci modelul are $k + 2k^2$ parametri: 21 de parametri pentru două active și 6 105 pentru zece. Mai mult, nimic din @eq:vec nu păstrează $\mathbf{H}_t$ pozitiv definită; restricțiile care ar face acest lucru sînt complicate și greu de impus în timpul estimării.

Modelul VEC diagonal păstrează doar diagonala matricelor $\mathbf{A}$ și $\mathbf{B}$, astfel încît fiecare element urmează propriul GARCH(1,1):

$$ h_{ij,t} = c_{ij} + a_{ij}\varepsilon_{i,t-1}\varepsilon_{j,t-1} + b_{ij}h_{ij,t-1}. $$

Rămîn $3k$ parametri, 9 pentru $N = 2$, dar caracterul pozitiv definit tot nu este garantat, deoarece fiecare element se actualizează fără legătură cu celelalte.

## Modelul BEKK {h=9e4b2aa5}

@EngleKroner1995 au propus o formă numită după Baba, Engle, Kraft și Kroner (BEKK), care este pozitiv definită prin construcție:

$$ \begin{aligned} \mathbf{H}_t = \mathbf{C}\mathbf{C}^\top &+ \mathbf{A}^\top\boldsymbol{\varepsilon}_{t-1}\boldsymbol{\varepsilon}_{t-1}^\top\mathbf{A} \\ &+ \mathbf{B}^\top\mathbf{H}_{t-1}\mathbf{B}, \end{aligned} $$ {#eq:bekk}

cu $\mathbf{C}$ inferior triunghiulară și $\mathbf{A}$, $\mathbf{B}$ matrice complete de dimensiune $N \times N$. $\mathbf{A}$ descrie reacția la șocurile de ieri, iar $\mathbf{B}$ persistența lui $\mathbf{H}_{t-1}$; elementele matricei constante $\mathbf{C}\mathbf{C}^\top$ se notează $c_{ij}^*$.

Garanția vine din forma fiecărui termen, un „pătrat” $\mathbf{X}^\top\mathbf{M}\mathbf{X}$ al unei matrice $\mathbf{M}$ pozitiv semidefinite: matricea unitate în primul termen (cu $\mathbf{X} = \mathbf{C}^\top$), produsul exterior $\boldsymbol{\varepsilon}_{t-1}\boldsymbol{\varepsilon}_{t-1}^\top$ în al doilea și $\mathbf{H}_{t-1}$ în al treilea. Pentru orice vector $\mathbf{w}$,

$$ \mathbf{w}^\top\mathbf{X}^\top\mathbf{M}\mathbf{X}\mathbf{w} = (\mathbf{X}\mathbf{w})^\top\mathbf{M}(\mathbf{X}\mathbf{w}) \ge 0, $$

iar un termen constant $\mathbf{C}\mathbf{C}^\top$ de rang maxim face suma strict pozitivă. Numărul de parametri este $N(N+1)/2 + 2N^2$: 11 pentru două active, 255 pentru zece.

Elementele din afara diagonalei ale unei matrice $\mathbf{A}$ complete au un sens economic: ele măsoară transmiterea volatilității (*volatility spillovers*), efectul unui șoc al activului 1 asupra varianței activului 2. Restricția ca $\mathbf{A}$ și $\mathbf{B}$ să fie diagonale dă BEKK-ul diagonal, în care covarianța urmează

$$ \begin{aligned} h_{12,t} = c_{12}^* &+ a_{11}a_{22}\,\varepsilon_{1,t-1}\varepsilon_{2,t-1} \\ &+ b_{11}b_{22}\,h_{12,t-1}. \end{aligned} $$

BEKK-ul scalar merge mai departe, cu $\mathbf{A} = a\mathbf{I}$ și $\mathbf{B} = b\mathbf{I}$, astfel încît toate elementele au aceeași dinamică. Ambele restricții economisesc parametri și elimină transmiterea volatilității.

## Exemplu rezolvat: un pas al unui BEKK diagonal {h=180c0cea}

Ecuația covarianței unui BEKK diagonal se poate calcula de mînă. Valorile date sînt $c_{12}^* = 0{,}02$, $a_{11} = 0{,}30$, $a_{22} = 0{,}25$, $b_{11} = 0{,}94$, $b_{22} = 0{,}95$, covarianța de ieri $h_{12,t-1} = 0{,}5$ și șocurile de ieri $\varepsilon_{1,t-1} = -2$ și $\varepsilon_{2,t-1} = -1{,}5$. Coeficienții ecuației covarianței sînt produse: $a_{11}a_{22} = 0{,}075$ pentru reacția la produsul șocurilor și $b_{11}b_{22} = 0{,}893$ pentru persistență.

Produsul șocurilor este $(-2)(-1{,}5) = 3{,}000$, pozitiv pentru că ambele active au scăzut. Covarianța de azi este

$$ \begin{aligned} h_{12,t} &= 0{,}020 + 0{,}075 \cdot 3{,}000 + 0{,}893 \cdot 0{,}5 \\ &= 0{,}020 + 0{,}225 + 0{,}4465 = 0{,}6915. \end{aligned} $$

O scădere comună crește covarianța de la 0,5 la 0,6915; un șoc cu semne opuse ar fi dat un produs negativ și ar fi scăzut-o.

Exemplul arată și ce nu poate face modelul diagonal. Deoarece $a_{12} = a_{21} = 0$, un șoc al activului 1 afectează varianța activului 2 numai prin covarianță, niciodată direct: BEKK-ul diagonal nu are transmitere a volatilității.

## Problema dimensionalității {h=34b111c5}

Tabelul numără parametrii ecuației de varianță pentru fiecare model, pentru 2, 5, 10 și 50 de active; mediile $\boldsymbol{\mu}$ nu sînt numărate. Pentru CCC și DCC, introduse în secțiunea următoare, numărul este de 3 parametri GARCH pe activ plus cele $N(N-1)/2$ corelații, iar DCC mai adaugă 2.

Rîndurile diferă prin ordine de mărime. VEC complet are nevoie de 21 de parametri pentru două active, 465 de parametri pentru cinci, 6 105 pentru zece și 3 252 525 pentru cincizeci: pentru cincizeci de active are mai mulți parametri decît numărul de observații al oricărui eșantion de date zilnice. BEKK complet este mult mai economic, dar ajunge totuși la 6 275 de parametri pentru cincizeci de active. BEKK-ul diagonal, CCC și DCC rămîn apropiate, cu 1 375 pînă la 1 377 de parametri pentru cincizeci de active, iar BEKK-ul scalar este cel mai ieftin, cu 1 277. Această creștere a numărului de parametri odată cu dimensiunea se numește blestemul dimensionalității (*curse of dimensionality*).

## Numărul de parametri în funcție de numărul de active {h=d848e0c0}

Graficul arată numărul de parametri ai ecuației de varianță, pe o axă verticală logaritmică, în funcție de numărul de active $N$, de la 2 la 50, pentru șase modele: VEC(1,1), BEKK(1,1), VEC diagonal, BEKK diagonal, DCC și CCC. BEKK-ul scalar din tabel nu este desenat.

Pe o scară logaritmică o distanță verticală constantă înseamnă un raport constant. Curba VEC se află mult deasupra tuturor celorlalte și se îndepărtează continuu, trecînd de $10^6$ la $N = 50$. BEKK complet și VEC diagonal se află între ele, iar ultimele trei curbe (BEKK diagonal, DCC și CCC) aproape coincid, deoarece numerele lor diferă cu cel mult doi parametri.

## Interpretarea numărului de parametri {h=7f71242a}

Termenii dominanți ai formulelor dau ritmurile de creștere. Pentru VEC, $2k^2$ cu $k \approx N^2/2$ crește ca $N^4/2$; pentru BEKK complet numărul crește ca $2N^2$. Ambele devin imposibil de estimat dincolo de cîteva active, deoarece un algoritm de optimizare numerică nu poate căuta în mod fiabil printre sute sau mii de parametri cu o verosimilitate ca @eq:loglik.

CCC și DCC cresc ca $N^2/2$, dar numărul ascunde o diferență importantă: corelațiile nu se obțin prin optimizare numerică. Așa cum arată secțiunea următoare, ele sînt corelații de selecție ale reziduurilor standardizate, calculate direct. Parametrii GARCH se estimează cîte o serie pe rînd, cu cîte trei parametri. Singura optimizare comună din DCC privește doi parametri, $a$ și $b$, oricare ar fi numărul de active. De aceea DCC poate fi folosit acolo unde VEC și BEKK nu pot.

## Recapitulare: VEC și BEKK {h=e72ae1ec}

VEC este cel mai general model liniar, cu $k + 2k^2$ parametri și fără garanția caracterului pozitiv definit. BEKK este pozitiv definit prin construcție, iar elementele $a_{ij}$ din afara diagonalei matricei $\mathbf{A}$ măsoară transmiterea volatilității. Versiunile diagonală și scalară reduc numărul de parametri, dar elimină transmiterea volatilității.

## @end

VEC și BEKK modelează întreaga matrice $\mathbf{H}_t$ într-o singură ecuație, iar aceasta este sursa costului lor. Secțiunea următoare urmează un alt drum: descompune $\mathbf{H}_t$ în volatilități, pe care modelele GARCH univariate le tratează deja, și o matrice de corelații cu foarte puțini parametri de dinamică.

# Corelații condiționate constante și dinamice

Secțiunea descompune matricea de covarianță în volatilități și corelații, prezintă modelul cu corelație condiționată constantă și modelul cu corelație condiționată dinamică al lui Engle, explică estimarea lor în doi pași și testul modelului DCC față de CCC.

## Separarea volatilităților de corelații {h=0c4c17bc}

Orice matrice de covarianță se poate scrie ca volatilități înmulțite cu corelații:

$$ \begin{gathered} \mathbf{H}_t = \mathbf{D}_t\mathbf{R}_t\mathbf{D}_t, \\ \mathbf{D}_t = \mathrm{diag}(\sigma_{1,t}, \dots, \sigma_{N,t}), \end{gathered} $$ {#eq:drd}

unde $\mathbf{R}_t$ este matricea corelațiilor condiționate, cu 1 pe diagonală. Element cu element, $h_{ij,t} = \rho_{ij,t}\,\sigma_{i,t}\sigma_{j,t}$, adică @eq:rho citită invers. Fiecare volatilitate urmează un model GARCH(1,1) univariat [@Bollerslev1986]:

$$ \sigma_{i,t}^2 = \omega_i + \alpha_i\varepsilon_{i,t-1}^2 + \beta_i\sigma_{i,t-1}^2. $$

Împărțind fiecare șoc la propria volatilitate obținem reziduurile standardizate (*standardised residuals*) $z_{i,t} = \varepsilon_{i,t}/\sigma_{i,t}$. Fiecare are varianța condiționată 1, iar covarianța condiționată dintre $z_{i,t}$ și $z_{j,t}$ este $h_{ij,t}/(\sigma_{i,t}\sigma_{j,t}) = \rho_{ij,t}$. Matricea de covarianță condiționată a vectorului $\mathbf{z}_t$ este deci chiar $\mathbf{R}_t$: după eliminarea volatilităților, rămîn de modelat doar corelațiile.

Cea mai simplă alegere este modelul cu corelație condiționată constantă (CCC) al lui @Bollerslev1990, cu $\mathbf{R}_t = \mathbf{R}$ pentru orice $t$. $\mathbf{H}_t$ este atunci pozitiv definită ori de cîte ori $\mathbf{R}$ este și varianțele GARCH sînt pozitive, iar $\mathbf{R}$ se estimează prin matricea corelațiilor de selecție ale lui $z_{i,t}$. În CCC covarianțele se mișcă totuși, dar numai prin volatilități. Corelațiile pe ferestre mobile din prima secțiune contrazic această ipoteză.

## Robert Engle și modelul DCC {h=9747ac92}

Robert F. Engle, născut în 1942, a primit în 2003 Premiul Nobel pentru economie pentru metodele de analiză a seriilor de timp economice cu volatilitate variabilă în timp. @Engle2002 a propus ca și corelațiile să urmeze o recurență de aceeași formă ca GARCH. O matrice auxiliară $\mathbf{Q}_t$, cu elementele $q_{ij,t}$, se actualizează zilnic pe baza vectorului reziduurilor standardizate de ieri:

$$ \begin{aligned} \mathbf{Q}_t &= (1 - a - b)\,\bar{\mathbf{Q}} + a\,\mathbf{z}_{t-1}\mathbf{z}_{t-1}^\top \\ &\quad + b\,\mathbf{Q}_{t-1}, \\ \rho_{ij,t} &= \frac{q_{ij,t}}{\sqrt{q_{ii,t}\,q_{jj,t}}}. \end{aligned} $$ {#eq:dcc}

Rescalarea din a doua parte este necesară deoarece $\mathbf{Q}_t$ este o medie ponderată a unor matrice ale căror diagonale nu sînt exact 1; împărțirea la rădăcinile pătrate ale diagonalei o transformă într-o matrice de corelații propriu-zisă, cu 1 pe diagonală.

$\bar{\mathbf{Q}}$ este matricea corelațiilor necondiționate ale lui $\mathbf{z}_t$ și se fixează la valoarea ei de selecție înainte de optimizare. Aceasta este țintirea corelației (*correlation targeting*): recurența este ancorată la corelația observată pe termen lung, iar de estimat rămîn doar $a$ și $b$. Parametrul $a \ge 0$ este reacția la șocul comun de ieri, $b \ge 0$ persistența, iar $a + b < 1$ asigură revenirea lui $\mathbf{Q}_t$ la $\bar{\mathbf{Q}}$.

Cu $a = b = 0$ recurența dă $\mathbf{Q}_t = \bar{\mathbf{Q}}$ în fiecare zi, adică modelul CCC. DCC conține deci CCC ca un caz particular, iar această incluziune face posibil un test al raportului de verosimilitate între cele două.

## Estimarea în doi pași {h=2f796dd1}

Înlocuind @eq:drd în logaritmul verosimilității @eq:loglik, acesta se descompune în două părți, $\ell = \ell_V(\theta_1) + \ell_C(\theta_1, a, b)$. Partea de volatilitate $\ell_V$ depinde doar de $\theta_1$, parametrii GARCH ai tuturor seriilor, și este suma verosimilităților GARCH univariate. Partea de corelație $\ell_C$ depinde de $\theta_1$ prin reziduurile standardizate și de $a$ și $b$.

Descompunerea sugerează o estimare în doi pași. În pasul 1 se estimează separat cîte un GARCH(1,1) pentru fiecare serie, ca în [Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf), și se păstrează volatilitățile estimate $\hat\sigma_{i,t}$ și reziduurile standardizate $\hat z_{i,t} = \hat\varepsilon_{i,t}/\hat\sigma_{i,t}$. În pasul 2, $\bar{\mathbf{Q}}$ se fixează la corelația de selecție a lui $\hat{\mathbf{z}}_t$, iar logaritmul verosimilității de corelație se maximizează doar după $(a, b)$:

$$ \ell_C(a, b) = -\frac12\sum_{t=1}^{T}\left(\log|\mathbf{R}_t| + \hat{\mathbf{z}}_t^\top\mathbf{R}_t^{-1}\hat{\mathbf{z}}_t - \hat{\mathbf{z}}_t^\top\hat{\mathbf{z}}_t\right). $$ {#eq:lc}

Estimatorul este consistent, rapid și funcționează pentru $N$ mare. Punctul lui slab este inferența: erorile standard din pasul 2 tratează estimațiile din pasul 1 ca fiind cunoscute și ignoră eroarea lor de estimare, deci sînt prea mici.

Testul față de CCC compară logaritmii maximizați ai verosimilității de corelație: $LR = 2(\ell_C^{DCC} - \ell_C^{CCC})$, comparat cu distribuția $\chi^2(2)$, a cărei valoare critică la 5% este 5,99. O valoare mare respinge corelația constantă. Deoarece ipoteza nulă $a = b = 0$ se află pe frontiera spațiului parametrilor, valoarea critică $\chi^2(2)$ este conservatoare: o respingere obținută cu ea este o respingere sigură. @Tse2000 a propus un alt test al corelațiilor constante, bazat pe principiul multiplicatorilor Lagrange.

## Exemplu rezolvat: un pas al recurenței DCC {h=b38f2b22}

Recurența @eq:dcc se poate urmări de mînă pentru două active. Valorile date sînt $a = 0{,}05$, $b = 0{,}93$, o corelație pe termen lung $\bar q_{12} = 0{,}5$ (cu $\bar q_{11} = \bar q_{22} = 1$), matricea de ieri $\mathbf{Q}_{t-1}$ cu 1 pe diagonală și 0,5 în afara ei și reziduurile standardizate de ieri $z_{1,t-1} = -2$ și $z_{2,t-1} = -2{,}5$, o scădere comună mare. Ponderea matricei pe termen lung este $1 - a - b = 0{,}020$.

Elementul din afara diagonalei și, deoarece în ele intră pătratele reziduurilor, elementele de pe diagonală sînt

$$ \begin{aligned} q_{12,t} &= 0{,}020 \cdot 0{,}5 + 0{,}05 \cdot (-2)(-2{,}5) \\ &\quad + 0{,}93 \cdot 0{,}5 \\ &= 0{,}010 + 0{,}250 + 0{,}465 = 0{,}725, \\ q_{11,t} &= 0{,}020 + 0{,}05 \cdot 4 + 0{,}93 = 1{,}150, \\ q_{22,t} &= 0{,}020 + 0{,}05 \cdot 6{,}25 + 0{,}93 = 1{,}2625. \end{aligned} $$

Corelația este $\rho_{12,t} = 0{,}725/\sqrt{1{,}150 \cdot 1{,}2625} = 0{,}602$.

Scăderea comună mare crește corelația de mîine de la 0,50 la 0,602. Rescalarea contează: fără ea, valoarea 0,725 a lui $q_{12,t}$ ar fi citită drept corelație și ar exagera reacția. Șocurile mari cresc și diagonala lui $\mathbf{Q}_t$, iar împărțirea la $\sqrt{q_{11,t}q_{22,t}}$ ține cont de acest lucru.

## Interpretarea parametrilor a și b {h=38e20476}

Parametrul $a$ este coeficientul de știri: măsoară cît mișcă un șoc comun de o zi corelația. Estimațiile zilnice tipice se află între 0,01 și 0,05, deci o singură zi mișcă puțin corelația, iar o schimbare mare cere o succesiune de șocuri comune.

Suma $a + b$ este persistența. Scăzînd valoarea pe termen lung din ambii membri ai lui @eq:dcc și înlocuind $\mathbf{z}_{t-1}\mathbf{z}_{t-1}^\top$ cu media ei condiționată, aproximativ $\mathbf{Q}_{t-1}$, se vede că o abatere a lui $q_{ij,t}$ de la $\bar q_{ij}$ se reduce, în medie, cu factorul $a + b$ în fiecare zi. După $n$ zile ea este înmulțită cu $(a+b)^n$ și se înjumătățește cînd $(a+b)^n = 0{,}5$, de unde timpul de înjumătățire (*half-life*) $\ln 0{,}5/\ln(a + b)$ zile. Cu $a + b = 0{,}98$ timpul de înjumătățire este de 34 de zile; cu $0{,}99$ este de 69 de zile. Aproape de 1, schimbări mici ale lui $a + b$ schimbă mult timpul de înjumătățire.

Interpretarea este aceeași ca pentru $\alpha$ și $\alpha + \beta$ într-un model GARCH(1,1) din [Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf). Două variante tratează limite ale modelului de bază: DCC-ul asimetric al lui @CappielloEngleSheppard2006 permite ca scăderile comune să miște corelațiile mai mult decît creșterile comune, iar DCC-ul corectat al lui @Aielli2013 modifică recurența astfel încît estimatorul în doi pași să rămînă consistent.

## Recapitulare: CCC și DCC {h=3a9a6cd0}

Descompunerea @eq:drd scrie $\mathbf{H}_t$ ca volatilități GARCH combinate cu o matrice de corelații. În CCC matricea de corelații este constantă; în DCC matricea auxiliară $\mathbf{Q}_t$ urmează o recurență de tip GARCH cu doi parametri, $a$ și $b$. Estimarea se face în doi pași, întîi modelele GARCH univariate, apoi $(a, b)$ cu țintirea corelației, iar un test al raportului de verosimilitate compară DCC cu CCC.

## @end

Secțiunea a construit modelul; următoarea îl aplică perechii S&P 500 și DAX, urmărind ambii pași ai estimării și întrebînd dacă corelația este constantă și cum se comportă în crizele din 2008 și 2020.

# DCC pentru S&P 500 și DAX

Secțiunea estimează pas cu pas un model DCC pentru randamentele zilnice ale S&P 500 și DAX din ianuarie 2000 pînă în septembrie 2026 și îl folosește pentru a studia corelațiile în crizele din 2008 și 2020.

## Pasul 1: două modele GARCH univariate {h=d6677b38}

Datele sînt randamente logaritmice zilnice, în procente, în cele 6 605 zile în care s-a tranzacționat pe ambele piețe, de la 4 ianuarie 2000 pînă la 18 septembrie 2026, de la furnizorul de date EOD Historical Data (EODHD). Fiecare serie primește un GARCH(1,1) cu medie constantă, estimat prin QML gaussian. Tabelul dă media $\mu$, cei trei parametri GARCH și persistența $\alpha + \beta$, împreună cu excesul de kurtosis al randamentelor brute și al reziduurilor standardizate.

Estimațiile GARCH au forma obișnuită pentru indicii bursieri. Pentru S&P 500, $\alpha = 0{,}122$ și $\beta = 0{,}857$, cu persistența 0,980; pentru DAX, $\alpha = 0{,}098$ și $\beta = 0{,}886$, cu persistența 0,984. Reacția la știri este moderată, iar șocurile de volatilitate se sting lent.

Coloanele de kurtosis arată ce realizează pasul 1. Excesul de kurtosis al S&P 500 scade de la 10,5 pentru randamente la 1,8 pentru reziduurile standardizate, iar cel al DAX de la 6,2 la 1,4: cea mai mare parte a cozilor groase provine din volatilitatea variabilă. Ce rămîne este totuși peste zero, deci cozile lui $z_t$ sînt mai groase decît cele ale distribuției Normale. Corelația randamentelor este 0,599, iar cea a reziduurilor este $\bar q_{12} = 0{,}575$; aceasta din urmă este estimația CCC și ancora recurenței DCC.

## Pasul 1: volatilitățile condiționate {h=80beec47}

Graficul arată volatilitatea GARCH(1,1) prognozată cu un pas înainte pentru fiecare indice, anualizată ca $\hat\sigma_{i,t}\sqrt{252}$ și exprimată în procente, pe axa verticală, în funcție de timp, din 2000 pînă în 2026. Cele două linii sînt S&P 500 și DAX, iar benzile colorate marchează crizele din 2008 și 2020.

În anii calmi cele două volatilități rămîn scăzute, iar DAX se află de obicei puțin deasupra S&P 500. Cele mai mari vîrfuri apar în toamna lui 2008 și în martie 2020, iar vîrfuri mai mici apar în 2001–2003, 2011, 2022 și aprilie 2025. Cele două linii urcă și coboară împreună.

## Interpretarea volatilităților {h=5d1c7c2d}

Ambele volatilități ating maximul în aceleași săptămîni: S&P 500 ajunge la 109% pe 17 martie 2020, iar DAX la 80% pe 25 martie 2020. Volatility clustering apare simultan pe cele două piețe. Doar acest lucru face covarianța $h_{12,t} = \rho_{12,t}\sigma_{1,t}\sigma_{2,t}$ să crească puternic într-o criză, chiar dacă corelația nu s-ar schimba deloc.

Persistența de 0,980 și 0,984 înseamnă că șocurile de volatilitate se sting lent, ca în [Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf). Pasul 2 pune o altă întrebare decît pasul 1: după ce volatilitățile au fost împărțite, este constantă corelația a ceea ce rămîne? Această separare este rostul modelului DCC: ea distinge o creștere a covarianței provocată de o volatilitate mai mare de o creștere provocată de o legătură mai strînsă.

## Pasul 2: modelul DCC estimat {h=768efc7b}

Pasul 2 estimează cei doi parametri DCC pe reziduurile standardizate din pasul 1. Coeficientul de știri este $a = 0{,}0152$, cu eroarea standard (*standard error*, SE) 0,0035, iar parametrul de persistență este $b = 0{,}9766$, cu SE 0,0065. Suma $a + b = 0{,}9918$ implică un timp de înjumătățire al unui șoc de corelație de 85 de zile.

Estimația lui $a$ este mică, la capătul inferior al intervalului tipic, dar de peste patru ori mai mare decît eroarea ei standard, deci corelația reacționează la știri. Suma apropiată de 1 înseamnă că o abatere a corelației de la valoarea pe termen lung durează luni. Erorile standard provin din hessiana verosimilității din pasul 2 și ignoră eroarea de estimare din pasul 1, deci sînt prea mici; chiar ținînd cont de acest lucru, $a$ este clar separat de zero.

Statistica raportului de verosimilitate față de CCC este 131,8, mult peste valoarea critică de 5,99 la pragul de 5%. Corelația constantă este respinsă: corelația dintre S&P 500 și DAX se schimbă în timp chiar și după eliminarea schimbărilor de volatilitate.

## Corelația DCC dintre S&P 500 și DAX {h=b02e5ed3}

Graficul arată trei estimări ale corelației pe axa verticală, în funcție de timp, zilnic, de la 4 ianuarie 2000 pînă la 18 septembrie 2026: corelația DCC prognozată cu un pas înainte $\hat\rho_{12,t}$, corelația pe ferestre mobile de 250 de zile și valoarea constantă CCC $\bar q_{12} = 0{,}575$, ca linie orizontală întreruptă. Benzile colorate marchează crizele din 2008 și 2020.

Corelația pe ferestre mobile începe abia în 2001, după ce prima ei fereastră este completă, în timp ce corelația DCC este disponibilă de la începutul eșantionului. Cele două estimări dinamice urmează aceleași oscilații lungi: valori mari în 2002–2003 și 2010–2012, valori mai mici în 2005–2006 și 2018–2019.

## Interpretarea corelației DCC {h=da71fa70}

Corelația DCC variază între 0,29 în martie 2000 și 0,75 în ianuarie 2010, în jurul unei medii de 0,57. Urmează aceleași oscilații ca și corelația pe ferestre mobile, dar reacționează în cîteva zile, nu într-un an, și nu cere alegerea unei ferestre. Ponderile ei scad geometric cu vechimea observației, cu rata $b$, în loc să fie egale în fereastră și nule în afara ei.

Estimarea pe ferestre mobile este mai instabilă: abaterea ei standard pe eșantion este 0,122, față de 0,083 pentru DCC. Ea sare cînd o zi extremă intră în fereastră și sare din nou, în sens opus, exact 250 de zile mai tîrziu, cînd ziua iese din fereastră. Scăderea corelației pe ferestre mobile în 2025–2026 este un exemplu. Zilele asincrone din aprilie 2025, descrise în prima secțiune, intră în fereastră și o trag în jos pînă la aproximativ 0,21 exact 250 de zile, după care ea sare înapoi. Corelația DCC înregistrează aceleași zile, dar le uită treptat.

## Corelațiile în crize {h=a8cf94cb}

Tabelul compară trei perioade: o perioadă calmă, 2017–2019, și cele două crize, septembrie 2008 – martie 2009 (lunile de după falimentul Lehman Brothers, din fotografie) și februarie – aprilie 2020. Pentru fiecare perioadă dă corelația DCC medie dintre S&P 500 și DAX și volatilitatea GARCH anualizată medie a S&P 500. În perioada calmă corelația are media 0,56, iar volatilitatea 13%; în 2008–2009, 0,63 și 51%; în 2020, 0,70 și 58%.

Corelațiile sînt mai mari în crize, dar schimbarea este moderată în comparație cu schimbarea volatilității, care aproximativ se împătrește. Cea mai mare parte a creșterii covarianței într-o criză vine deci din volatilitate, iar o parte mai mică din corelație.

Comparația cere atenție. @ForbesRigobon2002 au arătat că corelația de selecție a randamentelor crește mecanic cînd crește volatilitatea, chiar dacă legătura dintre piețe nu s-a schimbat: un șoc comun mai mare domină zgomotul specific fiecărei piețe. O corelație mai mare a randamentelor brute într-o criză nu este deci, prin ea însăși, o dovadă de contagiune. DCC reduce acest efect, deoarece măsoară corelația pe reziduurile standardizate $z_t$, nu pe randamentele $r_t$, astfel încît creșterea volatilității este eliminată înainte de calculul corelației.

## Corelația DCC în crizele din 2008 și 2020 {h=8844e067}

Graficul are două panouri, cîte unul pentru fiecare criză: iunie 2008 – iunie 2009 în stînga și decembrie 2019 – august 2020 în dreapta. Fiecare panou arată, în funcție de timp, corelația DCC dintre S&P 500 și DAX pe axa din stînga, de la 0 la 1, și volatilitatea GARCH anualizată a S&P 500, în procente, pe axa din dreapta.

Cele două axe permit compararea momentelor în care se mișcă cele două mărimi, nu a nivelurilor lor. În ambele panouri linia volatilității urcă și coboară abrupt, în timp ce linia corelației se mișcă într-un interval mult mai îngust.

## Interpretarea celor două crize {h=0439f26e}

Cele două crize s-au desfășurat diferit. În 2008 volatilitatea S&P 500 a atins maximul de 90% pe 16 octombrie 2008, în timp ce corelația a urcat lent, de la 0,57 la maximul ei de 0,68, atins abia pe 23 iunie 2009, la cîteva luni după vîrful volatilității. Criza a pornit din sistemul financiar american și s-a extins treptat spre Europa.

În 2020 ordinea a fost inversă. Corelația a sărit la 0,73 pe 28 februarie 2020, înaintea vîrfului volatilității, de 109%, din 17 martie 2020. Un șoc global, pandemia de COVID-19 (*COronaVIrus Disease 2019*), a lovit ambele piețe în aceleași zile, iar succesiunea de scăderi comune a împins repede în sus recurența DCC.

Corelațiile și volatilitățile nu ating neapărat maximul în același timp. Un model care le leagă, cum este o singură covarianță pe ferestre mobile, ar amesteca cele două povești; modelarea lor separată, ca în DCC, arată diferența dintre o criză care se propagă și un șoc care lovește peste tot deodată.

## Recapitulare: S&P 500 și DAX {h=b00a6d75}

Pasul 1 dă volatilități GARCH persistente, cu persistența 0,980 și 0,984, care ating maximul simultan. Pasul 2 dă $a + b = 0{,}9918$ și un raport de verosimilitate de 131,8 față de CCC: corelația nu este constantă. Corelațiile sînt mai mari în 2008 și 2020, dar cea mai mare schimbare dintr-o criză apare în volatilitate.

## @end

Două active arată metoda; un portofoliu are de obicei mai multe. Secțiunea următoare estimează un singur model DCC pentru cinci piețe foarte diferite, inclusiv piața românească de acțiuni, leul și Bitcoin, și compară corelațiile lor în perioada calmă și în crahul din 2020.

# Cinci piețe: S&P 500, DAX, BET, EUR/RON, Bitcoin

Secțiunea estimează un singur model DCC pentru cinci active, doi indici bursieri mari, indicele românesc BET, cursul EUR/RON și Bitcoin, și compară corelațiile medii dintr-o perioadă calmă și din crahul COVID-19.

## Un model DCC cu cinci active {h=21b9b8aa}

Datele sînt randamente logaritmice zilnice în cele 2 815 zile în care se tranzacționează toate cele cinci piețe, din 6 ianuarie 2015. BET este indicele Bursei de Valori București, din fotografie. Pentru EUR/RON se folosește cursul de referință al Băncii Naționale a României (BNR), stabilit la ora 13:00 a Bucureștiului, iar pentru Bitcoin închiderea zilei în timpul universal coordonat (*Coordinated Universal Time*, UTC). Orele de închidere diferă și mai mult decît în prima secțiune, așa că și corelațiile zilnice ale acestor serii poartă deplasarea discutată acolo.

Pasul 1 dă cinci modele GARCH(1,1), cu persistența 0,961 pentru S&P 500, 0,965 pentru DAX, 0,941 pentru BET, 0,985 pentru EUR/RON și 0,958 pentru Bitcoin. Pasul 2 dă $\hat a = 0{,}0088$ (SE 0,0019) și $\hat b = 0{,}9782$ (SE 0,0062), deci timpul de înjumătățire al unui șoc de corelație este de 53 de zile. Raportul de verosimilitate față de CCC este 88,5, din nou mult peste valoarea critică de 5,99 la pragul de 5%.

Modelul are 10 perechi de corelații, dar o singură pereche de parametri de dinamică $(a, b)$ pentru toate. Cele 10 corelații pe termen lung provin din $\bar{\mathbf{Q}}$, corelația de selecție a reziduurilor standardizate, deci optimizarea din pasul 2 lucrează tot cu doar doi parametri.

## Corelațiile dinamice a patru perechi {h=beb6c481}

Graficul arată corelațiile DCC prognozate cu un pas înainte pentru patru dintre cele zece perechi, pe axa verticală, în funcție de timp, din 2015 pînă în 2026: S&P 500 și DAX, BET și DAX, S&P 500 și Bitcoin, BET și EUR/RON. O linie punctată marchează zero, iar banda colorată marchează crahul COVID-19, de la jumătatea lui februarie pînă în iunie 2020.

Cele patru linii sînt clar separate ca nivel: S&P 500 și DAX sus, BET și DAX și S&P 500 și Bitcoin la mijloc, iar BET și EUR/RON în jurul lui zero sau sub zero. Mișcările lor în timp sînt subiectul frame-ului următor.

## Interpretarea celor patru perechi {h=c4eadb6e}

S&P 500 și DAX au legătura cea mai puternică, cu o corelație medie de 0,54 în 2015–2026. BET și DAX au media 0,32 și variază între 0,15 și 0,53, maximul fiind atins în aprilie 2022. Saltul acestei perechi la începutul crahului din 2020 este cel mai mare dintre cele patru.

S&P 500 și Bitcoin variază între $-0{,}05$ și 0,42, maximul fiind atins în mai 2022, cu media 0,21. Înainte de 2020 corelația a rămas scăzută; după 2020 a rămas clar mai mare. Beneficiul de diversificare al lui Bitcoin pentru un portofoliu de acțiuni s-a redus.

BET și EUR/RON au o corelație apropiată de zero, cu media $-0{,}06$. Cursul este exprimat în lei pentru un euro, deci o creștere a EUR/RON înseamnă un leu mai slab, iar corelația negativă mică arată că un leu mai slab tinde să însoțească un BET mai slab. O trăsătură a graficului provine din model, nu din date: toate cele patru linii au același $(a, b)$, deci perechi care s-ar putea ajusta cu viteze diferite sînt forțate la același ritm. Acesta este prețul parcimoniei modelului DCC.

## Corelațiile medii: perioadă calmă și criză {h=285503ed}

Graficul arată două hărți termice (*heatmaps*) ale matricei medii a corelațiilor DCC prognozate cu un pas înainte pentru cele cinci active. Cea din stînga face media pe perioada calmă 2017–2019, iar cea din dreapta pe crahul COVID-19, de la jumătatea lui februarie pînă în iunie 2020, 89 de zile comune de tranzacționare. Rîndurile și coloanele sînt cele cinci active, fiecare celulă conține corelația medie, iar scala de culori merge de la $-1$ (albastru) la $1$ (roșu închis).

Diagonalele sînt 1 prin construcție. Celulele pentru EUR/RON sînt aproape albe în ambele matrice, în timp ce celulele care leagă BET și Bitcoin de piețele mari de acțiuni devin vizibil mai închise în panoul crizei.

## Interpretarea celor două matrice {h=7dabb0f5}

Media celor zece corelații dintre perechi crește de la 0,12 în perioada calmă la 0,19 în crah. Creșterea este inegală. Pentru BET și DAX corelația medie trece de la 0,28 la 0,48, pentru S&P 500 și BET de la 0,22 la 0,38, iar pentru DAX și Bitcoin de la 0,10 la 0,27. S&P 500 și DAX, deja puternic legate, trec de la 0,55 la 0,64.

Cele mai mari creșteri privesc BET și Bitcoin, activele slab legate de piețele mari în perioadele calme, adică tocmai activele de la care se aștepta diversificarea unui portofoliu de acțiuni. Cînd sosește un șoc global, aceste legături slabe se întăresc, iar beneficiul diversificării scade exact cînd contează mai mult.

EUR/RON rămîne aproape de zero sau ușor negativ față de toate piețele de acțiuni în ambele perioade, între $-0{,}03$ și $-0{,}10$. În acest set de cinci active, este singurul care și-a păstrat rolul de diversificare în crah.

## Recapitulare: cinci piețe {h=db8c3651}

Un singur model DCC descrie cele zece corelații a cinci active cu doar doi parametri de dinamică, iar raportul de verosimilitate de 88,5 respinge corelația constantă. Corelațiile pieței românești cu Europa și ale lui Bitcoin cu acțiunile se mișcă mult. În crahul din 2020 legăturile slabe s-au întărit cel mai mult, deci diversificarea a slăbit în criză.

## @end

Corelațiile și volatilitățile sînt date de intrare; managementul riscului are nevoie de numere care pot fi comparate cu rezultatele efective. Secțiunea următoare transformă matricea de covarianță condiționată în VaR-ul unui portofoliu și îl testează pe date care nu au fost folosite la estimare.

# VaR 1% al unui portofoliu cu DCC

Secțiunea calculează valoarea la risc a unui portofoliu pe baza lui $\mathbf{H}_t$, prezintă testele de backtesting Kupiec și Christoffersen și compară DCC, CCC și o matrice de covarianță statică pe un portofoliu S&P 500 și DAX, din 2015 pînă în 2026.

## De la Hₜ la VaR-ul unui portofoliu {h=c406edb7}

Un portofoliu cu ponderile $\mathbf{w}$ are randamentul $r_{p,t} = \mathbf{w}^\top\mathbf{r}_t$, cu media condiționată $\mu_p = \mathbf{w}^\top\boldsymbol{\mu}$ și volatilitatea condiționată $\sigma_{p,t} = \sqrt{\mathbf{w}^\top\mathbf{H}_t\mathbf{w}}$, conform @eq:pvar aplicată matricei de covarianță condiționată. VaR 1% este pierderea depășită cu probabilitatea 1%, adică minus cuantila de 1% a randamentului portofoliului:

$$ \mathrm{VaR}_{0{,}01,t} = -q_{0{,}01}(r_{p,t}) = -\left(\mu_p + q\,\sigma_{p,t}\right). $$ {#eq:var}

Valoarea lui $q$ depinde de distribuția presupusă a randamentului standardizat al portofoliului. Pentru distribuția Normală, $q = z_{0{,}01} = -2{,}326$. Simularea istorică filtrată (*filtered historical simulation*, FHS) folosește în schimb cuantila empirică de 1% a randamentelor standardizate trecute ale portofoliului, $(r_{p,t} - \mu_p)/\sigma_{p,t}$, care ține cont de cozile groase ale datelor, păstrînd dinamica volatilității din model.

Se compară trei variante pentru $\mathbf{H}_t$: DCC; CCC, cu volatilități GARCH, dar cu o corelație constantă; și o matrice statică, o singură matrice de covarianță de selecție folosită pentru toate zilele. VaR-ul static este același în fiecare zi, VaR-ul CCC se mișcă doar odată cu volatilitățile, iar VaR-ul DCC se mișcă atît cu volatilitățile, cît și cu corelația.

## Exemplu rezolvat: VaR 1% pentru două active {h=8607e4fa}

Prognozele de azi sînt volatilitățile zilnice $\sigma_1 = 1{,}5\%$ și $\sigma_2 = 1{,}2\%$, ponderile sînt 50% și 50%, iar media este zero. Într-o stare calmă, cu $\rho_t = 0{,}3$, @eq:pvar dă

$$ \begin{aligned} \sigma_p^2 &= 0{,}25 \cdot 2{,}25 + 0{,}25 \cdot 1{,}44 \\ &\quad + 2 \cdot 0{,}25 \cdot 0{,}3 \cdot 1{,}8, \end{aligned} $$

unde $1{,}8 = 1{,}5 \cdot 1{,}2$ este produsul volatilităților. Volatilitatea portofoliului este $\sigma_p = 1{,}092\%$, iar VaR 1% normal este $2{,}326 \cdot 1{,}092 = 2{,}54\%$.

Într-o stare de criză, cu $\rho_t = 0{,}9$ și aceleași volatilități, $\sigma_p = 1{,}316\%$, iar VaR 1% este 3,06%. Doar saltul corelației adaugă aproximativ o cincime la VaR. Într-o criză reală cresc și volatilitățile, deci creșterea efectivă este mai mare.

Pentru un portofoliu de 1 milion de lei, VaR 1% este de aproximativ 25,4 mii de lei în cazul calm și de 30,6 mii de lei în cazul de criză. Un model cu corelație fixă ar raporta primul număr în ambele zile.

## Backtesting pentru VaR (1/2): frecvența {h=a1113139}

Un model VaR se evaluează prin backtesting: compararea prognozelor lui cu randamentele care au urmat efectiv. O încălcare (*violation*) are loc în ziua $t$ cînd $r_{p,t} < -\mathrm{VaR}_{0{,}01,t}$, adică atunci cînd pierderea depășește VaR-ul. Un VaR 1% corect are încălcări în 1% din zile, iar aceste zile apar la întîmplare, nu grupate. Cele două proprietăți se testează separat.

Testul lui @Kupiec1995 verifică frecvența. Cu $x$ încălcări în $n$ zile și rata observată $\hat p = x/n$, statistica

$$ LR_{uc} = -2\log\frac{(1-0{,}01)^{n-x}\,0{,}01^{x}}{(1-\hat p)^{n-x}\,\hat p^{x}} $$ {#eq:kupiec}

compară verosimilitatea încălcărilor observate la rata nominală de 1% cu verosimilitatea lor la rata observată. Indicele $uc$ vine de la *unconditional coverage* (acoperirea necondiționată). Pentru un model corect $LR_{uc}$ urmează aproximativ o distribuție $\chi^2(1)$, iar modelul este respins la 5% dacă $LR_{uc} > 3{,}84$. Testul respinge atît prea multe încălcări, un VaR prea mic, cît și prea puține, un VaR prea mare.

## Backtesting pentru VaR (2/2): independența {h=77b71026}

Testul lui @Christoffersen1998 verifică independența: este o încălcare mai probabilă în ziua de după o încălcare? El compară $\pi_{11}$, probabilitatea unei încălcări dat fiind că ieri a avut loc o încălcare, cu $\pi_{01}$, probabilitatea unei încălcări dat fiind că ieri nu a avut loc. Pentru un model corect cele două sînt egale, deoarece un VaR care folosește toată informația disponibilă nu lasă nimic previzibil în încălcări.

Statistica se construiește din numărări: $n_{ij}$ este numărul de zile în starea $j$ care urmează unei zile în starea $i$, unde starea 1 este o încălcare, iar 0 lipsa ei. Estimațiile sînt $\hat\pi_{01} = n_{01}/(n_{00}+n_{01})$ și $\hat\pi_{11} = n_{11}/(n_{10}+n_{11})$, iar $\hat\pi$ este rata totală a încălcărilor. Statistica

$$ LR_{ind} = -2\log\frac{(1-\hat\pi)^{n_{00}+n_{10}}\,\hat\pi^{\,n_{01}+n_{11}}}{(1-\hat\pi_{01})^{n_{00}}\,\hat\pi_{01}^{\,n_{01}}\,(1-\hat\pi_{11})^{n_{10}}\,\hat\pi_{11}^{\,n_{11}}} $$

are un numărător care presupune $\pi_{01} = \pi_{11}$, adică lipsa grupării, și un numitor care permite celor două probabilități să difere. Sub ipoteza de independență ea urmează aproximativ $\chi^2(1)$; un p-value mic respinge independența.

Un backtesting are sens doar în afara eșantionului (*out-of-sample*). Toți parametrii se estimează pe 2000–2014, 3 712 zile, iar filtrele se rulează apoi mai departe pe 2015–2026, 2 893 de zile, fără reestimare. VaR-ul fiecărei zile folosește doar informația disponibilă în ziua precedentă, ca în practică.

## VaR 1% al unui portofoliu S&P 500 și DAX, 2015–2026 {h=e60f9105}

Graficul arată, în funcție de timp, din 2015 pînă în 2026, randamentul zilnic, în procente, al unui portofoliu cu 50% în S&P 500 și 50% în DAX, împreună cu minus VaR 1% pentru două modele: DCC cu simulare istorică filtrată, o linie care se mișcă, și matricea de covarianță statică, o linie orizontală. Triunghiurile roșii marchează cele 48 de încălcări ale modelului DCC-FHS, iar cele verzi cele 26 de încălcări ale modelului static.

Linia DCC urmărește volatilitatea randamentelor: rămîne aproape de randamente în anii calmi și coboară mult sub ele în martie 2020, cînd randamentele însele oscilează puternic. Linia statică rămîne la același nivel tot timpul. Triunghiurile verzi ale modelului static sînt concentrate în cîteva episoade, cel mai vizibil la începutul lui 2020; triunghiurile roșii ale modelului DCC sînt răspîndite pe toată perioada.

## Interpretarea backtesting-ului {h=719462de}

Cu 2 893 de zile în afara eșantionului, un VaR 1% corect ar trebui să aibă aproximativ 29 de încălcări. Tabelul dă, pentru fiecare model, numărul și rata încălcărilor, statistica Kupiec $LR_{uc}$ și p-value-ul ei, p-value-ul testului de independență Christoffersen și numărul de încălcări din februarie – aprilie 2020. DCC cu cuantile normale are 65 de încălcări, o rată de 2,25%, $LR_{uc} = 33{,}6$ cu un p-value sub 0,001 și un p-value al testului de independență de 0,003. DCC cu FHS are 48 de încălcări, 1,66%, $LR_{uc} = 10{,}6$, p-value 0,001, p-value de independență 0,008. CCC cu cuantile normale are 57 de încălcări, 1,97%, $LR_{uc} = 21{,}4$, p-value sub 0,001, p-value de independență 0,030. Modelul static are 26 de încălcări, 0,90%, $LR_{uc} = 0{,}3$, p-value 0,577, p-value de independență 0,001.

Modelele dinamice cu cuantilele distribuției Normale au prea multe încălcări, deoarece reziduurile standardizate au cozi mai groase decît distribuția Normală, așa cum a arătat kurtosis-ul din pasul 1 și cum a găsit [Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf) pentru serii individuale. FHS înlocuiește $-2{,}326$ cu cuantila empirică $q = -2{,}59$ și reduce rata la 1,66%, încă peste 1%, dar mult mai aproape. Încălcările rămase sînt și mai puțin grupate decît la modelul static: în februarie – aprilie 2020 fiecare model dinamic are 4 încălcări.

VaR-ul static trece testul de frecvență, iar acesta este cazul cel mai instructiv. Cele 26 de încălcări ale lui sînt aproape de cele 29 așteptate, dar 10 dintre ele cad în februarie – aprilie 2020, iar după o încălcare ziua următoare este tot o încălcare cu probabilitatea 11,5%, deci independența este respinsă. VaR-ul static este 2,95% în fiecare zi: prea mare în anii calmi și mult prea mic într-un crah. Frecvența lui medie corectă provine din erori în ambele sensuri care se compensează. Un VaR dinamic se adaptează în ambele sensuri; problema lui rămasă este cuantila, nu covarianța.

## Recapitulare: VaR-ul portofoliului {h=f96969dc}

Volatilitatea portofoliului este $\sigma_{p,t} = \sqrt{\mathbf{w}^\top\mathbf{H}_t\mathbf{w}}$, iar VaR 1% este $-(\mu_p + q\,\sigma_{p,t})$. Un VaR se verifică prin backtesting în afara eșantionului, cu testul Kupiec, pentru frecvența încălcărilor, și cu testul Christoffersen, pentru independența lor. Covarianțele dinamice elimină cea mai mare parte a grupării încălcărilor; cozile groase cer o cuantilă mai bună decît cea a distribuției Normale.

## @end

VaR-ul folosește întreaga matrice $\mathbf{H}_t$ printr-o singură formă pătratică. Aplicația următoare îi folosește elementele separat: raportul dintre o covarianță și o varianță dă numărul de unități dintr-un activ de acoperire care trebuie vîndute, iar DCC permite ca acest număr să se schimbe zilnic.

# Rapoarte dinamice de acoperire

Secțiunea deduce raportul de acoperire cu varianță minimă și eficiența acoperirii, face raportul dinamic cu ajutorul modelului DCC și compară în afara eșantionului acoperirile statică, pe ferestre mobile și DCC ale BET cu DAX.

## Raportul de acoperire cu varianță minimă {h=e31d7d28}

Un investitor deține o unitate dintr-un activ $s$, de exemplu acțiuni românești reprezentate de BET, și vinde $h$ unități dintr-un activ de acoperire $f$, de exemplu un contract futures pe DAX. Randamentul acoperit este $r_{s,t} - h\,r_{f,t}$, iar varianța lui este $\sigma_s^2 - 2h\,\sigma_{sf} + h^2\sigma_f^2$, unde $\sigma_s^2$ și $\sigma_f^2$ sînt varianțele celor două randamente, iar $\sigma_{sf}$ covarianța lor. Anulînd derivata în raport cu $h$ se obține raportul de acoperire (*hedge ratio*) cu varianță minimă al lui @Ederington1979:

$$ h^* = \frac{\sigma_{sf}}{\sigma_f^2} = \rho\,\frac{\sigma_s}{\sigma_f}. $$ {#eq:hedge}

Versiunea statică a lui $h^*$ este panta regresiei prin metoda celor mai mici pătrate (*ordinary least squares*, OLS) a lui $r_s$ pe $r_f$, deoarece panta OLS este exact o covarianță împărțită la o varianță. Versiunea dinamică, propusă de @KronerSultan1993, folosește elementele lui $\mathbf{H}_t$ dintr-un model MGARCH, $h_t^* = h_{sf,t}/h_{ff,t}$, și se schimbă deci zilnic odată cu covarianța și varianța prognozate.

Eficiența acoperirii (*hedging effectiveness*) măsoară ponderea varianței eliminate: $HE = 1 - \Var(r_s - h\,r_f)/\Var(r_s)$. Înlocuind raportul optim constant în varianța poziției acoperite se obține $\sigma_s^2(1 - \rho^2)$, deci $HE = \rho^2$. O acoperire este doar atît de bună cît este corelația dintre activ și instrumentul de acoperire.

## Exemplu rezolvat: un raport de acoperire {h=4415a812}

Considerăm volatilitățile zilnice $\sigma_s = 1{,}2\%$ și $\sigma_f = 1{,}0\%$ și corelația $\rho = 0{,}8$. Covarianța este $\sigma_{sf} = \rho\,\sigma_s\sigma_f = 0{,}8 \cdot 1{,}2 \cdot 1{,}0 = 0{,}96$, iar conform @eq:hedge raportul de acoperire este $h^* = 0{,}96/1 = 0{,}96$: investitorul vinde 0,96 unități din $f$ pentru fiecare unitate din $s$.

Eficiența acoperirii este $\rho^2 = 0{,}64$, deci acoperirea elimină 64% din varianță. Volatilitatea poziției acoperite este $\sigma_s\sqrt{1 - \rho^2} = 0{,}72\%$ în loc de 1,2%. Volatilitatea scade mai puțin decît varianța, cu 40% în loc de 64%, deoarece este rădăcina ei pătrată.

Dacă mîine modelul DCC prognozează $\rho_t = 0{,}5$ și $\sigma_{f,t} = 2\%$, cu $\sigma_s$ neschimbat, raportul devine $0{,}5 \cdot 1{,}2/2 = 0{,}3$. Atît corelația mai mică, cît și volatilitatea mai mare a activului de acoperire reduc numărul de unități de vîndut, iar poziția trebuie ajustată.

## Acoperirea BET cu DAX, 2015–2026 {h=1dd6a3b9}

Graficul arată trei rapoarte de acoperire, în unități de DAX vîndute pentru o unitate de BET, în funcție de timp, din 2015 pînă în 2026: raportul DCC $h_t = \hat\rho_t\hat\sigma_{BET,t}/\hat\sigma_{DAX,t}$, panta OLS pe ferestre mobile din ultimele 250 de zile și raportul OLS static estimat pe 2005–2014, ca linie orizontală întreruptă. Datele sînt randamente zilnice în zilele comune de tranzacționare ale celor două piețe din 2005, iar toți parametrii modelului DCC sînt estimați pe 2005–2014.

Raportul DCC este linia cea mai agitată, cu salturi scurte, cel mai mare la sfîrșitul lui 2018. Raportul OLS pe ferestre mobile se mișcă în trepte, iar saltul lui mare de la începutul lui 2020 durează aproximativ un an, lungimea ferestrei. Linia statică se află deasupra ambelor rapoarte dinamice în cea mai mare parte a perioadei.

## Interpretarea rapoartelor de acoperire {h=6e362978}

Raportul static de 0,51 reflectă perioada 2005–2014, care include criza din 2008, cînd BET și DAX erau strîns legate. După 2015 raportul DCC are media 0,29 și variază între 0,07 și 1,00, maximul fiind atins pe 20 decembrie 2018. Raportul static supraacoperă deci în cea mai mare parte a perioadei ulterioare.

Eficiența acoperirii în afara eșantionului, pe 2 893 de zile, este de 13,6% pentru raportul static, 18,0% pentru raportul OLS pe ferestre mobile și 19,1% pentru raportul DCC. Rapoartele dinamice elimină mai mult risc, iar DCC face acest lucru fără alegerea unei ferestre.

Toate cele trei valori sînt modeste. Corelația DCC dintre BET și DAX are media 0,32 din 2015, deci $\rho^2$ este de aproximativ o zecime, iar niciun raport nu poate elimina mare parte din varianța BET cu ajutorul DAX. Un raport DCC zilnic implică și reechilibrare zilnică și costuri de tranzacționare, care nu sînt incluse în aceste cifre; după costuri, avantajul modelului DCC față de raportul static ar fi mai mic.

## Recapitulare: acoperirea riscului {h=4f9d6ad6}

Raportul dinamic de acoperire cu varianță minimă este $h_t^* = h_{sf,t}/h_{ff,t} = \rho_t\sigma_{s,t}/\sigma_{f,t}$, iar eficiența acoperirii este ponderea varianței eliminate. Pentru BET acoperit cu DAX, DCC elimină în afara eșantionului 19,1% din varianță, față de 13,6% pentru raportul static. O acoperire trebuie evaluată în afara eșantionului și după costurile de tranzacționare.

## @end

Aplicațiile au arătat ce adaugă în practică matricea de covarianță condiționată. Secțiunea următoare așază capitolul printre cele anterioare și enumeră ce nu poate face modelul DCC.

# Legături cu alte capitole și limite

Secțiunea leagă modelele GARCH multivariate de GARCH-ul univariat, de modelele VAR și de cointegrare și enumeră principalele limite ale modelului DCC și extensiile care le tratează.

## Legături cu Capitolele 5, 6 și 7 {h=74a5c72b}

Orice model MGARCH conține modele GARCH univariate: diagonala lui $\mathbf{H}_t$ este formată din varianțe condiționate, ca în [Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf). În DCC legătura este literală, deoarece modelele GARCH univariate sînt chiar pasul 1 al estimării.

Un model VAR din [Capitolul 6](../RO/Cursuri/capitol6_modele_var_cauzalitate_granger.pdf) descrie transmiterea în medie, de exemplu dacă randamentul DAX de ieri ajută la prognoza randamentului BET de azi. Un model BEKK cu o matrice $\mathbf{A}$ completă descrie transmiterea în varianță, adică dacă șocul de ieri al DAX crește varianța de azi a BET. Cele două se combină firesc: un VAR pentru media $\boldsymbol{\mu}_t$ și un model MGARCH pentru $\mathbf{H}_t$. @DieboldYilmaz2009 au construit indici de transmitere (*spillover indices*) din descompunerea varianței unui VAR.

Cointegrarea, din [Capitolul 7](../RO/Cursuri/capitol7_cointegrare_vecm.pdf), este o legătură pe termen lung între nivelurile prețurilor; corelația este o legătură pe termen scurt între randamente. Cele două sînt distincte. Două prețuri pot fi puternic corelate de la o zi la alta și totuși să se îndepărteze definitiv, sau pot fi cointegrate cu o corelație zilnică mică. Un model vectorial cu corecția erorii (*vector error correction model*, VECM) poate avea erori de tip DCC, cu echilibru în medie și corelație dinamică în șocuri.

## Limite și extensii {h=151eb604}

DCC își plătește parcimonia cu restricții. O singură pereche $(a, b)$ guvernează toate corelațiile, deci perechi care se ajustează cu viteze diferite sînt forțate la același ritm, așa cum a arătat modelul cu cinci piețe. Volatilitățile provin din modele GARCH univariate separate, deci DCC nu are transmitere între volatilități. Iar cvasi-verosimilitatea Normală nu spune nimic despre cozi: pentru măsurile de risc sînt necesare inovații Student $t$ sau simularea istorică filtrată, așa cum a arătat backtesting-ul.

Corelațiile reacționează și asimetric: cresc mai mult după scăderi comune decît după creșteri comune, lucru pe care îl surprinde DCC-ul asimetric al lui @CappielloEngleSheppard2006. Pentru sute de active chiar și DCC devine greu de folosit, deoarece matricea de selecție $\bar{\mathbf{Q}}$ este zgomotoasă, iar verosimilitatea din pasul 2 cere inversarea unei matrice mari în fiecare zi; modelele factoriale, verosimilitatea compusă (*composite likelihood*) și shrinkage aplicat lui $\bar{\mathbf{Q}}$ sînt răspunsurile obișnuite.

În sfîrșit, corelația măsoară doar dependența liniară. Două active pot avea o corelație moderată și totuși să scadă brusc împreună mult mai des decît implică un model normal. Extremele comune se descriu prin copule și prin dependența în cozi (*tail dependence*); @LonginSolnik2001 au studiat direct corelația piețelor internaționale de acțiuni în cozi.

## @end

DCC este instrumentul de bază al volatilității multivariate, deoarece împarte o problemă dificilă în modele GARCH univariate și o recurență a corelațiilor cu doi parametri. Limitele lui, o viteză comună pentru toate perechile, lipsa transmiterii între volatilități și cozile normale, sînt cunoscute și au extensii standard.

# Contribuția posibilă a AI

Secțiunea descrie ce poate face un asistent de inteligență artificială (*artificial intelligence*, AI) în modelarea volatilității multivariate și ce trebuie verificat de fiecare dată.

## Contribuția posibilă a AI {h=b31a8d40}

Un asistent AI poate scrie repede o primă versiune a codului de care are nevoie capitolul: verosimilitatea de corelație a modelului DCC, un filtru BEKK sau testul Kupiec. Poate da și o a doua explicație, formulată altfel, a ideilor mai greu de prins la prima lectură, cum sînt descompunerea $\mathbf{D}_t\mathbf{R}_t\mathbf{D}_t$, țintirea corelației sau formula raportului de acoperire.

Cel mai mare cîștig apare în explorare: modele DCC pentru multe perechi deodată, de exemplu piețele din Europa Centrală și de Est, sectoare sau valute, și o primă privire asupra comportamentului corelațiilor lor în crize. Prompt-ul de pe slide este un exemplu de cerere precisă: numește datele, alinierea pe zilele comune de tranzacționare, cei doi pași ai estimării, țintirea corelației și comparația în afara eșantionului cu OLS static. Cu cît cererea fixează mai multe dintre aceste alegeri, cu atît rezultatul este mai ușor de verificat.

## Verificări necesare {h=32d23b1b}

Prima verificare este biblioteca. Pachetul arch nu are model DCC, deci un răspuns care apelează arch_model(..., vol='DCC') a inventat o funcție. A doua este momentul: $\mathbf{Q}_t$ trebuie să folosească $\mathbf{z}_{t-1}$, nu $\mathbf{z}_t$; altfel VaR-ul și raportul de acoperire folosesc informația de mîine, iar backtesting-ul pare mai bun decît ar putea fi orice prognoză reală. A treia este rescalarea din @eq:dcc: fiecare $\mathbf{R}_t$ trebuie să aibă 1 pe diagonală și corelații mai mici decît 1 în valoare absolută.

Celelalte verificări privesc datele și evaluarea. Prețurile trebuie aliniate pe zilele comune de tranzacționare înainte de calculul randamentelor, iar închiderile asincrone trebuie avute în vedere. Modelele trebuie estimate pe o perioadă și testate pe alta. Un test simplu al oricărui cod DCC este o simulare: se generează date dintr-un DCC cu $(a, b)$ cunoscuți și se verifică dacă programul îi regăsește. Fiecare referință sugerată de un asistent trebuie să existe, iar identificatorul ei digital (*digital object identifier*, DOI) trebuie verificat.

## @end

Un asistent scurtează scrierea codului, dar nu înlocuiește verificările specifice modelelor multivariate: momentul recurenței, validitatea fiecărei matrice de corelații, alinierea calendarelor și o evaluare onestă în afara eșantionului.

# Rezumat

Rezumatul adună ideile principale ale capitolului, formulele de reținut și întrebările de autoevaluare.

## Idei de reținut {h=8ffe7244}

Riscul portofoliului, acoperirea riscului și contagiunea depind de matricea de covarianță condiționată $\mathbf{H}_t$, care se schimbă în timp. VEC este cel mai general model liniar, dar are mult prea mulți parametri; BEKK este pozitiv definit prin construcție și măsoară transmiterea volatilității. CCC și DCC descompun $\mathbf{H}_t = \mathbf{D}_t\mathbf{R}_t\mathbf{D}_t$ și se estimează în doi pași, iar pasul 2 al modelului DCC are nevoie doar de $(a, b)$, oricare ar fi numărul de active.

Datele arată că corelațiile reale se mișcă: prin integrarea pieței românești cu Europa, prin crize și prin apariția lui Bitcoin ca activ riscant. Corelația constantă este respinsă pentru S&P 500 și DAX și pentru modelul cu cinci piețe. Aplicațiile transformă $\mathbf{H}_t$ în VaR 1% al unui portofoliu, evaluat prin backtesting, și în rapoarte dinamice de acoperire, evaluate în afara eșantionului.

## Formule de reținut {h=35297833}

Tabelul adună formulele capitolului: corelația condiționată @eq:rho; modelul BEKK @eq:bekk; descompunerea @eq:drd folosită de CCC și DCC; recurența DCC cu rescalarea ei @eq:dcc; timpul de înjumătățire $\ln 0{,}5/\ln(a + b)$; VaR 1% al portofoliului @eq:var; raportul de acoperire @eq:hedge, împreună cu eficiența acoperirii.

Formulele formează un lanț. Descompunerea separă ce tratează deja modelele GARCH univariate de corelații; recurența DCC face corelațiile dinamice; timpul de înjumătățire rezumă persistența lor; iar VaR-ul și raportul de acoperire transformă matricea $\mathbf{H}_t$ obținută în decizii.

## Autoevaluare (1/2) {h=6a931074}

Prima întrebare aplică numărarea parametrilor BEKK-ului diagonal: pentru $N = 3$ matricea constantă are $N(N+1)/2 = 6$ elemente distincte, iar diagonalele lui $\mathbf{A}$ și $\mathbf{B}$ au $2N = 6$, deci 12 în total. A doua aplică @eq:rho și condiția de caracter pozitiv definit: cu $h_{11} = 1$, $h_{22} = 4$ și $h_{12} = 1{,}2$ corelația este $1{,}2/\sqrt{4} = 0{,}6$, iar determinantul $1 \cdot 4 - 1{,}44$ este pozitiv, deci matricea este pozitiv definită.

A treia întrebare este un pas din @eq:dcc cu $a = 0{,}02$, $b = 0{,}97$, $\bar q_{12} = 0{,}4$ și $z_{t-1} = (1, -1)$. Ponderea termenului pe termen lung este 0,01, iar produsul reziduurilor este $-1$, deci

$$ \begin{aligned} q_{12,t} &= 0{,}01 \cdot 0{,}4 + 0{,}02 \cdot (-1) + 0{,}97 \cdot 0{,}4 \\ &= 0{,}372: \end{aligned} $$

mișcările în sens opus reduc corelația. A patra calculează timpul de înjumătățire pentru $a + b = 0{,}95$: $\ln 0{,}5/\ln 0{,}95 \approx 13{,}5$ zile.

## Autoevaluare (2/2) {h=3cb6295d}

Prima întrebare aplică @eq:hedge: cu $\sigma_s = 2\%$, $\sigma_f = 1\%$ și $\rho = 0{,}5$ raportul de acoperire este $0{,}5 \cdot 2/1 = 1$, iar eficiența acoperirii $\rho^2 = 0{,}25$. A doua privește backtesting-ul: 26 de încălcări în 2 500 de zile sînt aproape de cele 25 așteptate, deci testul de frecvență este trecut, dar 10 încălcări într-o singură lună formează o grupare pe care testul de independență o respinge și arată un VaR care nu se adaptează crizelor, ca VaR-ul static al portofoliului S&P 500 și DAX.

A treia întrebare revine la tranzacționarea asincronă: știrile de la New York ajung la București abia în ziua următoare de tranzacționare, deci randamentele zilnice din aceeași zi calendaristică nu se potrivesc, iar corelația zilnică este mai mică decît cea săptămînală. A patra privește contagiunea: o corelație mai mare a randamentelor într-o criză nu dovedește, prin ea însăși, contagiunea, deoarece volatilitatea mai mare crește mecanic corelația de selecție [@ForbesRigobon2002]; comparația trebuie făcută pe corelațiile reziduurilor standardizate. [Capitolul 15](../RO/Cursuri/capitol15_recapitulare.pdf) recapitulează întregul curs.

## @end

Capitolul a extins modelarea volatilității de la o singură serie la mai multe. Obiectul central este matricea de covarianță condiționată, iar ideea centrală a modelului DCC este modelarea separată a volatilităților și a corelațiilor, cu modele GARCH univariate pentru primele și o recurență cu doi parametri pentru celelalte. Pe date reale corelațiile se mișcă, cresc în crize cel mai mult pentru legăturile cele mai slabe și schimbă riscul unui portofoliu și mărimea unei acoperiri.
