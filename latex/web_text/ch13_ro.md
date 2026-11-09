<!--
Textul de curs al versiunii web, capitolul 13 (RO). Generat în pagină de latex/build_web_lecture.py.
Format: # titlul secțiunii (introducere), ## titlul slide-ului {h=hash} (text), @after (text după blocul interactiv), ## @end (sinteză).
Ecuații numerotate: $$ ... $$ {#eq:eticheta}, trimitere @eq:eticheta. Citări: @Cheie -> Autor (an), [@Cheie] -> (Autor, an).
Hash-urile curente: python3 build_web_lecture.py 13 --hashes
-->

## @intro

@text
La cîțiva ani o dată, o piață crește mult mai repede decît veniturile pe care le poate aduce, apoi scade. Indicele Nasdaq 100 al acțiunilor tehnologice s-a mărit de peste patru ori între octombrie 1998 și martie 2000 și a pierdut două treimi din valoare în anul următor; Bitcoin și-a înmulțit prețul de aproximativ 26 de ori în 2017, apoi a pierdut 83%. După ce s-au încheiat, astfel de episoade sînt ușor de numit. Întrebarea capitolului este mai grea: poate fi recunoscută o bulă speculativă (*speculative bubble*) în seria prețurilor cît timp încă crește și poate fi estimat momentul în care se încheie?

Capitolul dă două răspunsuri statistice. Primul vine din teoria așteptărilor raționale: o bulă pe care investitorii o dețin în cunoștință de cauză trebuie să crească, în medie, exploziv, așa că lasă în prețul logaritmic urma unei rădăcini autoregresive mai mari decît 1. Testele de rădăcină unitară la dreapta ale lui Phillips, Shi și Yu caută această urmă și datează episoadele. Al doilea vine din modelul LPPL (*log-periodic power law*, legea de putere log-periodică) al lui Johansen, Ledoit și Sornette, care descrie o bulă ca o creștere mai rapidă decît cea exponențială, însoțită de oscilații tot mai rapide, care se încheie la un timp critic $t_c$.

Ambele instrumente sînt aplicate închiderilor zilnice și săptămînale ale indicilor S&P 500, Nasdaq 100, BET (*Bucharest Exchange Trading*), Shanghai Composite și ale Bitcoin. Rezultatele sînt amestecate, iar capitolul le prezintă așa cum sînt. Testele de rădăcini explozive datează bine creșterile, dar tîrziu și fără să spună cînd se vor încheia. Ajustările LPPL descriu creșterile îndeaproape, însă timpul lor critic se mută odată cu datele, iar pe tot eșantionul alarmele indicatorului LPPL nu sînt urmate de crahuri mai des decît zilele obișnuite. Judecarea unei astfel de metode cere un protocol fixat dinainte, cu alarme false și frecvențe de bază.

@chart tsa_ch13_episodes
Șase creșteri și crahuri din datele cursului: închideri zilnice pe scară logaritmică, de la șase luni înainte de minimul creșterii pînă la optsprezece luni după maximum. Zona hașurată este creșterea de la minim la maximum; linia roșie punctată marchează maximul.

@goals
- **Descrieți** o bulă prin valoarea fundamentală și prin semnăturile ei statistice.
- **Explicați** de ce o bulă rațională trebuie să crească exploziv și **testați** rădăcinile explozive cu statisticile SADF și GSADF.
- **Scrieți** ecuația LPPL și **interpretați** fiecare dintre cei șapte parametri ai ei.
- **Estimați** modelul LPPL cu metoda în doi pași a lui Filimonov și Sornette și **verificați** condițiile filtrului.
- **Construiți** un indicator de încredere din multe ferestre și **evaluați** alarmele lui cu alarme false și frecvențe de bază.

@plan
- Motivație | Întrebarea capitolului, traseul, bibliografia și instrumentele.
- Bule în istorie | Bule speculative începînd din 1636, valoarea fundamentală și bula, șase episoade din datele cursului.
- Bule raționale și rădăcini explozive | Creșterea explozivă a unei bule raționale; testele SADF, GSADF și BSADF pe Nasdaq 100, Bitcoin, BET și Shanghai.
- Modelul LPPL | Crahurile ca puncte critice, creșterea superexponențială, rata de hazard și ecuația LPPL, log-periodicitatea și filtrul.
- Estimarea | Metoda în doi pași a lui Filimonov și Sornette, peisajul funcției de cost și șase ajustări cu 30 de zile înainte de maximum.
- Estimarea pe mai multe ferestre | Indicatorul de încredere LPPLS, comportamentul lui în jurul a patru maxime și un timp critic care se mută odată cu datele.
- Evaluarea prognozelor de crah | Look-ahead bias, selecția, alarmele false și frecvențele de bază pe S&P 500 și Bitcoin.
- Contribuția posibilă a AI | Ce poate scrie un asistent și ce trebuie verificat.
- Rezumat | Ideile principale, formulele, autoevaluarea și o idee de proiect.

# Motivație

Prima secțiune formulează întrebarea capitolului, explică modul în care capitolul trebuie studiat și prezintă rezultatele învățării, bibliografia și instrumentele.

## Întrebarea capitolului {h=4031fd8e}

Întrebarea este dacă o bulă speculativă poate fi recunoscută într-o serie de prețuri înainte să se spargă. Două definiții o fac precisă. O bulă speculativă este un preț care crește mult peste valoarea justificată de fluxurile de numerar așteptate ale activului, susținut de așteptarea că activul poate fi vîndut mai tîrziu altcuiva, la un preț mai mare. Un crah (*crash*) este o scădere mare a prețului într-un timp scurt; în acest capitol, o scădere de cel puțin 20% față de maximum. Cele două noțiuni sînt distincte: o bulă se poate încheia cu o scădere lentă în loc de un crah, iar un crah poate veni fără o bulă înaintea lui.

Traseul are patru etape. Istoria bulelor oferă exemplele și semnăturile statistice care trebuie căutate. Bulele raționale duc la rădăcini autoregresive explozive și la testele de rădăcină unitară la dreapta ale lui Phillips, Shi și Yu, care inversează testul Dickey–Fuller din [Capitolul 3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf). Modelul LPPL al lui Johansen, Ledoit și Sornette adaugă o formă pentru creștere și o dată pentru sfîrșitul ei, timpul critic. Ultima etapă este statistică: estimarea modelului, combinarea mai multor ferestre de estimare într-un indicator și evaluarea indicatorului pe tot eșantionul, inclusiv în perioadele în care nu s-a întîmplat nimic.

## Ghid de studiu individual {h=519abd11}

Acest capitol este scris pentru studiu individual și nu are seminar. Fiecare secțiune se încheie cu o recapitulare a ideilor principale, iar exemplele rezolvate sînt făcute pentru a fi refăcute pe hîrtie înainte de citirea soluției: bula Blanchard–Watson, fereastra din Nasdaq 100, valoarea LPPL calculată de mînă și ajustarea pentru Shanghai cer doar un calculator. După fiecare grafic, un slide de interpretare dă cifrele din spatele lui; exercițiul util este să citiți întîi graficul și să comparați propria lectură cu interpretarea.

Notebook-ul cursului produce fiecare grafic și fiecare cifră din capitol. Modificarea unei setări, de exemplu lungimea ferestrei de estimare, condițiile filtrului sau pragul de alarmă, și urmărirea felului în care se schimbă concluzia este cea mai bună cale de a înțelege cît de fragile sînt unele rezultate. Autoevaluarea de la final și quiz-ul capitolului de pe site-ul cursului verifică ideile principale. Cunoștințele necesare sînt prețurile și randamentele logaritmice din [Capitolul 1](../RO/Cursuri/capitol1_procese_stochastice_stationaritate.pdf), testul Dickey–Fuller din [Capitolul 3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf) și regresia prin metoda celor mai mici pătrate (*ordinary least squares*, OLS).

## Rezultatele învățării {h=7ae8acc4}

Primele două rezultate sînt economice și econometrice. O bulă este definită ca diferența dintre preț și o valoare fundamentală (*fundamental value*) neobservată, deci poate fi studiată doar prin consecințele ei asupra traiectoriei prețului. Consecința principală, pentru o bulă deținută de investitori raționali, este creșterea explozivă; testarea ei folosește statisticile SADF (*supremum augmented Dickey–Fuller*) și GSADF (*generalised supremum augmented Dickey–Fuller*), care calculează regresia Dickey–Fuller augmentată (*augmented Dickey–Fuller*, ADF) din [Capitolul 3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf) pe multe subeșantioane.

Celelalte rezultate privesc modelul LPPL. Ecuația lui are șapte parametri, fiecare cu o interpretare: timpul critic, nivelul, mărimea și exponentul creșterii de tip lege de putere, apoi amplitudinea, frecvența și faza oscilațiilor. Estimarea folosește metoda în doi pași a lui Filimonov și Sornette, iar fiecare ajustare (*fit*) este verificată cu un set de condiții de filtrare înainte să fie numărată ca semnal de bulă. În final, multe ajustări pe ferestre de lungimi diferite sînt combinate într-un indicator de încredere (*confidence indicator*), iar alarmele indicatorului sînt judecate față de frecvența de bază a crahurilor, cu toate alarmele false numărate.

## Bibliografie și instrumente {h=1239ce35}

@HuangPetukhina2022 tratează instrumentele generale folosite aici, testele de rădăcină unitară și metoda celor mai mici pătrate neliniare, în Python. Modelul LPPL provine de la @JohansenLedoitSornette2000; @Sornette2017 prezintă povestea pentru un public mai larg, iar @FilimonovSornette2013 dau metoda de calibrare folosită în capitol. Testele de rădăcini explozive provin de la @PhillipsWuYu2011 și @PhillipsShiYu2015a, cu o sinteză în @PhillipsShi2020. Pentru istoria bulelor, @KindlebergerAliber2005, @Shiller2015 și @Garber1990 oferă trei perspective diferite.

Codul Python al capitolului scrie explicit statisticile SADF și GSADF și calibrarea LPPL, fără un pachet special, astfel încît fiecare pas poate fi urmărit. Datele sînt închiderile zilnice ale indicilor S&P 500, Nasdaq 100, BET, Shanghai Composite și ale Bitcoin, de la EOD Historical Data (EODHD); seriile săptămînale sînt formate din închiderile zilnice acolo unde un test le cere. Notebook-ul cursului se deschide în Google Colab și rulează fără o instalare locală.

## @end

Întrebarea are deci două părți: recunoașterea unei bule cît timp crește și datarea sfîrșitului ei. Restul capitolului răspunde la prima parte cu teste de rădăcini explozive și la a doua cu modelul LPPL, apoi întreabă cît de bune sînt aceste răspunsuri cînd sînt judecate fără a ști ce a urmat.

# Bule în istorie

Secțiunea adună dovezile istorice: maniile clasice, crahurile din 1929 și 1987, definiția unei bule prin valoarea fundamentală și șase creșteri și crahuri din datele cursului, care servesc drept cazuri de test pentru restul capitolului.

## Bule speculative, 1636–2021 {h=5a344bbf}

Mania lalelelor din Țările de Jos, în 1636–1637, este cel mai vechi exemplu bine cunoscut. Contractele pe bulbi rari de lalele s-au tranzacționat la prețuri extreme și s-au prăbușit în februarie 1637. Pictura de pe slide, *Carul nebunilor al Florei* de Hendrik Gerritsz Pot, este o satiră realizată în jurul momentului prăbușirii. @Garber1990 susține că o parte a poveștii este un mit: bulbii rari erau scumpi din motive întemeiate, iar argumentul pentru o manie se sprijină mai ales pe prețurile bulbilor obișnuiți.

Episoadele ulterioare includ Compania Mării Sudului (*South Sea Company*) în 1720, Wall Street în 1929, Japonia în 1989, acțiunile dot-com în 2000, piața imobiliară în 2007 și criptomonedele în 2017 și 2021. @KindlebergerAliber2005 descriu un tipar comun: o poveste nouă care pare să justifice prețuri mai mari, credit ieftin, prețuri în creștere care atrag cumpărători noi, care înainte nu se interesau de acel activ, și în final o panică în care toată lumea încearcă să vîndă în același timp. Tiparul este o narațiune, nu un test, dar sugerează ce trebuie căutat în date: o accelerare a prețurilor și un sfîrșit care vine din interiorul pieței.

## Crahurile din 1929 și 1987 {h=26d6d835}

Crahul din octombrie 1929 a încheiat boom-ul anilor 1920 și a deschis Marea Depresiune. Crahul din 19 octombrie 1987, Lunea Neagră (*Black Monday*), este cea mai mare scădere într-o singură zi a indicelui Dow Jones și a venit fără vreo știre importantă în acea zi. Cele două fotografii arată mulțimea din fața Bursei din New York pe 29 octombrie 1929 și bursa din Amsterdam la două zile după Lunea Neagră.

Crahul din 1987 a pornit literatura acestui capitol. @SornetteJohansenBouchaud1996 au găsit în anii dinaintea lui 1987 o accelerare a prețului cu oscilații care deveneau mai rapide pe măsură ce crahul se apropia, tipar formalizat ulterior ca model LPPL. Datele zilnice ale cursului încep în 1990, așa că anul 1987 este cunoscut aici doar din literatură; bulele de după sînt studiate cu datele proprii ale cursului.

Un crah fără știri indică o cauză endogenă. Dacă în ziua scăderii nu a sosit nicio informație nouă, scăderea nu poate fi reacția la acea informație; piața însăși trebuie să fi devenit fragilă dinainte, astfel încît o perturbare mică a putut declanșa o vînzare colectivă mare. Această perspectivă, a crahului ca sfîrșit al unei acumulări treptate și nu ca răspuns la un șoc, este punctul de plecare al modelului LPPL.

## Valoarea fundamentală și bula {h=2e9c91e2}

Valoarea fundamentală a unui activ este valoarea actualizată a fluxurilor de numerar pe care se așteaptă să le plătească. Pentru o acțiune, cu dividendele $D_{t+k}$ plătite peste $k$ perioade și o rată de actualizare constantă $r$ pe perioadă,

$$ F_t = \sum_{k=1}^{\infty}\frac{E_t[D_{t+k}]}{(1+r)^k}, $$ {#eq:fund}

unde $E_t[\cdot]$ este media condiționată de informația disponibilă la momentul $t$. Componenta de bulă este partea prețului pe care această sumă nu o explică, $B_t = P_t - F_t$.

Dificultatea este că $F_t$ nu se observă. Depinde de așteptările privind dividende foarte îndepărtate și de rata de actualizare, iar schimbări mici ale oricăreia schimbă mult valoarea $F_t$. Un statistician nu poate deci măsura direct $B_t$; singura posibilitate este să deducă implicațiile pe care o bulă le are asupra traiectoriei prețului și să le testeze. Acesta este lucrul pe care îl fac testele de rădăcini explozive și modelul LPPL, fiecare cu propria implicație.

Două viziuni asupra piețelor financiare încadrează dezbaterea. În viziunea piețelor eficiente, prețurile reflectă informația disponibilă, iar „bulele” sînt vizibile doar retrospectiv [@Fama2014]: o creștere mare urmată de o scădere este compatibilă cu o revizuire rațională a așteptărilor. În viziunea finanțelor comportamentale, herding-ul și extrapolarea randamentelor trecute împing prețurile departe de valoare [@Shiller2015]. @GreenwoodShleiferYou2019 aduc o dovadă importantă pentru acest capitol: creșterile mari ale portofoliilor sectoriale măresc probabilitatea unui crah ulterior, chiar dacă majoritatea lor nu se încheie cu unul.

## Șase creșteri și crahuri în datele cursului {h=30b5add5}

Graficul are șase panouri, cîte unul pentru fiecare episod: S&P 500 și Nasdaq 100 în 2000, BET în 2007, Shanghai Composite în 2015 și Bitcoin în 2017 și 2021. Fiecare panou reprezintă închiderea zilnică pe scară logaritmică, de la șase luni înainte de minimul creșterii pînă la optsprezece luni după maximum. Zona hașurată marchează creșterea, de la minim la maximum, iar linia roșie punctată marchează maximul.

Pe scară logaritmică o rată de creștere constantă este o linie dreaptă, deci panta fiecărei curbe este rata ei de creștere. Creșterile diferă mult ca pantă și ca formă: Shanghai Composite este aproape plat timp de cîteva luni și apoi urcă abrupt, Bitcoin în 2017 urcă cu mai mult de un ordin de mărime, iar S&P 500 crește cu oscilații mari. Și scăderile de după maximum diferă, de la alunecarea lungă a indicelui Nasdaq 100 la prăbușirea indicelui BET în 2008.

## Interpretarea celor șase episoade {h=46adaf24}

Tabelul rezumă fiecare creștere prin trei numere. Creșterea este $P_{\text{max}}/P_{\text{min}} - 1$; creșterea pe an este $\ln(P_{\text{max}}/P_{\text{min}})$ împărțit la lungimea creșterii în ani, o rată compusă continuu; scăderea este cea mai mică închidere din anul de după maximum, raportată la maximum. S&P 500 a crescut cu 60% între 31 august 1998 și 24 martie 2000, o creștere de 30% pe an, și a scăzut cu 27% într-un an. Nasdaq 100 a crescut cu 317%, cu 97% pe an, și a scăzut cu 66%.

Celelalte episoade confirmă intervalul. BET a crescut cu 68% între 27 iunie 2006 și 24 iulie 2007, cu 48% pe an, și a pierdut 50% în anul următor. Shanghai Composite a crescut cu 159% între 20 martie 2014 și 12 iunie 2015, cu 77% pe an, și a pierdut 49%. Bitcoin a crescut cu 2476% în aproximativ un an, pînă la 16 decembrie 2017, o creștere de 312% pe an, apoi a pierdut 83%; în 2021 a crescut cu 527% în șapte luni, cu 309% pe an, și a scăzut cu 53%.

Episoadele nu sînt la fel de puternice. Bitcoin și-a înmulțit prețul de aproximativ 26 de ori într-un an, în timp ce S&P 500 a crescut cu 60% într-un an și jumătate, un ritm rapid, dar nu extraordinar pentru un indice larg. O metodă care recunoaște bulele nu ar trebui deci să dea același verdict pentru toate șase; felul în care le ordonează este unul dintre testele metodei.

## Recapitulare: bule în istorie {h=f4cf2635}

O bulă este partea prețului de peste valoarea fundamentală @eq:fund, care nu se observă, deci o bulă poate fi studiată doar prin implicațiile ei asupra prețurilor. Cele șase episoade din datele cursului au creșteri între 60% și 2476%, urmate de scăderi între 27% și 83% în decurs de un an.

Semnăturile care trebuie căutate sînt creșterea explozivă și superexponențială, oscilațiile care se accelerează spre sfîrșitul creșterii și apoi un sfîrșit. Următoarele două secțiuni transformă fiecare semnătură într-un model și într-un test.

## @end

Istoria oferă exemplele și sugerează semnăturile, dar nu spune care creștere este o bulă cît timp este în desfășurare. Secțiunea următoare deduce prima implicație testabilă, creșterea explozivă, din teoria bulelor raționale.

# Bule raționale și rădăcini explozive

Secțiunea arată că o bulă deținută de investitori raționali trebuie să crească exploziv, simulează o astfel de bulă și construiește testele de rădăcină unitară la dreapta, care caută episoadele explozive. Testele sînt apoi aplicate pe Nasdaq 100, Bitcoin, BET și Shanghai Composite.

## Creșterea explozivă a unei bule raționale {h=f546c0ae}

Într-o piață fără oportunități de arbitraj și cu investitori neutri la risc, prețul de azi este egal cu valoarea actualizată a mediei prețului de mîine plus dividendul de mîine:

$$ P_t = \frac{E_t[P_{t+1} + D_{t+1}]}{1+r}. $$ {#eq:noarb}

Rezolvarea acestei ecuații înainte, prin înlocuirea aceleiași relații pentru $P_{t+1}$, $P_{t+2}$ și așa mai departe, dă valoarea fundamentală @eq:fund, cu condiția ca prețul mediu actualizat $E_t[P_{t+k}]/(1+r)^k$ să tindă la zero. Valoarea fundamentală este o soluție, dar nu singura.

Orice preț $P_t = F_t + B_t$ satisface și el @eq:noarb, cu condiția ca bula să satisfacă $E_t[B_{t+1}] = (1+r)B_t$. Înlocuirea în @eq:noarb arată de ce: $F_t$ satisface singur ecuația, iar termenul suplimentar $B_t$ se simplifică în ambii membri exact atunci cînd media lui actualizată din perioada următoare este egală cu valoarea lui de azi. Prin iterare se obține $E_t[B_{t+k}] = (1+r)^k B_t$: bula trebuie să crească, în medie, cu rata $r$. Motivul economic este simplu: un investitor cumpără un activ peste valoarea lui doar dacă se așteaptă ca supraevaluarea să crească suficient de repede pentru a plăti randamentul cerut.

Deoarece $1 + r > 1$, componenta de bulă este un proces exploziv, în sensul din [Capitolul 3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf): o autoregresie cu un coeficient mai mare decît 1. De aici rezultă implicația testabilă. Dacă dividendele sînt integrate de ordinul unu, $I(1)$, și valoarea fundamentală este $I(1)$, iar prețul este $I(1)$ fără bulă, dar exploziv cu bulă. Un comportament exploziv al prețurilor, absent din dividende, este urma statistică a unei bule raționale.

## Exemplu rezolvat: bula Blanchard–Watson {h=5aaf68f8}

O bulă care crește la nesfîrșit nu este realistă. @BlanchardWatson1982 au propus o bulă care se poate sparge: în fiecare perioadă supraviețuiește cu probabilitatea $\pi$ și crește atunci pînă la $\frac{1+r}{\pi}B_t$; altfel se sparge, iar $B_{t+1}$ scade aproape de zero. Media din perioada următoare este

$$ E_t[B_{t+1}] = \pi\cdot\frac{1+r}{\pi}B_t + (1-\pi)\cdot 0 = (1+r)B_t, $$

deci condiția de pe slide-ul anterior este îndeplinită, iar bula care se poate sparge rămîne rațională.

Cu $r = 2\%$ și $\pi = 0{,}98$ pe perioadă, bula crește în fiecare perioadă în care supraviețuiește cu $1{,}02/0{,}98 - 1 = 4{,}08\%$, de două ori randamentul cerut. Creșterea suplimentară este compensația pentru riscul spargerii: un investitor care deține bula o pierde cu probabilitatea de 2% în fiecare perioadă și trebuie plătit pentru acest risc cît timp ea durează. Numărul de perioade pînă la spargere are o distribuție geometrică, deci durata medie de viață este $1/(1-\pi) = 50$ de perioade. Bula supraviețuiește mai mult de $k$ perioade cu probabilitatea $0{,}98^k$, care este egală cu o jumătate pentru $k = \ln 0{,}5/\ln 0{,}98 \approx 34$: jumătate dintre aceste bule se sparg în cel mult 34 de perioade.

Exemplul conține toată logica bulelor raționale. Bula este pe deplin rațională pentru fiecare deținător, dar crește mai repede decît rata dobînzii cît timp durează, iar mărimea ei la spargere crește cu vîrsta. Cu cît durează mai mult, cu atît scăderea de la sfîrșit este mai mare.

## O bulă rațională simulată și trei rădăcini AR(1) {h=942e1eec}

Panoul din stînga reprezintă 400 de perioade dintr-un preț simulat. Valoarea fundamentală, cu albastru închis, este un random walk care rămîne între aproximativ 20 și 25 pe tot eșantionul. Prețul, cu roșu, este valoarea fundamentală plus o bulă Blanchard–Watson cu $r = 2\%$ și $\pi = 0{,}98$. În cea mai mare parte a timpului cele două linii coincid; de două ori prețul se îndepărtează de valoarea fundamentală într-un ritm tot mai rapid și revine brusc atunci cînd bula se sparge.

În traiectoria prezentată, primul episod exploziv merge în jos, pînă la aproximativ $-250$, înainte să se încheie în jurul perioadei 170, iar al doilea merge în sus, pînă aproape de 100, înainte să se spargă în jurul perioadei 345. Un preț negativ nu poate apărea pentru un activ real, deoarece proprietarul poate oricînd să renunțe la el; pentru o acțiune au sens economic doar bulele cu semn pozitiv, așa că doar al doilea episod este imaginea unei bule reale.

Panoul din dreapta reprezintă trei procese autoregresive de ordinul unu, AR(1), $y_t = \rho y_{t-1} + \varepsilon_t$ pe 200 de perioade, generate cu aceleași șocuri: $\rho = 0{,}9$ cu verde, $\rho = 1$ cu albastru închis și $\rho = 1{,}02$ cu roșu. Primele două rămîn într-o bandă de aproximativ 20 de unități în jurul lui zero, în timp ce al treilea se desparte de ele după aproximativ 50 de perioade și crește apoi abrupt.

## Interpretarea simulării {h=6bde54e0}

Bula simulată crește, se sparge și repornește. În această traiectorie componenta de bulă ajunge la cel mult 71 și se sparge de 2 ori după ce atinge o mărime mare. Prima bulă a acestei traiectorii este negativă și duce prețul sub zero, ceea ce pentru un activ real este exclus: un preț nu poate fi negativ, deci o bulă rațională trebuie să fie nenegativă. Între episoadele explozive prețul urmează valoarea fundamentală, așa că seria prețurilor alternează între un comportament de random walk și unul exploziv. Un test care privește tot eșantionul deodată vede un amestec; un test care privește subeșantionul potrivit vede explozia.

Cele trei procese AR(1) sînt cele trei cazuri din [Capitolul 3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf). Cu $\rho = 0{,}9$ procesul revine la medie și încheie cele 200 de perioade la $-0{,}79$; cu $\rho = 1$ rătăcește fără un nivel fix și se încheie la 15,5; cu $\rho = 1{,}02$ explodează și se încheie la 228. Cazul exploziv se cuantifică ușor: un șoc este înmulțit cu $1{,}02^k$ după $k$ perioade, deci se dublează după $\ln 2/\ln 1{,}02 \approx 35$ de perioade și este înmulțit cu $1{,}02^{100} = 7{,}24$ după 100 de perioade. O rădăcină doar puțin mai mare decît 1 este invizibilă pe intervale scurte și dominantă pe intervale lungi.

Aceasta este ideea testelor care urmează. O bulă rațională face prețul logaritmic exploziv în timpul creșterii, cu o rădăcină $\rho > 1$. Detectarea unei astfel de rădăcini explozive (*explosive root*), în partea potrivită a eșantionului, este urma statistică a bulei.

## Teste de rădăcină unitară la dreapta {h=97cdac9f}

Testele folosesc regresia Dickey–Fuller din [Capitolul 3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf) pe o fereastră a prețului logaritmic $y_t = \ln P_t$:

$$ \Delta y_t = a + \delta\,y_{t-1} + \varepsilon_t, \qquad \delta = \rho - 1, $$ {#eq:adf}

unde $\Delta y_t = y_t - y_{t-1}$, $a$ este interceptul, $\varepsilon_t$ eroarea, iar $\rho$ coeficientul AR(1) al lui $y_t$. Statistica testului este statistica $t$ a lui $\hat\delta$, iar regresia poate fi augmentată cu diferențe întîrziate, ca în testul ADF.

Ipotezele sînt inversate față de [Capitolul 3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf). Acolo, după @DickeyFuller1979, ipoteza nulă $\delta = 0$ (rădăcină unitară, *unit root*) este testată față de $\delta < 0$ (staționaritate) și este respinsă pentru statistici $t$ mari și negative, în coada din stînga. Aici alternativa care interesează este $\delta > 0$, o rădăcină explozivă, iar ipoteza nulă este respinsă pentru statistici $t$ mari și pozitive, în coada din dreapta (*right-tailed test*). Regresia este aceeași; se schimbă doar direcția în care se caută dovezile.

Un singur test pe tot eșantionul are o problemă serioasă: o bulă ocupă doar o parte a eșantionului, iar crahul care o urmează arată ca o revenire puternică la medie. Creșterea explozivă împinge $\hat\delta$ în sus, crahul îl împinge în jos, iar media spune puțin. Remediul este calculul statisticii pe multe ferestre. Notația descrie ferestrele prin fracțiuni din eșantion: $r_1, r_2 \in [0, 1]$, statistica $\mathrm{ADF}_{r_1}^{r_2}$ folosește observațiile de la $r_1T$ la $r_2T$, iar $r_0$ este fracțiunea celei mai mici ferestre permise.

## SADF, GSADF și BSADF (1/2) {h=4dda9e81}

Toate statisticile din această familie iau un supremum, cea mai mare valoare a statisticii ADF pe o mulțime de ferestre. Logica este cea a unei căutări: o singură fereastră în care statistica este mare ajunge pentru a semnala un episod exploziv, oriunde s-ar afla acea fereastră. Prețul căutării este că supremumul este mai mare decît orice statistică individuală chiar sub ipoteza nulă, deci are nevoie de propriile valori critice.

Statistica SADF a lui @PhillipsWuYu2011 fixează începutul ferestrei la prima observație și mută sfîrșitul înainte:

$$ \mathrm{SADF} = \sup_{r_2 \in [r_0,\,1]}\mathrm{ADF}_0^{r_2}. $$

Funcționează bine cînd eșantionul conține o singură bulă. După o primă bulă și un crah, însă, fiecare fereastră ulterioară conține încă acel crah, care trage statistica în jos, așa că SADF își pierde puterea față de o a doua bulă.

Statistica GSADF a lui @PhillipsShiYu2015a mută atît începutul, cît și sfîrșitul ferestrei:

$$ \mathrm{GSADF} = \sup_{r_2 \in [r_0,\,1]}\ \sup_{r_1 \in [0,\,r_2 - r_0]}\mathrm{ADF}_{r_1}^{r_2}. $$ {#eq:gsadf}

Deoarece începutul poate trece dincolo de un crah anterior, GSADF poate detecta mai multe bule în același eșantion. Cea mai mică fereastră recomandată de autori este $r_0 = 0{,}01 + 1{,}8/\sqrt{T}$ din eșantion, o fracțiune care scade cînd eșantionul crește.

## SADF, GSADF și BSADF (2/2) {h=7db68c80}

GSADF răspunde dacă eșantionul conține un episod exploziv; statistica BSADF (*backward supremum augmented Dickey–Fuller*) răspunde cînd. Pentru fiecare dată $r_2$, ea ține sfîrșitul ferestrei fixat la acea dată și caută după toate începuturile:

$$ \mathrm{BSADF}(r_2) = \sup_{r_1 \in [0,\,r_2 - r_0]}\mathrm{ADF}_{r_1}^{r_2}. $$

Statistica de la data $r_2$ folosește doar date pînă la $r_2$, deci poate fi calculată în timp real. GSADF este pur și simplu cel mai mare BSADF pe toate datele. După @PhillipsShiYu2015b, un episod începe cînd BSADF depășește valoarea critică și trebuie să dureze cel puțin $\log T$ observații, unde $\log$ este logaritmul natural; durata minimă împiedică raportarea depășirilor izolate, datorate zgomotului, drept bule.

Distribuțiile acestor statistici nu sînt standard și se obțin prin simulare Monte Carlo. Datele sînt simulate sub ipoteza nulă ca un random walk cu un drift slab, $y_t = T^{-1} + y_{t-1} + \varepsilon_t$; driftul $T^{-1}$ este atît de mic încît procesul are o rădăcină unitară, nu una explozivă. Fiecare serie simulată trece prin același calcul ca datele, iar cuantilele statisticilor simulate sînt valorile critice. Pentru BSADF se obține astfel o valoare critică pentru fiecare dată, deoarece numărul de ferestre crește cu $r_2$.

Citirea este aceeași pentru toate trei statisticile. O valoare peste valoarea critică de 95% respinge rădăcina unitară la pragul de 5% în favoarea unei rădăcini explozive; o valoare sub ea înseamnă că un eventual comportament exploziv nu este separat de un random walk de acest test.

## Exemplu rezolvat: o fereastră din Nasdaq 100 {h=f085b098}

Datele sînt închiderile săptămînale logaritmice ale indicelui Nasdaq 100 din ianuarie 1990 pînă în decembrie 2004, $T = 782$ de săptămîni. Cea mai mică fereastră este $r_0 = 0{,}01 + 1{,}8/\sqrt{782} = 0{,}074$ din eșantion, adică 58 de săptămîni. Exemplul calculează BSADF în săptămîna maximului, 24 martie 2000: dintre toate ferestrele care se încheie în acea săptămînă, cea mai mare statistică provine din fereastra care începe la 17 ianuarie 1992.

Pe această fereastră, cu $n = 427$ de variații săptămînale, OLS dă

$$ \widehat{\Delta y_t} = -0{,}0336 + 0{,}0060\,y_{t-1}. $$

Statistica $t$ a pantei este $t = \hat\delta/\mathrm{SE}(\hat\delta) = 0{,}006030/0{,}002072 = 2{,}91$, unde SE este eroarea standard (*standard error*) (cu valorile rotunjite 0,0060/0,0021 raportul ar fi 2,86). Aceasta este cea mai mare statistică dintre toate începuturile, deci BSADF la maximum este 2,91, mult peste valoarea critică de 95%, egală cu 0,54.

Semnul este cel care contează. Un $\hat\delta$ pozitiv înseamnă $\hat\rho = 1 + \hat\delta > 1$: cu cît prețul logaritmic de la sfîrșitul săptămînii trecute este mai mare, cu atît creșterea așteptată în această săptămînă este mai mare. Într-un random walk variația săptămînală nu depinde de nivel; într-o serie staționară un nivel ridicat anunță o scădere. Aici un nivel ridicat anunță o nouă creștere, semnul unei faze explozive.

## Episoadele explozive ale Nasdaq 100, 1990–2004 {h=b901e4a0}

Panoul de sus reprezintă închiderea săptămînală a indicelui Nasdaq 100 pe scară logaritmică, de la aproximativ 200 în 1990 pînă aproape de 5000 la maximul din martie 2000 și înapoi la aproximativ 1000 în 2002. Panoul de jos reprezintă statistica BSADF, cu albastru închis, și valoarea ei critică de 95%, linia roșie întreruptă, obținută din 1000 de random walk-uri simulate. Valoarea critică rămîne între aproximativ 0,1 și 0,8 pe toată perioada. Benzile hașurate, repetate în ambele panouri, sînt episoadele datate, perioadele în care BSADF rămîne peste valoarea critică cel puțin 7 săptămîni.

BSADF începe doar după primele 58 de săptămîni, cea mai mică fereastră. Rămîne sub zero în cea mai mare parte a primei jumătăți a anilor 1990, sare peste valoarea critică la mijlocul lui 1995 și apoi o depășește de mai multe ori pînă la episodul mare din 1998–2000, în care atinge maximul de aproximativ 2,9 în martie 2000. După maximum scade din nou și, de la sfîrșitul lui 2000, rămîne în general sub valoarea critică.

## Interpretarea testelor pentru Nasdaq 100 {h=1bae1834}

Statistica ADF pe tot eșantionul este $-1{,}27$, față de o valoare critică de 95% de $-0{,}13$ pentru testul la dreapta: nicio dovadă de comportament exploziv. Statistica SADF este 2,72, cu valoarea critică 1,48, iar statistica GSADF este 2,92, cu valoarea critică 2,28: ambele resping rădăcina unitară în favoarea unui comportament exploziv. Contrastul este problema descrisă mai sus. Crahul din 2000–2002 face parte din eșantionul complet și ascunde bula de un singur test; căutarea pe ferestre o găsește.

BSADF datează 6 episoade. Cel mai lung durează de la 23 octombrie 1998 pînă la 3 noiembrie 2000, 107 săptămîni, și acoperă creșterea principală. Episoade explozive mai scurte apar începînd din iunie 1995, așa că boom-ul anilor 1990 a venit în valuri, cum au constatat și @PhillipsWuYu2011.

Data de sfîrșit a episodului lung este tîrzie: vine la peste șapte luni după maximul din 27 martie 2000. BSADF compară fereastra curentă cu un random walk, iar o fereastră care conține o creștere lungă și o scădere scurtă arată încă explozivă o vreme. Testul datează exuberanța; nu anunță crahul, iar data lui de sfîrșit spune mai mult despre viteza statisticii decît despre piață.

## Bitcoin, BET și Shanghai: episoade datate {h=bb27ed0c}

Graficul aplică aceeași procedură pe trei piețe, cîte una pe coloană: Bitcoin din 2014 pînă în 2023, BET din 2000 pînă în 2012 și Shanghai Composite din 2010 pînă în 2018, toate cu închideri săptămînale logaritmice. Rîndul de sus reprezintă prețul pe scară logaritmică, cu episoadele datate hașurate; rîndul de jos reprezintă BSADF cu albastru închis și valoarea lui critică de 95% cu o linie roșie întreruptă.

Benzile hașurate sînt puține și clare. Bitcoin are două, una în 2017 și la începutul lui 2018 și una mai scurtă la începutul lui 2021. BET are trei: două în 2002–2006, în timpul creșterii lungi, și una în 2008–2009, în timpul scăderii. Shanghai Composite are una, în timpul creșterii abrupte din 2014–2015. În fiecare panou BSADF rămîne aproape de zero sau sub zero în cea mai mare parte a eșantionului și urcă abrupt în perioadele hașurate.

## Interpretarea celor trei piețe {h=13feb444}

Statistica GSADF este 3,85 pentru Bitcoin, 3,53 pentru BET și 3,46 pentru Shanghai Composite, toate peste valorile critice de 95%, care se află între 2,15 și 2,29. Toate trei piețele au deci episoade explozive. Pentru Bitcoin episoadele datate sînt aprilie 2017 – mai 2018 și ianuarie – mai 2021; pentru Shanghai Composite, noiembrie 2014 – iunie 2015.

BET este mai instructiv. Testul datează un episod timpuriu, aprilie 2002 – iulie 2003, apoi faza explozivă principală, decembrie 2003 – mai 2006, nu creșterea pînă în iulie 2007 care apare în tabelul celor șase episoade: acea creștere a fost puternică, 68% într-un an, dar nu explozivă în sensul testului. Semnalul BET din septembrie 2008 – aprilie 2009 marchează prăbușirea, nu o bulă. O scădere accelerată dă și ea $\hat\delta > 0$: cînd prețul scade cu atît mai repede cu cît este mai mic, nivelul este legat de variația următoare printr-un coeficient pozitiv. Testul detectează caracterul exploziv în ambele direcții, așa că datele trebuie citite întotdeauna alături de graficul prețului.

Episodul Bitcoin din 2017 durează pînă în mai 2018, la cinci luni după maximul din 16 decembrie 2017, tot pentru că ferestrele care includ creșterea țin statistica sus o vreme după maximum. Tiparul indicelui Nasdaq 100 se repetă: testele datează bine creșterea și tîrziu sfîrșitul ei.

## Limitele testelor de rădăcini explozive {h=33cc4f93}

Prima limită privește fundamentele. Un preț exploziv este o bulă doar dacă fundamentele nu sînt ele însele explozive; o perioadă în care așteptările privind dividendele cresc exploziv ar produce și ea un preț exploziv. Testul mai bun se face pe raportul dintre preț și dividend, care elimină partea fundamentală; @PhillipsShiYu2015a fac acest lucru pentru S&P 500. Bitcoin nu are dividende deloc, așa că testul pe preț este singura posibilitate, iar verdictul lui privește caracterul exploziv, nu supraevaluarea.

A doua limită privește volatilitatea. Valorile critice provin dintr-un random walk cu varianță constantă, în timp ce randamentele financiare au volatilitate variabilă, iar volatilitatea variabilă distorsionează mărimea testului: probabilitatea reală a unei respingeri false nu mai este 5%. Un wild bootstrap, care reeșantionează reziduurile cu semne aleatoare păstrînd mărimea lor la fiecare dată, dă valori critice robuste la această problemă [@PhillipsShi2020].

A treia limită privește momentul. BSADF folosește doar date trecute, deci poate rula în timp real și poate semnala o bulă în desfășurare. Nu spune însă nimic despre momentul în care bula se va încheia; statistica rămîne ridicată după maximum și scade doar după ce crahul face parte din date. Modelul LPPL din secțiunea următoare încearcă să răspundă exact la această întrebare.

## Recapitulare: bule raționale și rădăcini explozive {h=0c94316c}

O bulă rațională crește în medie cu rata $r$, $E_t[B_{t+1}] = (1+r)B_t$, deci este un proces exploziv. Testul ADF la dreapta @eq:adf ia $\delta > 0$ ca alternativă; SADF și GSADF @eq:gsadf iau supremumul pe ferestre, iar BSADF datează episoadele folosind datele pînă la fiecare dată.

Nasdaq 100, Bitcoin, BET și Shanghai Composite au toate episoade explozive. Testele datează creșterea, adesea cu un sfîrșit plasat la cîteva luni după maximum, și nu datează crahul.

## @end

Testele de rădăcini explozive răspund la prima jumătate a întrebării capitolului: recunosc o bulă cît timp crește, cu datele disponibile la acel moment. Nu spun cînd se încheie. Pentru aceasta este nevoie de un model al formei creșterii, care să conțină data de sfîrșit ca parametru; acesta este modelul LPPL.

# Modelul LPPL

Secțiunea construiește pas cu pas modelul legii de putere log-periodice: ideea crahului ca punct critic, creșterea superexponențială, legătura dintre rata de hazard a crahului și creșterea prețului, oscilațiile log-periodice și condițiile pe care trebuie să le îndeplinească un model ajustat pentru a descrie o bulă credibilă.

## Crahurile ca puncte critice {h=e49ac83a}

Didier Sornette, fizician la ETH Zürich, a propus împreună cu Anders Johansen și Olivier Ledoit modelul cunoscut după inițialele lor, modelul Johansen–Ledoit–Sornette (JLS) [@JohansenLedoitSornette2000]. Punctul lui de plecare este comportamentul traderilor: fiecare trader îi imită parțial pe cei din jurul lui. Imitația (*imitation*) creează un feedback pozitiv. Prețurile în creștere atrag cumpărători, cumpărătorii noi împing prețurile mai sus, iar prețurile mai mari atrag alți cumpărători, așa că creșterea se accelerează în loc să aibă un ritm constant.

Fizica oferă o analogie. În apropierea unui punct critic (*critical point*), de exemplu apa aproape de fierbere sau un magnet aproape de temperatura la care își pierde magnetizarea, interacțiunile locale dintre părțile unui sistem devin coordonate pe distanțe mari, iar sistemul devine extrem de sensibil la perturbații mici. În modelul JLS piața se apropie de o astfel de stare pe măsură ce imitația se extinde: tot mai mulți traderi au aceeași poziție, iar o știre mică poate declanșa o vînzare colectivă.

Timpul critic (*critical time*) $t_c$ este momentul cel mai probabil al sfîrșitului bulei. Nu este data sigură a unui crah. În model un crah devine tot mai probabil pe măsură ce $t_c$ se apropie, dar este posibil să nu aibă loc deloc: bula se poate încheia și cu o scădere lentă sau poate trece într-un alt regim. Distincția contează pentru evaluarea de la sfîrșitul capitolului, deoarece o estimație a lui $t_c$ este o estimație a sfîrșitului unui regim, nu o prognoză că la acea dată va avea loc un crah.

## Creșterea superexponențială {h=4f95ef05}

Creșterea exponențială are o rată de creștere constantă. Pe scară logaritmică este o linie dreaptă,

$$ \ln P(t) = a + g\,t, \qquad \frac{d\ln P}{dt} = g, $$

unde $a$ este prețul logaritmic inițial, iar $g$ rata constantă de creștere. Pe perioade lungi, indicii bursieri cresc de cele mai multe ori aproximativ în acest fel, cu zgomot în jurul dreptei.

Creșterea superexponențială (*super-exponential growth*) are o rată de creștere care crește ea însăși în timp, așa că pe scară logaritmică curba se îndoaie în sus. Cea mai simplă formă cu această proprietate este singularitatea de tip lege de putere (*power-law singularity*)

$$ \ln P(t) = A + B\,(t_c - t)^m, \qquad B < 0,\ 0 < m < 1, $$ {#eq:pls}

unde $t_c - t$ este timpul rămas pînă la timpul critic, $A$ prețul logaritmic atins la $t_c$, $B$ mărimea creșterii, iar $m$ exponentul. Deoarece $B < 0$, iar $(t_c - t)^m$ scade cînd $t$ se apropie de $t_c$, prețul logaritmic crește spre $A$.

Rata de creștere este derivata lui @eq:pls:

$$ \frac{d\ln P}{dt} = -Bm\,(t_c - t)^{m-1}. $$

Cu $m < 1$ exponentul $m - 1$ este negativ, deci rata de creștere tinde la infinit cînd $t \to t_c$, în timp ce prețul logaritmic rămîne finit, $\ln P(t_c) = A$. O mărime cu o rată de variație infinită la o dată finită se numește singularitate în timp finit (*finite-time singularity*). O astfel de creștere nu poate continua după $t_c$; trebuie să se încheie la timpul critic sau înaintea lui.

## Creștere exponențială și creștere superexponențială {h=a86f3e71}

Graficul compară trei traiectorii pe intervalul de la 0 pînă la timpul critic $t_c = 1$. Panoul din stînga reprezintă prețul logaritmic: creșterea exponențială cu $g = 0{,}8$ cu albastru închis, singularitatea de tip lege de putere @eq:pls cu $A = 1$, $B = -1$ și $m = 0{,}5$ cu roșu, și aceeași singularitate cu oscilații log-periodice adăugate, cu verde. Panoul din dreapta reprezintă rata de creștere $d\ln P/dt$ a fiecărei traiectorii, cu timpul critic marcat printr-o linie verticală punctată.

În stînga cele trei curbe pornesc de la zero și se încheie la niveluri apropiate, între aproximativ 0,8 și 0,9. Traiectoria exponențială este o linie dreaptă; cea roșie se află sub ea aproape tot timpul și o intersectează doar spre sfîrșit, cînd se îndoaie abrupt în sus; cea verde urmează traiectoria roșie printr-o succesiune de paliere și urcări. În dreapta rata exponențială este o linie orizontală, rata roșie urcă încet și apoi explodează aproape de $t_c$, iar rata verde oscilează în jurul celei roșii, cu valuri tot mai înalte și mai scurte pe măsură ce $t_c$ se apropie.

## Interpretarea traiectoriilor de creștere {h=c3d2142a}

La început cele trei traiectorii au rate de creștere apropiate: 0,8 pentru traiectoria exponențială și 0,50 pentru cea singulară. La începutul unei bule traiectoriile sînt deci greu de deosebit, iar doar un eșantion lung sau unul apropiat de sfîrșit arată diferența. Rata exponențială rămîne 0,8 tot timpul. Rata singulară urcă de la 0,50 la început la 0,70 la jumătatea intervalului, pentru $t = 0{,}5$, și la 3,98 chiar înainte de $t_c$: cea mai mare parte a accelerării este concentrată în ultima parte a intervalului.

Oscilațiile se suprapun peste trendul accelerat. Fiecare oscilație este o perioadă de creștere rapidă urmată de o perioadă de creștere lentă, o corecție în limbajul piețelor, iar oscilațiile devin mai rapide aproape de $t_c$. Slide-urile următoare deduc ambele trăsături, trendul accelerat și oscilațiile accelerate, din rata de hazard a crahului.

## De la riscul de crah la ecuația LPPL (1/2) {h=8f661587}

Modelul JLS scrie variația relativă a prețului înaintea crahului ca sumă a trei părți:

$$ \frac{dP}{P} = \mu(t)\,dt + \sigma\,dW - \kappa\,dj. $$ {#eq:jls}

Prima parte este creșterea așteptată pe un interval scurt $dt$, cu un drift $\mu(t)$ care se poate schimba în timp. A doua este zgomotul obișnuit: $W$ este o mișcare browniană (*Brownian motion*), versiunea în timp continuu a unui random walk, iar $\sigma$ volatilitatea. A treia este crahul: $j$ este un proces cu salt, egal cu 0 înaintea crahului, care sare la 1 în momentul crahului, iar saltul elimină o fracțiune $\kappa$ din preț.

Probabilitatea crahului este descrisă de rata de hazard (*hazard rate*) $h(t)$: probabilitatea pe unitatea de timp ca crahul să aibă loc acum, știind că nu a avut loc pînă acum. Pe un interval scurt, probabilitatea unui crah între $t$ și $t + dt$ este aproximativ $h(t)\,dt$. O rată de hazard care crește în timp înseamnă că fiecare zi fără crah este urmată de o zi în care crahul este mai probabil.

## De la riscul de crah la ecuația LPPL (2/2) {h=b403c829}

În absența arbitrajului și cu traderi neutri la risc, variația așteptată a prețului înaintea crahului trebuie să fie zero: $E[dP] = 0$. Termenul de zgomot are media zero, iar $E[dj] = h(t)\,dt$, așa că aplicarea mediei în @eq:jls dă $\mu(t)\,dt - \kappa\,h(t)\,dt = 0$, adică

$$ \mu(t) = \kappa\,h(t). $$

Driftul trebuie să plătească exact pierderea așteptată din crah. Cu cît hazardul este mai mare, cu atît prețul trebuie să crească mai repede cît timp crahul nu a avut loc: riscul și creșterea cresc împreună. Aceasta inversează intuiția obișnuită, în care un preț în creștere semnalează un risc în scădere.

Imitația aproape de un punct critic dă o rată de hazard care crește spre $t_c$, cu oscilații:

$$ h(t) \approx \alpha\,(t_c - t)^{m-1}\bigl[1 + \beta\cos\bigl(\omega\ln(t_c - t) - \phi'\bigr)\bigr], $$ {#eq:hazard}

unde $\alpha > 0$ este nivelul hazardului, $\beta$ mărimea relativă a oscilațiilor, $\omega$ log-frecvența lor, iar $\phi'$ faza lor. Deoarece $m < 1$, factorul $(t_c - t)^{m-1}$ crește nelimitat cînd $t \to t_c$. Înaintea crahului, $d\ln P \approx \mu(t)\,dt = \kappa\,h(t)\,dt$, făcînd abstracție de zgomot, deci prețul logaritmic este integrala lui $\kappa h(t)$. Pentru partea de lege de putere, integrala lui $\kappa\alpha(t_c - t)^{m-1}$ este $-\frac{\kappa\alpha}{m}(t_c - t)^m$ plus o constantă, adică termenul $B(t_c - t)^m$ din @eq:pls, cu $B = -\kappa\alpha/m < 0$. Partea oscilantă se integrează la fel și dă $(t_c - t)^m$ înmulțit cu un cosinus cu o altă fază. Împreună, ele dau ecuația de pe slide-ul următor.

## Ecuația LPPL {h=337419ca}

Pentru $t < t_c$, legea de putere log-periodică este

$$ \ln P(t) = A + B\,(t_c - t)^m + C\,(t_c - t)^m\cos\bigl(\omega\ln(t_c - t) - \phi\bigr). $$ {#eq:lppl}

Primii doi termeni sînt singularitatea de tip lege de putere @eq:pls; al treilea adaugă oscilații a căror amplitudine scade cu $(t_c - t)^m$ și al căror argument depinde de logaritmul timpului rămas pînă la $t_c$. Ecuația descrie prețul logaritmic în timpul bulei, pînă la sfîrșitul regimului; nu spune nimic despre mărimea crahului sau despre traiectoria de după $t_c$.

Tabelul de pe slide prezintă cei șapte parametri. Timpul critic $t_c$ este sfîrșitul cel mai probabil al bulei și trebuie să se afle după ultima observație folosită în ajustare. $A > 0$ este prețul logaritmic care ar fi atins la $t_c$ dacă bula ar dura pînă atunci. $B < 0$ dă mărimea creșterii de tip lege de putere, iar $0 < m < 1$ cît de puternic se accelerează ea. $C$ este mărimea oscilațiilor, mai mică în valoare absolută decît $|B|$, $\omega$ log-frecvența lor unghiulară, de obicei între 2 și 25, iar $\phi$ faza lor, în $[0, 2\pi)$.

Trei parametri, $t_c$, $m$ și $\omega$, intră neliniar în ecuație, și tocmai acest lucru face estimarea dificilă. Ceilalți patru intră liniar odată ce parametrii neliniari sînt fixați, după o rescriere în care faza $\phi$ dispare; aceasta este baza metodei de estimare din secțiunea următoare.

## Oscilațiile log-periodice {h=5f9cf022}

Termenul oscilant $\cos(\omega\ln(t_c - t) - \phi)$ este periodic în $\ln(t_c - t)$, nu în $t$. El parcurge un ciclu complet de fiecare dată cînd $\ln(t_c - t)$ se schimbă cu $2\pi/\omega$. Maximele $t_n$ ale cosinusului satisfac $\omega\ln(t_c - t_n) - \phi = 2\pi k$ pentru numere întregi $k$ succesive, deci logaritmii timpilor rămași pînă la $t_c$ la maxime succesive diferă cu $2\pi/\omega$. Prin exponențiere,

$$ \frac{t_c - t_n}{t_c - t_{n+1}} = \lambda = e^{2\pi/\omega}. $$ {#eq:lambda}

Timpul rămas pînă la $t_c$ scade cu același factor $\lambda$ la fiecare ciclu, așa că ciclurile devin tot mai scurte, într-o progresie geometrică.

Această proprietate se numește invarianță discretă la scară (*discrete scale invariance*): tiparul arată la fel cînd timpul pînă la $t_c$ este rescalat cu factorul $\lambda$, dar nu cu un factor oarecare. O interpretare naturală este o ierarhie a pieței, formată din traderi individuali, grupuri de traderi și instituții, în care imitația se propagă nivel cu nivel, fiecare nivel cu propria scară de timp.

Exemplul rezolvat face factorul concret. Cu $\omega = 8$, $\lambda = e^{2\pi/8} = 2{,}19$. Dacă o corecție are loc cu 200 de zile înainte de $t_c$, următoarea vine cu aproximativ $200/2{,}19 \approx 91$ de zile înainte de $t_c$, apoi cu aproximativ 42 și apoi cu aproximativ 19 zile înainte de $t_c$. Cu $\omega = 6{,}28$, $\lambda = 2{,}72$, iar cu $\omega = 10$, $\lambda = 1{,}87$. Literatura empirică raportează valori ale lui $\lambda$ apropiate de 2 [@Sornette2017], ceea ce corespunde unui $\omega$ apropiat de 9.

## Componentele ecuației LPPL {h=2b38bb5c}

Graficul descompune ecuația pe intervalul de la 0 pînă la $t_c = 1$. Panoul din stînga reprezintă partea de lege de putere $A + B(t_c - t)^m$ cu $A = 1$ și $B = -1$ pentru trei exponenți: $m = 0{,}3$ cu albastru închis, $m = 0{,}5$ cu roșu și $m = 0{,}8$ cu verde. Toate trei pornesc de la zero și urcă spre 1 la $t_c$; curba cu $m = 0{,}8$ este aproape dreaptă, în timp ce curba cu $m = 0{,}3$ rămîne jos multă vreme și urcă abrupt aproape de $t_c$.

Panoul din mijloc reprezintă oscilația $(t_c - t)^m\cos(\omega\ln(t_c - t))$ cu $m = 0{,}5$ pentru $\omega = 6$, cu violet, și $\omega = 10$, cu portocaliu. Ambele pornesc de la 1 și oscilează între valori pozitive și negative; oscilațiile se micșorează și devin mai rapide spre $t_c$, iar curba portocalie, cu $\omega$ mai mare, parcurge mai multe cicluri. Panoul din dreapta reprezintă traiectoria LPPL completă cu $m = 0{,}5$, $\omega = 8$ și $C/|B| = 0{,}08$: o curbă crescătoare cu paliere tot mai scurte și mai apropiate aproape de $t_c$. Aceste valori ilustrative nu îndeplinesc condiția de amortizare $m|B|/(\omega|C|) \ge 1$, folosită mai jos pentru a califica o ajustare: $0{,}5/(8 \cdot 0{,}08) = 0{,}78 < 1$, deci o ajustare cu acești parametri nu ar fi calificată.

## Interpretarea componentelor {h=a942ee5b}

Exponentul $m$ controlează unde se concentrează creșterea. Un $m$ mai mic o concentrează aproape de $t_c$: curba este plată multă vreme și apoi urcă brusc. La extreme, $m$ aproape de 1 dă o linie aproape dreaptă, un trend cu o rată de creștere aproape constantă, care cu greu poate fi numit bulă, iar $m$ aproape de 0 dă o curbă plată aproape peste tot, cu un salt la $t_c$. Estimațiile lui $m$ apropiate de oricare capăt al intervalului $(0, 1)$ nu descriu deci o bulă credibilă.

Termenul oscilant se micșorează cu $(t_c - t)^m$ și se accelerează, cu cicluri care se scurtează cu factorul $\lambda$ din @eq:lambda. Un $\omega$ mai mare dă un $\lambda$ mai mic și deci mai multe cicluri înainte de $t_c$. Adunate la trendul de tip lege de putere, oscilațiile apar ca niște corecții tot mai dese, care devin în același timp mai mici în mărime absolută.

Creșterile reale, ajustate în secțiunea următoare, arată acest amestec: un trend accelerat cu corecții tot mai dese. Dacă oscilațiile sînt o trăsătură reală a datelor sau un termen flexibil care îmbunătățește ajustarea este una dintre întrebările dezbătute în literatură.

## Exemplu rezolvat: o valoare LPPL calculată de mînă {h=c9eb80e3}

Parametrii sînt $A = 8$, $B = -1{,}2$, $C = 0{,}08$, $m = 0{,}5$, $\omega = 8$, $\phi = 0$ și $t_c = 2$, cu timpul măsurat în ani. La $t = 1$ timpul rămas este $t_c - t = 1$, deci $(t_c - t)^m = 1$ și $\ln 1 = 0$, ceea ce dă $\cos 0 = 1$. Prețul logaritmic este $\ln P = 8 - 1{,}2 + 0{,}08 = 6{,}880$, iar prețul este $P = e^{6{,}880} = 973$.

La $t = 1{,}75$ timpul rămas este 0,25, deci $(0{,}25)^{0{,}5} = 0{,}500$ și $\ln 0{,}25 = -1{,}386$. Cosinusul este $\cos(8\cdot(-1{,}386)) = 0{,}095$, iar

$$ \ln P = 8 - 1{,}2\cdot 0{,}500 + 0{,}08\cdot 0{,}500\cdot 0{,}095 = 7{,}404, $$

deci $P = 1642$. La $t = 1{,}99$ timpul rămas este 0,01, deci $(0{,}01)^{0{,}5} = 0{,}100$, iar cu $\ln 0{,}01 = -4{,}605$ cosinusul este $\cos(8\cdot(-4{,}605)) = 0{,}654$. Prețul logaritmic este $8 - 0{,}120 + 0{,}005 = 7{,}885$, iar $P = 2658$.

Prețul crește de la 973 la 2658 cînd $t$ trece de la 1 la 1,99 și ar ajunge la $e^A = e^8 = 2981$ la $t_c$. Cea mai mare parte a creșterii vine în ultimele luni: de la $t = 1{,}75$ la $t = 1{,}99$, în mai puțin de trei luni, prețul crește cu peste 60%. Termenul oscilant este mic în acest exemplu, 0,08 în prețul logaritmic la $t = 1$ și cîteva miimi la celelalte două date, dar mută momentele fazelor de creștere mai rapidă și mai lentă.

## Restricțiile parametrilor: filtrul {h=b8863204}

O ajustare a ecuației @eq:lppl se numește calificată (*qualified*) doar dacă parametrii ei descriu o bulă credibilă. Condițiile din acest capitol urmează @ShuZhu2020, care pornesc de la @SornetteEtAl2015. Primul grup privește parametrii. $B < 0$ înseamnă că prețul crește. $0{,}01 \le m \le 0{,}99$ înseamnă că creșterea se accelerează, iar prețul la $t_c$ rămîne finit. $2 \le \omega \le 25$ exclude oscilațiile prea lente pentru a fi observate sau atît de rapide încît ajustează zgomot. Numărul de oscilații din fereastra $[t_1, t_2]$ trebuie să fie suficient de mare:

$$ \frac{\omega}{\pi}\ln\frac{t_c - t_1}{t_c - t_2} \ge 2{,}5. $$

Membrul stîng numără semiciclurile, maximele și minimele, cosinusului între $t_1$ și $t_2$, deoarece un ciclu complet cere o variație de $2\pi/\omega$ a lui $\ln(t_c - t)$.

Alte două condiții privesc timpul critic și hazardul. Timpul critic trebuie să fie apropiat, în raport cu lungimea ferestrei: $t_2 \le t_c \le t_2 + 0{,}2\,(t_2 - t_1)$. Condiția de amortizare (*damping*)

$$ \frac{m|B|}{\omega|C|} \ge 1 $$

menține rata de hazard pozitivă. Rata de creștere a ecuației @eq:lppl este un termen de tip lege de putere proporțional cu $m|B|$ plus un termen oscilant de mărime aproximativ $\omega|C|$ atunci cînd $\omega$ este mult mai mare decît $m$; dacă termenul oscilant ar fi mai mare, rata de creștere, și odată cu ea hazardul $h(t) = \mu(t)/\kappa$, ar deveni negative la unele date, ceea ce modelul nu permite.

Al doilea grup privește reziduurile. Fiecare preț ajustat trebuie să se afle la cel mult 15% de prețul observat. Reziduurile, privite ca funcție de $\ln(t_c - t)$, trebuie să arate ciclul log-periodic conform periodogramei Lomb [@Lomb1976], o periodogramă pentru puncte neegal distanțate, la pragul de 10%. În sfîrșit, reziduurile trebuie să fie staționare: testele Dickey–Fuller și Phillips–Perron din [Capitolul 3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf) trebuie să respingă rădăcina unitară la pragul de 10%.

## Recapitulare: modelul LPPL {h=8975ab2a}

Imitația creează o rată de hazard @eq:hazard care crește spre $t_c$; condiția de lipsă a arbitrajului $\mu(t) = \kappa h(t)$ o transformă într-o creștere superexponențială a prețului. Prin integrare se obține ecuația LPPL @eq:lppl, cu șapte parametri.

Oscilațiile sînt log-periodice: timpul rămas pînă la $t_c$ scade cu factorul $\lambda = e^{2\pi/\omega}$ de la un ciclu la următorul. O ajustare contează ca semnal de bulă doar dacă trece de filtrul format din condițiile asupra parametrilor și asupra reziduurilor.

## @end

Modelul LPPL transformă semnăturile vagi din prima secțiune într-o ecuație cu o dată de sfîrșit. Pentru a fi util trebuie estimat din date zgomotoase, iar estimarea se dovedește veriga cea mai slabă: secțiunea următoare arată cum se face și cît de bine este identificat timpul critic.

# Estimarea

Secțiunea prezintă metoda în doi pași a lui Filimonov și Sornette, care reduce estimarea la o căutare după trei parametri, analizează forma funcției de cost care se minimizează și ajustează modelul pe cele șase episoade din datele cursului, cu date care se opresc cu 30 de zile înainte de maximum.

## Metoda în doi pași a lui Filimonov și Sornette (1/2) {h=1535e4c8}

Parametrii se estimează prin metoda celor mai mici pătrate: ei minimizează suma pătratelor reziduurilor (*sum of squared residuals*, SSR) pe fereastra $[t_1, t_2]$,

$$ \mathrm{SSR} = \sum_{t=t_1}^{t_2}\bigl[\ln P_t - \mathrm{LPPL}(t)\bigr]^2, $$

unde $\mathrm{LPPL}(t)$ este membrul drept al ecuației @eq:lppl. O căutare directă după șapte parametri nu este de încredere: funcția de cost are multe minime locale (*local minima*), iar un algoritm numeric de optimizare se oprește într-unul dintre ele, în funcție de punctul de pornire. @FilimonovSornette2013 reduc căutarea la trei parametri.

Ideea este dezvoltarea cosinusului. Cu $\tau = t_c - t$,

$$ C\cos(\omega\ln\tau - \phi) = C_1\cos(\omega\ln\tau) + C_2\sin(\omega\ln\tau), $$

unde $C_1 = C\cos\phi$ și $C_2 = C\sin\phi$, după formula cosinusului unei diferențe. Faza dispare din ecuație și este înlocuită de un al doilea coeficient liniar.

După dezvoltare, ecuația LPPL este liniară în $A$, $B$, $C_1$ și $C_2$:

$$ \ln P(t) = A + B\,f_t + C_1\,g_t + C_2\,h_t, $$ {#eq:linear}

cu $f_t = \tau^m$, $g_t = \tau^m\cos(\omega\ln\tau)$ și $h_t = \tau^m\sin(\omega\ln\tau)$. Odată ce $t_c$, $m$ și $\omega$ sînt fixați, $f_t$, $g_t$ și $h_t$ sînt numere cunoscute pentru fiecare dată, iar @eq:linear este o regresie liniară obișnuită.

## Metoda în doi pași a lui Filimonov și Sornette (2/2) {h=faa85001}

Pasul 1 este liniar. Pentru valori date ale lui $(t_c, m, \omega)$, $\ln P_t$ este regresat prin OLS pe o constantă, $f_t$, $g_t$ și $h_t$:

$$ (\hat A, \hat B, \hat C_1, \hat C_2)' = (X'X)^{-1}X'y, $$

unde $X$ este matricea cu coloanele $1, f_t, g_t, h_t$, iar $y$ vectorul prețurilor logaritmice. Reziduurile acestei regresii dau costul concentrat $\mathrm{SSR}(t_c, m, \omega)$, o funcție doar de cei trei parametri neliniari, deoarece cei patru parametri liniari sînt fixați mereu la cele mai bune valori ale lor.

Pasul 2 este neliniar: $\mathrm{SSR}(t_c, m, \omega)$ este minimizat pe spațiul de căutare $t_c \in [t_2, t_2 + (t_2 - t_1)/3]$, $m \in [0, 1]$ și $\omega \in [1, 50]$. Codul evaluează costul pe o grilă de $8\times 8\times 16$ puncte și pornește algoritmul simplex Nelder–Mead, un algoritm de optimizare fără derivate, din cel mai bun punct al grilei. Grila protejează împotriva minimelor locale; algoritmul simplex rafinează cel mai bun candidat.

La final amplitudinea și faza oscilației se recuperează din cei doi coeficienți liniari: $C = \sqrt{C_1^2 + C_2^2}$ și $\phi = \operatorname{atan2}(C_2, C_1)$, unghiul punctului $(C_1, C_2)$ în intervalul $(-\pi, \pi]$. Spațiul de căutare este mai larg decît filtrul: o ajustare se poate opri la $m$ aproape de 1 sau la $\omega$ aproape de 1, iar filtrul, nu algoritmul de optimizare, este cel care respinge astfel de ajustări.

## Peisajul funcției de cost {h=b1c0a088}

Graficul arată funcția de cost pentru Nasdaq 100, cu o fereastră de la minimul din octombrie 1998 pînă la 30 de zile înainte de maximul din martie 2000. Panoul din stînga este o hartă termică a lui $\log\mathrm{SSR}$, cu timpul critic $t_c$ pe axa orizontală și exponentul $m$ pe axa verticală; în fiecare punct costul este minimizat după $\omega \in [2, 25]$, cu pasul liniar rezolvat pentru fiecare dintre cele 135 360 de puncte ale grilei. Culorile închise înseamnă cost mic; steaua roșie marchează minimul global, iar linia portocalie întreruptă maximul real.

Panoul din dreapta este profilul costului: pentru fiecare $t_c$, minimul SSR după $m$ și $\omega$, tot cu maximul real marcat. Harta termică este aproape uniform închisă pe cea mai mare parte a planului, cu cost ridicat doar pentru $m$ mic și $t_c$ apropiat de sfîrșitul ferestrei. Profilul este o linie ondulată care coboară ușor din martie pînă în august 2000, cu vîrfuri și văi regulate.

## Interpretarea peisajului funcției de cost {h=985b65e1}

Profilul după $t_c$ are 5 minime locale. Un algoritm de optimizare pornit de la un alt $t_c$ poate deci ajunge într-o altă vale și raporta un alt timp critic, cu aproape aceeași calitate a ajustării. Valurile provin din termenul oscilant: mutarea lui $t_c$ mută faza ciclurilor log-periodice față de date, iar ajustarea se îmbunătățește și se înrăutățește alternativ, pe măsură ce ciclurile modelului intră și ies din pas cu corecțiile din date.

Peisajul este și plat. Cel mai slab și cel mai bun $t_c$ de pe profil diferă cu doar 6,3% în SSR, deci datele abia deosebesc un timp critic în martie de unul în august. Minimul global se află la $m$ aproape de 1 și la $t_c$ pe 19 iulie 2000, la aproape patru luni după maximul real. Cu $m$ aproape de 1 termenul de tip lege de putere este aproape o linie dreaptă, așa că cea mai bună ajustare spune că această fereastră nu arată o accelerare clară; datele abia identifică $t_c$.

Lecția este că o singură ajustare LPPL nu este niciodată suficientă. O singură estimație a lui $t_c$ dintr-o singură fereastră depinde de minimul local, de fereastră și de schimbări mici ale datelor. Remediul, folosit în secțiunea următoare, este ajustarea pe multe ferestre și păstrarea doar a ajustărilor care trec de filtru.

## Exemplu rezolvat: ajustarea pentru Shanghai la sfîrșitul ferestrei {h=c2962b0a}

Fereastra pentru Shanghai Composite merge de la 20 martie 2014 pînă la 13 mai 2015, 281 de zile de tranzacționare. Estimațiile neliniare sînt $t_c = 2015{,}428$ în ani zecimali, adică 6 iunie 2015, $m = 0{,}392$ și $\omega = 5{,}50$. Pasul 1 dă apoi $\hat A = 8{,}893$, $\hat B = -1{,}270$, $\hat C_1 = 0{,}020$ și $\hat C_2 = 0{,}074$, deci $\hat C = \sqrt{0{,}020^2 + 0{,}074^2} = 0{,}076$.

Exemplul calculează valoarea ajustată la ultima dată a ferestrei, $t_2 = 2015{,}361$. Timpul rămas este $\tau = t_c - t_2 = 0{,}067$ ani, aproximativ 24 de zile; $\tau^m = 0{,}067^{0{,}392} = 0{,}347$ și $\ln\tau = -2{,}702$. Argumentul oscilației este $\omega\ln\tau = 5{,}50\cdot(-2{,}702)$, cu $\cos(\omega\ln\tau) = -0{,}656$ și $\sin(\omega\ln\tau) = -0{,}754$. Cei trei termeni ai ecuației @eq:linear sînt nivelul $\hat A = 8{,}893$, creșterea de tip lege de putere $\hat B\tau^m = -0{,}440$ și oscilația $\tau^m(\hat C_1\cos + \hat C_2\sin) = -0{,}024$, deci

$$ \widehat{\ln P} = 8{,}893 - 0{,}440 - 0{,}024 = 8{,}429, $$

față de un preț logaritmic observat de 8,384.

În niveluri, valoarea ajustată este $e^{8{,}429} = 4579$ de puncte, față de 4376 observat, o eroare de 4,7%, mult sub limita de 15% a filtrului. Descompunerea arată rolurile termenilor: nivelul $A$ este prețul logaritmic care ar fi atins la $t_c$, termenul de tip lege de putere măsoară cît de departe sub acest nivel se află încă prețul, iar oscilația adaugă o corecție mică, ce depinde de faza ciclului la acea dată.

## Ajustări LPPL cu 30 de zile înainte de maximum {h=1ca8672c}

Graficul are șase panouri, cîte unul pentru fiecare episod, fiecare reprezentînd prețul logaritmic în funcție de timp. Fereastra fiecărei ajustări merge de la minimul dinaintea creșterii pînă la 30 de zile calendaristice înainte de maximum. Linia roșie este ajustarea LPPL, prelungită dincolo de fereastră pînă la timpul critic estimat $\hat t_c$; linia verticală punctată marchează sfîrșitul ferestrei $t_2$, linia neagră întreruptă maximul real, iar linia verde linie-punct $\hat t_c$. Datele de după fereastră, reprezentate în culoarea seriei, nu au fost folosite în ajustare și sînt arătate doar pentru comparație.

În interiorul ferestrelor ajustările urmează datele îndeaproape în toate cele șase panouri. Diferențele apar după $t_2$. Pentru S&P 500 și Nasdaq 100 ajustarea prelungită continuă să crească pînă la un $\hat t_c$ în august 2000, la luni după maximum, în timp ce prețul se aplatizează și scade. Pentru BET timpul critic estimat coincide cu sfîrșitul ferestrei, înainte de maximum. Pentru Shanghai Composite $\hat t_c$ cade cu cîteva zile înainte de maximum. Pentru Bitcoin în 2017 și 2021 ajustarea prelungită urcă dincolo de maximum, pînă la un $\hat t_c$ aflat cu săptămîni mai tîrziu.

## Interpretarea celor șase ajustări {h=e5910dfc}

Tabelul dă, pentru fiecare episod, sfîrșitul ferestrei $t_2$, timpul critic estimat, diferența în zile dintre $\hat t_c$ și maximul real, estimațiile lui $m$ și $\omega$ și două verdicte: dacă ajustarea trece de condițiile asupra parametrilor și dacă trece și de condițiile asupra reziduurilor. Pentru S&P 500 fereastra se încheie la 23 februarie 2000, iar $\hat t_c$ este 21 august 2000, la 150 de zile după maximum, cu $\hat m = 1{,}00$ și $\hat\omega = 1{,}0$. Pentru Nasdaq 100 fereastra se încheie la 25 februarie 2000, iar $\hat t_c$ este 11 august 2000, la 137 de zile după maximum, cu $\hat m = 1{,}00$ și $\hat\omega = 5{,}9$.

Celelalte patru episoade dau imagini diferite. Pentru BET, $\hat t_c$ este egal cu sfîrșitul ferestrei, 22 iunie 2007, cu 32 de zile înainte de maximum, cu $\hat m = 0{,}69$ și $\hat\omega = 1{,}0$. Pentru Shanghai Composite, $\hat t_c$ este 6 iunie 2015, cu 6 zile înainte de maximul din 12 iunie, cu $\hat m = 0{,}39$ și $\hat\omega = 5{,}5$. Pentru Bitcoin, $\hat t_c$ este 22 februarie 2018, la 68 de zile după maximum, în 2017, și 15 mai 2021, la 32 de zile după maximum, în 2021, cu $\hat m$ egal cu 0,71 și 0,82 și $\hat\omega$ egal cu 14,1 și 17,4.

Doar ajustarea pentru Shanghai trece de tot filtrul, cu $\hat t_c$ la mai puțin de o săptămînă de maximum. Ajustările pentru S&P 500 și Nasdaq 100 ating limita $m = 1$ a spațiului de căutare, ceea ce înseamnă că ferestrele nu arată o creștere superexponențială, iar ajustările pentru S&P 500 și BET ating și limita inferioară $\omega = 1$. Cele două ajustări pentru Bitcoin plasează $t_c$ prea departe după sfîrșitul ferestrei pentru condiția asupra timpului critic și nu trec de condiția erorii de 15%: variațiile zilnice ale Bitcoin sînt prea mari pentru ca o singură traiectorie netedă să rămînă la cel mult 15% de fiecare preț observat.

## Recapitulare: estimarea {h=2b23e57f}

Metoda lui @FilimonovSornette2013 estimează patru parametri liniari prin OLS și caută după cei trei parametri neliniari, $t_c$, $m$ și $\omega$, folosind forma liniară @eq:linear. Suprafața de cost este accidentată, cu mai multe minime locale, și plată în direcția lui $t_c$, deci timpul critic este slab identificat.

O fereastră dă o ajustare. Cu date care se opresc cu 30 de zile înainte de maximum, doar unul dintre cele șase episoade, Shanghai Composite în 2015, dă o ajustare complet calificată.

## @end

O singură ajustare este fragilă: depinde de începutul ferestrei, care în timp real nu este cunoscut, și de o funcție de cost care abia deosebește un timp critic de altul. Secțiunea următoare înlocuiește ajustarea unică prin multe ajustări pe ferestre de lungimi diferite și le rezumă într-un singur număr.

# Estimarea pe mai multe ferestre

Secțiunea construiește indicatorul de încredere LPPLS, ponderea ajustărilor calificate printre multe ferestre care se încheie la aceeași dată. Arată indicatorul în jurul a patru maxime și felul în care timpul critic estimat se mută pe măsură ce sosesc date noi.

## Indicatorul de încredere LPPLS {h=0b30f3f0}

Ajustările din secțiunea anterioară porneau fereastra de la minimul dinaintea creșterii, o dată cunoscută doar după bulă, deoarece minimul este identificat ca minim doar după ce prețurile au crescut. În timp real începutul bulei nu este cunoscut. Remediul propus de @ShuZhu2020 este folosirea mai multor începuturi: pentru fiecare dată de sfîrșit $t_2$, care joacă rolul zilei de azi, modelul este ajustat pe 29 de ferestre $[t_2 - L, t_2]$, cu lungimile $L = 750, 725, \dots, 50$ de zile de tranzacționare.

Indicatorul este ponderea ferestrelor care se încheie la $t_2$ și au o ajustare calificată [@SornetteEtAl2015]. El se află între 0, cînd nicio fereastră nu vede o bulă, și 1, cînd toate văd. Numele LPPLS (*log-periodic power law singularity*, singularitatea legii de putere log-periodice) este folosit în această literatură pentru aceeași ecuație @eq:lppl și a dat numele indicatorului.

Capitolul folosește două versiuni ale indicatorului. Versiunea cu parametri numără ajustările care trec de condițiile asupra lui $B$, $m$, $\omega$, $t_c$, a numărului de oscilații și a amortizării. Versiunea completă cere în plus condițiile asupra reziduurilor: eroarea de 15%, testul Lomb și reziduurile staționare, ca în @ShuZhu2020. Ambele se calculează doar cu date pînă la $t_2$, deci sînt indicatori în timp real. O alarmă este emisă cînd indicatorul depășește un prag fixat dinainte, aici 0,2.

## Timpul critic estimat pe mai multe ferestre de estimare {h=a5f85496}

Graficul arată cele 29 de ajustări pentru două episoade, Nasdaq 100 în 2000 în stînga și Shanghai Composite în 2015 în dreapta, cu toate ferestrele încheiate cu 30 de zile înainte de maximum. Axa orizontală este lungimea ferestrei $t_2 - t_1$, în observații, de la 50 la 750; axa verticală este timpul critic estimat. Punctele verzi sînt ajustările care trec de condițiile asupra parametrilor, iar crucile roșii cele care nu trec; linia roșie întreruptă este maximul real, iar linia punctată sfîrșitul ferestrelor $t_2$.

Pentru Nasdaq 100 punctele verzi se află aproape de maximum, la lungimi ale ferestrei de aproximativ 75 pînă la 300 și 550 pînă la 675 de observații, în timp ce multe cruci roșii se află mult mai sus, cu $\hat t_c$ pînă în martie 2001. Pentru Shanghai Composite majoritatea estimațiilor se află între mai și septembrie 2015, punctele verzi sînt la lungimi ale ferestrei între 250 și 350, iar cele mai lungi ferestre dau cruci roșii în 2016.

## Interpretarea ferestrelor {h=a2e5b914}

Pentru Nasdaq 100, cu ferestre încheiate la 25 februarie 2000, 11 dintre cele 29 de ferestre trec de condițiile asupra parametrilor, iar 1 trece de tot filtrul, deci indicatorul cu parametri este egal cu $11/29 = 0{,}38$ la acea dată. Mediana lui $\hat t_c$ pentru ajustările calificate este 30 martie 2000, iar intervalul dintre cuantilele lor de 10% și 90% merge de la 25 februarie pînă la 2 aprilie 2000; maximul real a fost la 27 martie 2000. În acest caz ajustările calificate indică săptămînile potrivite.

Pentru Shanghai Composite, cu ferestre încheiate la 13 mai 2015, 5 dintre cele 29 de ferestre trec de condițiile asupra parametrilor, iar mediana lui $\hat t_c$ este 6 iunie 2015, față de un maxim real la 12 iunie. Condiția asupra lui $m$ eșuează cel mai des: doar 41% dintre ferestrele pentru Shanghai au $0{,}01 \le m \le 0{,}99$, așa că majoritatea ajustărilor se opresc la limita spațiului de căutare pentru exponent.

Graficul arată și o dependență de lungimea ferestrei. Ferestrele scurte pun $t_c$ imediat după $t_2$, în timp ce ferestrele lungi îl împrăștie, uneori departe în viitor. Lungimea ferestrei este deci o alegere de modelare de care depinde rezultatul, chiar dacă este ușor de trecut cu vederea; medierea pe multe lungimi, așa cum face indicatorul, face dependența vizibilă în loc să o ascundă.

## Indicatorul în jurul a patru maxime (1/2) {h=ec321dbb}

Graficul acoperă două episoade, Nasdaq 100 în 2000 și Shanghai Composite în 2015, de la 18 luni înainte pînă la 4 luni după maximum. Prețul logaritmic este reprezentat pe axa din stînga, în culoarea seriei, iar indicatorul pe axa din dreapta, de la 0 la 1, calculat la fiecare 5 zile de tranzacționare. Zona verde este versiunea cu parametri a indicatorului, iar linia violet versiunea completă; linia roșie întreruptă marchează maximul.

Pentru Nasdaq 100 zona verde este ridicată în două perioade: prima în iarna și primăvara lui 1999, cu peste un an înainte de maximum, și a doua în ultimele luni dinaintea maximului. Linia violet rămîne jos tot timpul. Pentru Shanghai Composite ambele versiuni sînt aproape de zero în cea mai mare parte a perioadei și urcă împreună în cele două luni dinaintea maximului.

## Indicatorul în jurul a patru maxime (2/2) {h=937a8083}

Graficul folosește aceeași așezare și aceleași culori pentru BET în 2007 și Bitcoin în 2017; pentru Bitcoin, care se tranzacționează în fiecare zi, indicatorul este calculat la fiecare 7 zile. Pentru BET zona verde este ridicată la începutul perioadei, la începutul lui 2006, apoi mică timp de peste un an, cu excepția unei urcări scurte la începutul lui 2007, și din nou ridicată în jurul și după maximul din iulie 2007; linia violet urcă mai ales după maximum.

Pentru Bitcoin zona verde este prezentă în cea mai mare parte a creșterii, cu bare care devin mai înalte pe măsură ce prețul urcă, iar cele mai înalte se află aproape de maximul din decembrie 2017. Linia violet, versiunea completă, este aproape tot timpul la zero.

## Interpretarea indicatorului {h=61035d4d}

Pentru Nasdaq 100 versiunea cu parametri ajunge la 0,52 pe 7 aprilie 1999, cu un an înainte de maximum, și urcă din nou în ultimele luni dinaintea lui; versiunea completă rămîne sub 0,14. Pentru Shanghai Composite ambele versiuni urcă în cele două luni dinaintea maximului, iar ambele își ating maximul pe 22 aprilie 2015, cu șapte săptămîni înainte de maximul din 12 iunie: 0,24 pentru versiunea completă și 0,28 pentru versiunea cu parametri.

Pentru BET cel mai puternic semnal cu filtrul complet, 0,28, vine pe 28 august 2007, la o lună după maximum. La acea dată ferestrele conțin întreaga creștere și începutul scăderii, iar ajustările descriu ultima parte a creșterii, nu viitorul. Pentru Bitcoin în 2017 versiunea cu parametri își atinge maximul, 0,66, pe 29 decembrie 2017, după maximul din 16 decembrie, în timp ce versiunea completă este aproape tot timpul 0, deoarece condiția erorii de 15% nu este îndeplinită în aproape nicio fereastră.

Indicatorul se aprinde în timpul bulelor, dar devreme, tîrziu și cu întreruperi, iar alegerea filtrului schimbă complet imaginea: cu condițiile asupra parametrilor Bitcoin arată ca o bulă în cea mai mare parte a lui 2017, cu toate condițiile aproape niciodată. Un cititor care vede doar una dintre cele două versiuni trage o altă concluzie decît unul care o vede pe cealaltă.

## Instabilitatea timpului critic estimat {h=0663e9a4}

Graficul urmărește timpul critic estimat pe măsură ce datele cresc. Începutul ferestrei este fixat la minimul dinaintea creșterii, iar sfîrșitul $t_2$ se mută săptămînal de la 8 luni înainte pînă la 1 lună după maximum. Axa orizontală este $t_2$, axa verticală este $\hat t_c$ al ajustării care se încheie la $t_2$; punctele verzi sînt ajustări care trec de condițiile asupra parametrilor, iar crucile roșii ajustări care nu trec. Diagonala punctată este dreapta $t_c = t_2$, o ajustare care plasează sfîrșitul bulei la ultima observație, iar linia roșie întreruptă este maximul real.

Pentru Nasdaq 100, în stînga, crucile formează o linie dreaptă care urcă regulat deasupra diagonalei: cînd $t_2$ se mută din august 1999 în aprilie 2000, $\hat t_c$ se mută din noiembrie 1999 în octombrie 2000. Aceste estimații se află la capătul superior al intervalului de căutare, $t_2 + (t_2 - t_1)/3$, care se mută înainte odată cu $t_2$. Doar două ajustări, în februarie 2000, întrerup linia cu valori apropiate de maximum. Pentru Bitcoin în 2017, în dreapta, punctele rămîn aproape de diagonală și o urmează în sus din mai 2017 pînă în ianuarie 2018, cu puncte verzi mai ales începînd din august 2017.

## Interpretarea timpului critic care se mută {h=c2b106ab}

Pentru Nasdaq 100, pe 38 de date de sfîrșit, $\hat t_c$ variază de la 9 noiembrie 1999 pînă la 30 octombrie 2000, un interval de 356 de zile, și niciuna dintre aceste ajustări nu trece de condițiile asupra parametrilor. Pentru Bitcoin în 2017, $\hat t_c$ variază de la 21 mai 2017 pînă la 12 aprilie 2018; 38% dintre ajustări trec de condițiile asupra parametrilor, iar eroarea mediană, distanța mediană dintre $\hat t_c$ și maximul real, este de 65 de zile.

Cele două cazuri arată două feluri de a eșua. Pentru Bitcoin, 4 dintre cele 39 de ajustări pun $\hat t_c$ exact la sfîrșitul ferestrei $t_2$, iar 11 la cel mult 20 de zile de el: aceste ajustări spun „sfîrșitul este acum”. Pentru Nasdaq 100 estimația rămîne la marginea spațiului de căutare și se mută înainte odată cu ea. În ambele cazuri timpul critic estimat urmează sfîrșitul datelor în loc să rămînă la o dată fixă. O prognoză care se mută mereu odată cu datele este o descriere a trecutului, nu o predicție.

Graficul arată și cum apare o falsă impresie de succes. Printre multele ajustări, unele au plasat $t_c$ aproape de maximul real. Alegerea, după crah, a singurei ferestre al cărei $\hat t_c$ a fost corect și prezentarea ei drept prognoză înseamnă look-ahead bias: folosește cunoașterea momentului crahului pentru a alege fereastra.

## Recapitulare: estimarea pe mai multe ferestre {h=42a630e7}

Începutul unei bule nu este cunoscut în timp real, așa că modelul este ajustat pe 29 de ferestre de lungimi diferite pentru fiecare dată de sfîrșit. Indicatorul este ponderea ferestrelor calificate și depinde puternic de filtru: versiunea cu parametri și versiunea completă pot spune povești diferite.

Timpul critic estimat se mută odată cu sfîrșitul datelor. El trebuie raportat prin împrăștierea lui pe ferestre și pe date de sfîrșit, niciodată ca o singură valoare.

## @end

Ferestrele multiple fac vizibilă incertitudinea ajustării LPPL și o transformă într-un indicator care poate fi calculat în timp real. Dacă indicatorul ajută la anticiparea crahurilor este o întrebare separată, la care răspunde secțiunea următoare prin evaluarea alarmelor lui pe tot eșantionul.

# Evaluarea prognozelor de crah

Secțiunea prezintă capcanele evaluării prognozelor de crah, rulează indicatorul pe tot eșantionul pentru S&P 500 și Bitcoin, cu toate alarmele lui, compară rata de reușită după alarme cu frecvența de bază și pune rezultatele alături de literatură.

## Erori frecvente în evaluarea prognozelor de crah {h=41ac8f9f}

Prima capcană este look-ahead bias: folosirea unei informații care nu era disponibilă la data prognozei $t_2$. Ea intră pe căi subtile: alegerea începutului ferestrei $t_1$ la minimul creșterii, o dată cunoscută doar mai tîrziu; alegerea episoadelor studiate după ce crahurile au avut loc; sau alegerea condițiilor filtrului după ce s-a văzut care dau rezultate bune. Fiecare dintre aceste alegeri folosește viitorul pentru a face trecutul să pară previzibil.

A doua capcană este selecția (*selection*): studierea doar a bulelor care s-au spart. Creșterile care s-au încheiat fără crah sau care au continuat ani de zile sînt uitate, deși o metodă care a dat o alarmă în timpul lor a greșit. Un eșantion format doar din crahuri nu poate măsura alarmele false.

A treia capcană privește alarmele false (*false alarms*) și frecvențele de bază (*base rates*). O alarmă este utilă doar dacă un crah este mai probabil după o alarmă decît într-o zi obișnuită. Comparația relevantă este între $P(\text{crah} \mid \text{alarmă})$, ponderea datelor cu alarmă urmate de un crah, și $P(\text{crah})$, frecvența necondiționată a crahurilor pe toate datele. Dacă crahurile urmează după 7% dintre toate datele, o alarmă urmată de un crah în 7% din cazuri nu adaugă nicio informație. Un test corect fixează totul dinainte, ferestrele, filtrul, pragul și definiția crahului, și rulează pe tot eșantionul.

## Alarmele pe tot eșantionul (1/2) {h=e6cbe0f8}

Graficul aplică acest protocol pe S&P 500 din ianuarie 1993 pînă în septembrie 2026, cu indicatorul calculat la fiecare 5 zile de tranzacționare. Un crah este definit ca o scădere de cel puțin 20% în cele 182 de zile de după data alarmei. Panoul din stînga reprezintă indicele pe scară logaritmică, cu datele de alarmă ale versiunii cu parametri la pragul 0,2: punctele verzi ar marca alarmele urmate de un crah, iar crucile roșii marchează alarmele false.

Toate alarmele din panoul din stînga sînt cruci roșii. Ele apar în grupuri în timpul creșterilor constante: în 1995–1999, în 2006–2007, în 2012–2014, în 2017–2018 și în ultimii ani ai eșantionului. Panoul din dreapta reprezintă rata de reușită (*hit rate*), ponderea datelor cu alarmă urmate de un crah, în funcție de pragul $c$ de la 0,05 la 0,4, pentru versiunea cu parametri și pentru versiunea completă, cu frecvența necondiționată ca linie roșie întreruptă. Ambele linii ale ratei de reușită se află sub linia întreruptă la toate pragurile, între 0,02 și 0,03 la cele mai mici două praguri și la zero începînd de la 0,2.

## Alarmele pe tot eșantionul (2/2) {h=d7b97d19}

Graficul repetă evaluarea pentru Bitcoin din ianuarie 2014 pînă în septembrie 2026, cu indicatorul calculat la fiecare 7 zile. În panoul din stînga alarmele versiunii cu parametri la pragul 0,2 se grupează la începutul lui 2014, în 2016–2017, la sfîrșitul lui 2020 și începutul lui 2021 și în 2024–2025; punctele verzi, alarme urmate de o scădere de 20% sau mai mult în șase luni, sînt amestecate cu cruci roșii.

În panoul din dreapta rata de reușită a versiunii cu parametri, cu verde, este aproape de linia întreruptă a frecvenței necondiționate la pragurile mici și urcă peste ea la pragurile mari. Versiunea completă, cu violet, are puncte doar la cele mai mici două praguri, deoarece la pragurile mai mari nu emite nicio alarmă.

## Interpretarea evaluării {h=0a03d84b}

Pentru S&P 500, o scădere de 20% în șase luni urmează după 7% dintre toate datele, dar după 0% dintre datele cu alarmă ale versiunii cu parametri la pragul 0,2. Alarmele formează 20 de grupuri, și niciunul nu este urmat de o astfel de scădere în șase luni: alarmele apar în creșteri calme și constante, pe care ajustările LPPL le descriu ca o creștere accelerată. Dintre cele 4 scăderi de 20% sau mai mult din ianuarie 1993, 2 au avut o alarmă în cele șase luni dinaintea maximului lor, în 2007 și în 2022, dar scăderile au venit la mai mult de șase luni după alarme.

Pentru Bitcoin frecvența de bază este 47%: aproape jumătate dintre toate datele sînt urmate de o scădere de 20% în șase luni. După o alarmă ponderea este 49%, aproape niciun cîștig. În același timp 12 dintre cele 14 scăderi de 20% sau mai mult au avut o alarmă înaintea maximului lor, ceea ce pare impresionant pînă cînd este comparat cu frecvența de bază. Scăderile de 20% sînt atît de frecvente la Bitcoin încît aproape orice alarmă pare corectă. Pragurile mai mari ajută puțin, dar pe foarte puține date: la pragul 0,3 rata de reușită este 56%, pe 4% dintre date.

Concluzia rezultă din cele două piețe luate împreună. Indicatorul spune „aceasta arată ca un regim de bulă”, în sensul unei creșteri accelerate; nu spune „crahul vine în șase luni”. Într-un indice calm regimul continuă adesea, iar într-un activ volatil crahurile vin fie că există o alarmă, fie că nu.

## Literatura despre prognoza crahurilor {h=febff74c}

Autorii metodei raportează succese. @JiangEtAl2010 au diagnosticat dinainte bulele pieței chineze de acțiuni din 2005–2007 și 2008–2009, iar @SornetteEtAl2015 au anunțat maximul pieței din Shanghai din 2015 înainte să aibă loc și l-au analizat ulterior. @GerlachDemosSornette2019 au analizat în detaliu bulele Bitcoin din 2012 pînă în 2018 cu indicatorul de încredere.

Evaluările critice ajung la concluzii mai prudente. @Feigenbaum2001 a constatat că termenul log-periodic adaugă puțin la explicarea prețurilor odată ce eroarea de estimare este luată în calcul. @BreeJoseph2013 au studiat multe crahuri și au constatat că toate condițiile LPPL sînt rar îndeplinite înaintea crahului și că parametrii estimați sînt instabili.

Cifrele acestui capitol merg în aceeași direcție ca studiile critice. Indicatorul reacționează la creșterile puternice, ca în cazul Shanghai Composite în 2015, dar momentele pe care le indică sînt imprecise, timpul lui critic se mută odată cu datele, iar pe tot eșantionul alarmele lui sînt adesea false. Succesele raportate în literatură sînt cazuri reale, dar cîteva cazuri reușite nu măsoară rata de reușită față de frecvența de bază.

## Posibilitățile și limitele modelului LPPL {h=31643909}

Modelul LPPL poate descrie o creștere cu o accelerare și corecții tot mai mici prin cîțiva parametri, fiecare cu o interpretare: cît de puternic se accelerează creșterea, cît de repede se accelerează corecțiile, cînd este probabil să se încheie regimul. Prin indicator, poate măsura și cît de mult seamănă azi o piață cu o bulă, pe multe ferestre, cu datele disponibile azi.

Modelul nu poate da data unui crah: $t_c$ marchează sfîrșitul cel mai probabil al regimului, iar estimația lui se mută odată cu datele. Nu poate garanta un crah, deoarece regimul se poate încheia cu o scădere lentă. Nu poate spune nici dacă prețul este peste valoarea fundamentală: o traiectorie superexponențială este o afirmație despre forma prețului, iar doar fundamentele pot spune dacă prețul este prea mare.

Modelul este deci folosit cel mai bine împreună cu alte instrumente: cu testele de rădăcini explozive din acest capitol, cu fundamente precum rapoartele de evaluare și cu măsuri de risc precum VaR 1%, valoarea la risc (*value at risk*) la nivelul de 1% din [Capitolul 5](../RO/Cursuri/capitol5_volatilitate_conditionata_garch.pdf). Folosit singur, descrie bulele mai bine decît prognozează sfîrșitul lor.

## Recapitulare: evaluarea prognozelor de crah {h=4c88ef88}

O evaluare corectă fixează ferestrele, filtrul, pragul și definiția crahului înainte de a privi rezultatele și rulează pe tot eșantionul. Rata de reușită după alarme este comparată cu frecvența de bază, iar fiecare alarmă falsă este numărată.

Modelul LPPL descrie bine bulele. Ca instrument pentru datarea crahurilor, rezultatele lui în datele cursului și în literatura critică sînt slabe.

## @end

Cele două jumătăți ale întrebării capitolului au răspunsuri diferite. Un regim de bulă poate fi recunoscut cît timp crește, prin testele de rădăcini explozive și prin indicatorul LPPL. Sfîrșitul regimului, și în special data unui crah, nu poate fi prognozat în mod fiabil cu aceste instrumente; evaluarea pe tot eșantionul, cu frecvențe de bază și alarme false, este cea care face acest lucru vizibil.

# Contribuția posibilă a AI

Secțiunea descrie ce poate face un asistent de inteligență artificială (*artificial intelligence*, AI) pentru metodele acestui capitol și ce trebuie verificat de fiecare dată.

## Contribuția posibilă a AI {h=a3852c52}

Un asistent AI poate scrie o primă versiune a codului: recursia BSADF pe toate ferestrele, ajustarea în doi pași a lui Filimonov și Sornette și indicatorul pe multe ferestre. Poate da și o a doua explicație, în alte cuvinte și cu propriile cifre, a argumentelor mai dificile ale capitolului, cum ar fi deducerea driftului din rata de hazard sau sensul log-periodicității.

Cel mai mare cîștig este în explorare: calculul indicatorului pe multe piețe și perioade, cu filtre diferite. Explorarea este utilă doar sub un protocol fixat dinainte; altfel devine o căutare a setării care face metoda să arate bine. Prompt-ul de pe slide este un exemplu de cerere precisă: numește datele, metoda pentru fiecare grup de parametri, sursa condițiilor filtrului și graficul, astfel încît rezultatul să poată fi verificat pas cu pas.

## Verificări necesare {h=aa2ffdd7}

Prima verificare este o simulare: generați o traiectorie LPPL cu parametri cunoscuți, adăugați zgomot și verificați că codul recuperează $t_c$, $m$ și $\omega$. Un cod care nu poate recupera parametri cunoscuți din date curate nu poate fi crezut pe date reale. A doua verificare privește semnele și cozile: $B < 0$ pentru o bulă în creștere, iar testul de rădăcini explozive folosește coada din dreapta, nu coada din stînga din [Capitolul 3](../RO/Cursuri/capitol3_radacini_unitare_modele_arima.pdf). SADF și GSADF au nevoie de propriile valori critice simulate, nu de tabelul Dickey–Fuller.

Celelalte verificări privesc timpul și dovezile. Fiecare mărime calculată la data $t_2$ trebuie să folosească doar date pînă la $t_2$; o singură folosire a datelor ulterioare, de exemplu o fereastră care începe la un minim identificat mai tîrziu, invalidează evaluarea. Un crah „prognozat” de un răspuns AI după ce a avut loc nu este o prognoză: pentru a-l judeca este nevoie de lista completă a alarmelor și a alarmelor false. Fiecare referință citată de un asistent trebuie să existe, iar identificatorul ei digital (*digital object identifier*, DOI) trebuie verificat.

## @end

Un asistent scurtează scrierea codului și explorarea multor piețe, dar verificările specifice acestui capitol, simularea cu parametri cunoscuți, coada din dreapta, absența informației din viitor și lista completă a alarmelor, rămîn responsabilitatea analistului.

# Rezumat

Rezumatul adună ideile principale ale capitolului, formulele de reținut, întrebările de autoevaluare și o idee de proiect.

## Idei de reținut {h=fbcd98ee}

O bulă este partea unui preț de peste valoarea lui fundamentală, iar o bulă deținută de investitori raționali trebuie să crească exploziv. Testele ADF la dreapta, SADF, GSADF și BSADF, detectează și datează episoadele explozive folosind datele pînă la fiecare dată; ele datează bine creșterile și tîrziu sfîrșitul lor. Modelul LPPL descrie o bulă ca o creștere superexponențială cu oscilații log-periodice, care se încheie la un timp critic $t_c$.

Modelul se estimează în doi pași, OLS pentru patru parametri și o căutare după $t_c$, $m$ și $\omega$; ajustările sînt filtrate și repetate pe multe ferestre. Prognozele sînt judecate prin rata de reușită față de frecvența de bază, cu toate alarmele false. În datele cursului datarea crahurilor rămîne dificilă: indicatorul recunoaște regimurile de bulă, dar alarmele lui nu sînt urmate de crahuri mai des decît datele obișnuite.

## Formule de reținut {h=6760f147}

Tabelul adună formulele în ordinea capitolului: bula rațională, $P_t = F_t + B_t$ cu $E_t[B_{t+1}] = (1+r)B_t$; regresia ADF la dreapta @eq:adf, cu alternativa $\delta > 0$; GSADF @eq:gsadf și BSADF, cu cea mai mică fereastră $r_0 = 0{,}01 + 1{,}8/\sqrt T$; ecuația LPPL @eq:lppl și forma ei liniară @eq:linear, cu $C = \sqrt{C_1^2 + C_2^2}$; raportul de scalare @eq:lambda; condiția de amortizare $m|B|/(\omega|C|) \ge 1$; și indicatorul, ponderea ferestrelor calificate care se încheie la $t_2$.

Formulele formează două lanțuri. Primul merge de la condiția de lipsă a arbitrajului la creșterea explozivă și la testele care o caută. Al doilea merge de la rata de hazard la ecuația LPPL, de la ecuație la forma ei liniară pentru estimare și de la ajustările individuale la indicatorul construit din multe ferestre.

## Autoevaluare (1/2) {h=06bf7b0c}

Prima întrebare aplică bula Blanchard–Watson cu $r = 1\%$ și $\pi = 0{,}95$. Într-o perioadă în care supraviețuiește, bula crește cu $1{,}01/0{,}95 - 1 \approx 6{,}3\%$, iar durata ei medie de viață este $1/(1 - 0{,}95) = 20$ de perioade. Față de exemplul rezolvat, o probabilitate de supraviețuire mai mică înseamnă o viață mai scurtă și o creștere mai rapidă cît timp durează, deoarece creșterea suplimentară plătește un risc mai mare de spargere.

A doua întrebare aplică testul la dreapta. Cu $\hat\delta = 0{,}004$ și o eroare standard de 0,002, $t = 0{,}004/0{,}002 = 2$, peste valoarea critică de 1,5, deci rădăcina unitară este respinsă în favoarea unei rădăcini explozive la pragul de 5%. A treia întreabă de ce se folosește coada din dreapta: alternativa care interesează este $\rho > 1$, o rădăcină explozivă, nu $\rho < 1$, staționaritatea, iar dovada pentru $\rho > 1$ este o statistică $t$ mare și pozitivă.

## Autoevaluare (2/2) {h=a788249e}

Prima întrebare aplică @eq:lambda: cu $\omega = 6{,}28$, $\lambda = e^{2\pi/6{,}28} \approx e \approx 2{,}72$, deci timpul rămas pînă la $t_c$ scade cu un factor de aproximativ 2,72 între două corecții. A doua privește filtrul: o ajustare cu $m = 1{,}0$, la marginea spațiului de căutare, nu este calificată, deoarece filtrul cere $m \le 0{,}99$; cu $m = 1$ termenul de tip lege de putere este liniar în $t$, un trend fără accelerare.

A treia întrebare privește evaluarea. Un indicator care dă alarme înaintea a 8 din 10 crahuri nu este încă dovedit a fi un bun instrument de prognoză. Sînt necesare și alarmele false, și frecvența de bază: dacă alarmele apar aproape în fiecare zi, prinderea a 8 crahuri din 10 este ușoară și nu spune nimic. Aceasta este exact situația Bitcoin din acest capitol, unde 12 din 14 scăderi au avut o alarmă înaintea maximului lor, dar rata de reușită după alarme a fost aproape egală cu frecvența de bază.

## Idee de proiect {h=2b713cad}

Proiectul întreabă dacă BET este azi într-o bulă, cu un protocol în timp real fixat înainte de a privi rezultatul. Prima parte aplică BSADF pe BET săptămînal și pe indicele BET Total Return (BET-TR), care include dividendele reinvestite, și calculează indicatorul acestui capitol pe BET zilnic. A doua parte aplică evaluarea din secțiunea anterioară pe BET începînd din 2000: toate alarmele, toate alarmele false și frecvența de bază a scăderilor de 20%.

Extensiile posibile înlocuiesc prețul cu raportul preț–profit, care aduce fundamentele în test, sau studiază împreună Bitcoin și Ethereum. O altă extensie compară episoadele datate cu regimurile modelelor Markov-switching din [Capitolul 10](../RO/Cursuri/capitol10_spatiul_starilor_kalman_markov_switching.pdf). Pentru istoria crahurilor de la Bursa de Valori București, @PeleMazurencuMarinescu2012 oferă un punct de plecare. Capitolul următor, [Capitolul 14](../RO/Cursuri/capitol14_modele_garch_multivariate.pdf), trece la modelele GARCH (*generalised autoregressive conditional heteroskedasticity*) multivariate.

## @end

Capitolul a răspuns la întrebarea lui în două părți. Bulele lasă urme statistice, rădăcini explozive și creștere superexponențială, care pot fi detectate cît timp bula este în desfășurare. Data sfîrșitului nu poate fi citită în mod fiabil din aceste urme: timpul critic este slab identificat, se mută odată cu datele, iar alarmele construite din el trebuie judecate față de frecvența de bază, cu fiecare alarmă falsă numărată.
