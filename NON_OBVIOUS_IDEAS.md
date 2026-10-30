# WIĄZKA / RGT — zastosowania nieoczywiste (poza przerabianiem zapytań)

Zestaw pomysłów, w których **nie** chodzi o „przekształć zapytanie i szukaj
lepiej", tylko o to, że model daje coś, czego nie da żadna metryka cosine ani
żaden clustering: **relację jako obiekt algebraiczny działający w przestrzeni
wektorów tekstu** — odwracalnie, z siłą `λ`, z oceną „czy w ogóle definida"
i z oceną „która to była relacja?".

Punkt wyjścia (podany w rozmowie): zamiast klasteryzować po odległości —
szukać skupień **po relacji** (synonimia = przyciąganie, antonimia =
odpychanie). Poniżej uogólnienie tego pomysłu i 11 dalszych, w tym
prywatność, CI dla modeli, dane treningowe, nieostrość psycholingwistyczna
i psychometria rozumowania.

Wersja „wprost, RAG-owa" jest w `LLM_APPLICATION_IDEAS.md` i
`LLM_USE_CASES.md`; tu jest to, co obok.

> **Każdy pomysł ma pełną specyfikację** (formalizacja, algorytm na realnym API,
> protokół pomiaru, baseline'y, ryzyka i test go/no-go) w `docs/IDEAS/` — indeks
> wraz z warunkami brzegowymi: `docs/IDEAS/00-INDEKS.md`. Poniżej wersja
> skrócona: szybki przegląd i tabela priorytetów.

---

## 0. Prymitywy (żeby każdy pomysł można było zweryfikować)

| # | prymityw | API / kod | co daje unikalnego |
|---|---|---|---|
| P1 | `T_r` = obrót + skala, **odwracalny dokładnie** | `RelationFlow.apply`, `inverse` | transformację można *cofnąć* bez utraty informacji |
| P2 | izometria do skali (`ISOT`) | `core/metrics/geometry.py` | cosine i normy nie są niszczone — transformacja nie „psuje" przestrzeni |
| P3 | dial `λ ∈ [0,1]`, `T(0)=I` | `apply(..., lam=)` | natężenie sensu jako liczba ciągła, nie decyzja binarna |
| P4 | `decode(x, y)` → relacja + confidence | `RelationDecoder` | **odwrotny problem**: „co tu zaszło między tymi dwoma wektorami?" |
| P5 | `applicability(v, r)` + budżet, kalibrowany (`ECE`) | `ApplicabilityHead` | „czy ta operacja ma tu sens" — typowanie bez etykiet |
| P6 | `C_rs` — uczona macierz przemienności | `commutation_C.csv`, `CurvatureLoss` | **drugiego rzędu** wiedza o relacjach: które osy są niezależne, gdzie kolejność ma znaczenie |
| P7 | spójność ścieżkowa (holonomia `HD`, `PCE(k)`) | `core/metrics/geometry.py` | miara *spójności wewnętrznej*, niezależna od poprawności odpowiedzi |
| P8 | operator ważny **poza grafem** (`OMD`, shadow manifold) | `core/metrics/offline.py`, F4 | aplikacja do dowolnego tekstu, nie tylko do haseł słownika |

Wspólny mianownik nowych zastosowań: **P4 + P5 + P6 zamieniają „podobieństwo"
w „typowaną relację"**, a **P1 + P3** dają nad tym kontrolę (siła, odwrót).

---

## 1. Klasteryzacja relacyjna: skupienia jako orbity, nie jako chmury

**Prymitywy:** P4, P6, P1.

**Teza.** Cosine trzyma „bateria świetna" i „bateria fatalna" w tym samym
klastrze — to udokumentowana wada metryki (por. *Is Cosine-Similarity of
Embeddings Really About Similarity?*, *Intuitive Contrasting Map for Antonym
Embeddings*). Zamiast metryki: **klaster = orbita podgrupy relacji
dozwolonych** (`syn`, `hyper(λ<t₀)`), a **antonimia jako krawędź ujemna**.

**Jak.**
1. Nad zbiorem N wektorów: `decode` dla par z `cos > 0.6` (kandydaci) →
   krawędzie `typ + conf`;
2. graf znakowany: `syn`/`hypo`/`hyper` → przyciąganie (waga = conf),
   `ant` → odpychanie; pary spoza słownika → brak krawędzi (nie: „słabe
   przyciąganie");
3. klasteryzacja grafu znakowanego: *correlation clustering* / etykietowanie
   rozpropagowane na grafie znakowanym / strukturalna równowaga Heidera;
4. oś polaryzacji = kierunek, wzdłuż którego krawędzie ujemne są najgęstsze
   (z wektora różnic transportów) — **nazwana przez relację**, nie
   domyślona przez PCA.

**Co poprawia i metryka.** ARI/purity vs etykiety ludzkie na opiniach i
dyskusjach; metryka własna **contradiction-purity** (frakcja klastrów bez
pary `ant`) — baseline cosine-kmeans i SemDeDup-style próg cosine powinien
wypadać źle, i to jest teza do zmierzenia. Dodatkowa metryka: liczba klastrów
przy tym samym ARI (redukcja przy zachowaniu sprzeczności).

**Fazy.** (1) skrypt `decode`-batch nad próbką 20k wektorów + graf znakowany,
4 dni; (2) clustering + baseline'y (cosine-kmeans, HDBSCAN, SemDeDup),
1 tydzień; (3) etykiety ludzkie na 3 zbiorach (opinie, debaty, QA), 1 tydzień.

**Ryzyka.** `decode` jest wytrenowany na relacjach słownikowych — dla par
zdaniowych trzeba dolać shadow-zdania w F4 (patrz §Uwagi na końcu).
Dla par „brak relacji" dekoder może halucynować — mitygacja: próg conf +
`applicability` jako filtr ważności krawędzi.

## 2. Deduplikacja danych treningowych, która nie zjada zaprzeczeń

**Prymitywy:** P4, P6, P1.

**Teza.** SemDeDup i przyjaciele usuwają „duplikaty semantyczne" po progu
cosine. Skutek uboczny: wycięte zostają **pary contrastive** („X jest
skuteczny" / „X nie jest skuteczny"), czyli dokładnie dane uczące obsługi
negacji i sprzeczności. To znany, ale mało naprawiony problem — istniejące
poprawki dotyczą sprawiedliwości (`FairDeDup`), nie negacji.

**Jak.** Dedup typowany: w obrębie orbity `syn` → zostaw 1 reprezentanta;
para `ant` → **zatrzymaj obie i wypuść jako parę contrastive** do
fine-tuningu; para `hyper` → zatrzymaj ogólnik i wystarczającą liczbę
instancji (nie wszystkie). Realizacja: ten sam graf znakowany co w p. 1 +
reguła decyzyjna per typ krawędzi.

**Co poprawia i metryka.** Przy tym samym budżecie tokenów: dokładność na
ewaluacjach wrażliwych na negację i na contrast setach (spadek Accuracy po
zamianie „nie" jest klasycznym objawem); krzywa uczenia; osobno: recall
stosunków logicznych (NLI-style probe). Baseline: SemDeDup, próg cosine,
losowe podpróbkowanie.

**Fazy.** (1) implementacja reguły typującej na istniejącym korpusie, 3 dni;
(2) dwa runy fine-tune (baseline vs typowana), 1 tydzień GPU-kolejka;
(3) ewaluacja wrażliwa na negację + ablacja (bez `ant`-pariusowania), 3 dni.

**Ryzyka.** Koszt `decode`-batch (O(kandidatów), nie O(N²) — kandydaci z
ANN); ryzyko merytoryczne małe, bo reguła jest jawna i konfigurowalna.

## 3. Augmentacja wektorowa dla uczenia embedderów (darmowa superwizja)

**Prymitywy:** P1, P2, P3.

**Teza.** Augmentacja kontrastowa tekstu (back-translation, TF-IDF-dropout)
**przestawia normy i rejestr**, czyli psuje geometrię, potem mierzycą cosine.
Tutaj augmentacją jest `T_r(v)` — z konstrukcji **izometria do skali** (P2),
więc pary pozytywne mają identyczną strukturę geometryczną, a trudne negatywy
to `T_ant(v)` i łańcuchy nieprzemienne — czyli negatywy **znaczeniowo**
trudne, a nie losowe.

**Jak.** InfoNCE, w której pozytywy = orbita `syn`/`hyper(λ)`, trudne negatywy
= orbita `ant` + `T_r∘T_s(v)` dla par o `C_rs < 0.1`. Zero par tekstowych —
wystarczy graf i istniejące embeddingi. Wariant drugiego użycia: **regularizer
na istniejący embedder** — kara za niezgodność holonomii przestrzeni z
holonomią grafu („przestrzeń ma być płaska tam, gdzie płaska jest ontologia").

**Co poprawia i metryka.** MRR/nDCG na zadaniach retrieval STS/PL-podobnych,
vs back-translation i vs standardowy contrastive; osobno **stability**:
ISOT i rozkład norm po augmentacji (tu baseline'y wypadają źle z definicji).
Szczególnie wartościowe dla języków o małej zasobności (polski), gdzie pary
paretyczne są drogie.

**Fazy.** (1) sampler par z operatora (wektory, bez dekodera), 2 dni;
(2) run InfoNCE na małym embedderze, 3 dni; (3) porównanie + ablacja negatywów
(losowe vs relacyjne), 1 tydzień.

**Ryzyka.** Operator jest tak dobry, jak jego `PCE(2)` — złe łańcuchy dają
złe negatywy; mitygacja: dopuścić tylko relacje z `PCE(1)` i `PCE(2)`
poniżej progu (mamy te metryki per relacja).

## 4. Niepewność bez logprobów: entropia orbitowa (typed semantic entropy)

**Prymitywy:** P4, P5, P7.

**Teza.** Semantic entropy (Farquhar et al., Nature 2024) i jej warianty
(`Beyond Semantic Entropy: … Pairwise Semantic Similarity`, 2025) grupują
k odpowiedzi **po progu podobieństwa / przez NLI** — czyli binarnie: „to samo
czy nie". Grupowanie **typowane relacją** jest mocniejsze: pięć odpowiedzi w
jednej orbicie `syn` = pewność; dwie w orbitach połączonych `ant` = realny
konflikt; `applicability` niskie dla wszystkiego = „model gada poza językiem
opisu" (odrębny stan, a nie wysoka entropia).

**Jak.** Dla k próbek: graf znakowany (p. 1) → podział na orbity → entropia
po orbitach; dodatkowo wektor objaśnienia: „odmawiam, bo próbki 2 i 5 są w
relacji `ant` z conf 0.83" (kalibrowanej przez `ECE`, nie wymyślonej).
Działa na modelach bez dostępu do logitsów (API).

**Co poprawia i metryka.** AUROC poprawności / risk-coverage przy abstain,
vs semantic entropy i SelfCheckGPT; druga metryka: **konflikt-recall** (czy
abstain włączany jest właśnie tam, gdzie odpowiedzi są sprzeczne, a nie tylko
różne słowami) — tej metryki obecne metody nie optymalizują, a my mamy na nią
naturalne goldy z grafu.

**Fazy.** (1) prototyp na 3 benchmarkach QA (Truete-ish, HaluEval-PL
własny), 4 dni; (2) baseline'y (sem-ent, selfcheck, logprob-ent), 3 dni;
(3) kalibracja progu abstain + raport, 2 dni.

**Ryzyka.** zależność od F4 (dekoder działa na wektorach zdań — patrz Uwagi);
przy k=10 próbek koszt `decode` to ~45 par — trywialny.

## 5. Holonomia jako probe na uboczne skutki edycji wiedzy (ripple effects)

**Prymitywy:** P7, P4, P6.

**Teza.** Edycja wiedzy w LLM (ROME/MEMIT) psuje sąsiednie fakty — znane
i nierozwiązane (*Why Does New Knowledge Create Messy Ripple Effects in
LLMs?*, 2024; *Efficiently Quantifying and Mitigating Ripple Effects*, 2024;
*CLaRE-ty Amid Chaos*, 2026). Obecne metody wykrywają to **generując** setki
probe'y. Proponujemy probe **geometryczny, bez generowania**: poprawna edycja
musi zachować lokalną płaskość połączenia wokół redagowanego węzła.

**Jak.** Przed i po edycji: dla promienia 2 wokół encji — `ΔHD` (holonomia na
czwórkach), `Δapplicability`, `Δconf` dekodera dla relacji spoza edytowanej
ścieżki. Sygnał: **duży ΔHD przy małej zmianie samej krawędzi** = edycja
przebudowała okolicę, czyli „kolateral". To jest dokładnie to, czego nie widać
we fluency/edit-success.

**Co poprawia i metryka.** AUC wykrycia „ta edycja zepsuje sąsiadów" zanim
uruchomi się kosztowną ewaluację; Spearman(ΔHD, rzeczywista degradacja sąsiadów);
koszt: ułamek baseline'u (brak generowania).

**Fazy.** (1) wstrzykiwanie edycji w mały model open-weight + liczenie ΔHD,
1 tydzień; (2) korelacja z oficjalnym ripple-benchmarkem, 3 dni; (3) bramka
„edit CI" na ΔHD, 2 dni.

**Ryzyka.** wymaga mapowania „stan modelu → przestrzeń wektorów" (joint
training lub dopasowanie Procrustesem do przestrzeni Słowosieci) — to
największa niewiadoma w tym pomyśle i trzeba ją zbadać w kroku 1a (1 dzień,
decyzja go/no-go).

## 6. Odwracalna pseudonimizacja semantyczna (prywatność z kluczem i pokrętłem)

**Prymitywy:** P1, P3, P5.

**Teza.** Wektory embedera wyciekają treść: ataki invertujące odtwarzają
całe zdania (*Sentence Embedding Leaks More Information than You Expect*,
2023; wersja wielojęzyczna 2024; diffusion 2026; nawet dlaembeddingów
zaszumionych — *Denoising-Aware Inversion*, 2026). Zgadywanie „pomasz
wektor" (szum, kwantyzacja) jest łamane właśnie tymi atakami. Proponujemy
**pseudonimizację relacyjną**: publikujemy `T_r(λ)v`, gdzie `r` i `λ` są
kluczem; `T⁻¹` (dokładne, P1) daje pełne odszyfrowanie uprawnionemu.

**Jak.** Wybór relacji nie jest obojętny: `hyper` naturalnie
„uogólnia tożsamość" (Kowalski → menedżer → osoba) = semantyczny
odpowiednik k-anonymity; `λ` steruje siłą; `applicability(v, hyper)` mówi,
czy da się jeszcze uogólniać (brak dalszego stopnia = ostrzeżenie).
Protokół doboru `λ`: dla siatki `λ` mierzyć (a) odzysk w ataku invertującym
(BLEU/ROUGE vs oryginał), (b) AUC ataku linkage/membership-inference,
(c) użyteczność (jakość zadania docelowego). Publikujemy **krzywą
prywatność-użyteczność** i minimalne `λ`, przy którym atak przegrywa.

**Co poprawia i metryka.** Bezpieczne wysyłanie embeddingów do strony
trzeciej / embedding-as-a-service / federacyjne bazy wektorowe. Metryki:
recovery-rate ataku, k-anonymity osiągnięte w sensie semantycznym, spadek
użyteczności (ΔMRR). Baseline: szum gaussian, kwantyzacja, losowna
rotacja (bez klucza = bez odszyfrowania), „brak ochrony".

**Fazy.** (1) replikacja ataku invertującego na naszej przestrzeni, 3 dni;
(2) siatka `λ` × relacje + krzywe, 4 dni; (3) porównanie z baseline'ami
ochrony i opis zagrożenia modelowego (kto zna `T_r`?), 3 dni.

**Ryzyka.** **najnowszy pomysł w tym zestawie i najmniej zbadany** — realne
pytanie, czy `T⁻¹` trzymane jako klucz wystarczy, gdy atakujący ma dostęp do
wielu wektorów (struktura orbity może pomagać w odwracaniu). Warto zrobić
najpierw jako eksperyment badawczy (1 tydzień), nie jako produkt.

## 7. CI-gate dla modeli: relacyjny odcisk palca (regresja zachowania)

**Prymitywy:** P6, P4, P5, P3.

**Teza.** Zmiana embeddera / kwantyzacja / kolejna wersja fine-tune potrafi
po cichu zmienić *znaczenie* przestrzeni, nie tylko jej metryki. Współrzędne
`θ_r`, macierz `C_rs` i rozkład `applicability` tworzą **odcisk semantyczny**:
wektor delta per relacja, z progami. To użycie modelu jako **przyrządu
pomiarowego**, nie predyktora — najrzadszy i najbardziej niedoceniany tryb.

**Jak.** Zestaw probe'ów (pary o znanej relacji + pary kontrolne). Po każdej
zmianie: `decode` conf per relacja, θ (jeśli przestrzeń jest przeliczana),
`C_rs` (jeśli uczona na nowej przestrzeni), rozkład `applicability`. Progi
per relacja → alarm „ta wersja zmieniła traktowanie negacji". Wykrywa też
klasyczną awarię operacyjną: indeks zbudowanym starym embedderem, zapytania
nowym (osobny sygnał: skok `OMD`).

**Co poprawia i metryka.** AUC wykrycia wstrzykniętych regresji (zamiana
embeddera, checkpoint skwantyzowany, agresywny fine-tune) **przy równym
false-alarm rate** vs proste miary dryfu (MMD/Procrustes między zbiorami
wektorów, shift średnich) — te ostatnie nie widzą zmian *struktury* relacji.
Czas detekcji: minuty, w CI.

**Fazy.** (1) definicja zestawu probe + baseline dryfu, 3 dni; (2) wstrzykiwanie
10 typów regresji, 3 dni; (3) bramka CI + dashboard, 2 dni.

**Ryzyka.** niskie; największa wartość przy organizacji z wieloma
indeksami/embedderami (u nas: wiele wersji Słowosieci i embedderów).

## 8. Wyrównanie dwóch przestrzeni bez słownika — przez sygnaturę algebry

**Prymitywy:** P6, P4, P1.

**Teza.** Standardowo dwie przestrzenie (dwa języki, dwa embedery, dwie
ontologie) wyrównuje się Procrustesem na **ziarnie słownikowym**, albo przez
mapowanie Sinkhornem — oba wymagają, by przestrzenie miały tę samą
„kształt" ogólną. Proponujemy **dopasowanie po zachowaniu transportu**: pojęcie
A w przestrzeni I odpowiada B w przestrzeni II, jeśli ich *profile relacji*
są zgodne: `{(r, cos(T_r a, a'))}` ↔ `{(r, cos(T_r b, b'))}` oraz rząd wiersza
`C_rs`. Zmienniczą jest **algebra, nie współrzędne** — więc nie trzeba wspólnej
przestrzeni wyjściowej ani ziarna.

**Co poprawia.** Transfer międzyjęzyczny dla języków bez słownika
porównawczego; scalenie dwóch ontologii / migracja ze starego słownika do
nowego (mapowanie hierarchii + wskazanie miejsc, gdzie `C_rs` się rozjeżdża
= różne traktowanie tych samych pojęć, czyli realny konflikt modeli
dziedzinowych).

**Metryki.** top-k accuracy dopasowania vs VecMap/GlotLid-style baseline'y
przy zerowym gradiencie startowym; precyzja mapowania onotlogii vs podejścia
oparte o dopasowanie nazw; dodatkowo: frakcja par oznaczonych jako „nie daje się
wyrównać spójnie" (to wartościowy produkt uboczny — wykrywa *naprawdę* różne
koncepcje).

**Fazy.** (1) dwa treningi operatora na dwóch przestrzeniach (np. PL/EN
WordNet), 3 dni; (2) matcher sygnatur + ewaluacja na trzymanym ziarnie,
1 tydzień; (3) zastosowanie do migracji słownika wew. organizacji, 1 tydzień.

**Ryzyka.** wymaga, by obie przestrzenie miały sensowny własny operator
(czyli własny graf) — bez drugiego grafu pomysł nie działa.

## 9. Surowce psycholingwistyczne z pokrętła: abstrakcyjność, wieloznaczność, typowość

**Prymitywy:** P3, P5, P7.

**Teza.** Trzy wielkości, których psycholingwistyka szuka ręcznie (normy
abstrakcyjności są małe, kosztowne i językowo-zależne):
- **abstrakcyjność** = ile kroków `hyper` do nasycenia (ile transportów,
  zanim `T` przestanie zmieniać najbliższe sąsiedztwo / `applicability`
  spadnie);
- **wieloznaczność** = rozrzut `applicability` po relacjach i lokalna
  krzywizna (wieloznaczne słowo ma „wiele ram" naraz → transport zależy od
  drogi, czyli `HD` rośnie);
- **typowość** = `cos` do środka orbity swojej kategorii.

Semantycznie to nie jest heurystyka: `hyper` *definiuje* porządek
konkret→abstrakt, a `λ` daje skalę ciągłą, a nie ranking.

**Co poprawia.** Sterowanie poziomem tekstu (readability/accessibility —
„przepisz na poziom 12-latka" z mierzalnym celem), upraszczanie tekstów
urzędowych i medycznych; przesiew w badaniach klinicznych (abstrakcyjność
jest markerem m.in. w badaniach nad schizofrenią i afazją — normy są dziś
ręczne i dla polszczyzny skąpe); ocena stylu odpowiedzi LLM.

**Metryki.** Spearman vs ludzkie normy (cel: nie gorszy niż normy między
podzbiorami — czyli internal consistency); downstream: dokładność klasyfikatora
poziomu tekstu i ewaluacja ludzka streszczeń z celem „obniż abstrakcyjność
o 0.3".

**Fazy.** (1) definicje operacyjne + skrypt liczący na całym słowniku, 3 dni;
(2) walidacja na istniejących normach PL/EN (lub 300 słów własnych etykiet),
1 tydzień; (3) zastosowanie: sterownik czytelności w pipeline'ie LLM, 3 dni.

**Ryzyka.** operacjonalizacja „nasycenia" wymaga decyzji (próg vs punkt
pragięcia krzywej) — ustawić na gold-zbiorze, nie na oko.

## 10. Dryf semantyczny bez wyrównywania osi czasu (współdzielona baza generatorów)

**Prymitywy:** P6, P4 (θ), P3.

**Teza.** Klasyczny problem diachronicznych embeddingów: żeby porównać
przestrzenie z dwóch okresów, trzeba je do siebie **dopasować** (Procrustes na
słownikach kotwiczących) — i cała wnioskowanie o dryfie wisi na jakości tego
dopasowania (*Diachronic word embeddings and semantic shifts: a survey*, 2018;
warianty aplikaacyjne 2025). Tutaj **wszystkie slice'y dzielą tę samą bazę
generatorów `B`**, więc `θ_r(t)` jest porównywalne **bez żadnego
dopasowywania osi**.

**Jak.** Dla każdego okna czasowego: przećwiczyć operator na grafie z tego
okna (albo tylko `θ` przy zamrożonym `B` — tanio). Delta `θ_r(t₂)-θ_r(t₁)`
= **dryf przypisany relacji**, w interpretabilnych jednostkach (kąt w
płaszczyźnie generatora). Zmiana `C_rs(t)` = zmiana *porządku świata*: które
osy przestały być niezależne (np. relacja „praca" i „etat" przestają
komutować z relacją „zdobywanie" → przeformułowanie dyskursu).

**Co poprawia.** Leksykografia (monitoring znaczeń i neologizmów), wykrywanie
zawłaszczania terminów (brand/meaning hijack), analiza dyskursu
publicznego z **przypisaniem winy do relacji**, a nie do „chmury słów".

**Metryki.** korelacja z adnotacjami leksykografów o przesunięciach
znaczeń; AUC wykrycia suchego wstrzykniętego przesunięcia (symulacja:
przebudowanie krawędzi w oknie t₂); porównanie z Procrustes-based baseline:
przy tym samym AUC — ile kotwic trzeba (u nas: zero).

**Fazy.** (1) trening z zamrożonym `B` per slice (dopisać opcję w
trenerze), 3 dni; (2) korpus 3–5 okien (np. 2010–2025 z Słowosieci +
nowe słowa), 1 tydzień; (3) adnotacja + ewaluacja, 1 tydzień.

**Ryzyka.** dryf kumulacyjny w samych treningach (różne seed'y) — mitygacja:
ten sam seed, zamrożony `B`, tylko `θ` i `C_rs` wolne; sprawdzić `ISOT`
między slice'ami.

## 11. Synteza pojęcia z ograniczeniami relacyjnymi (QP zamiast szukania)

**Prymitywy:** P1, P3, P5.

**Teza.** Wyszukiwanie cosine odpowiada na „co jest podobne do q". Nie
odpowiada na „co jest **przeciwieństwem X w rejestrze Y**" ani — co ważniejsze
— na „co jest podobne do A, ale **NIE** jest podobne do B". Operatory są
liniowe i odwracalne, więc to ostatnie jest **układem ograniczeń**:

    maks   cos(T_r v, a)
    war.   cos(T_s v, b) ≤ τ,   ‖v‖ = 1

dwa–trzy ograniczenia → rozwiązanie zamknięte przez mały problem własny
(2–3 wymiary). Wynik: **wektor pojęcia, które spełnia relacyjne wymagania**,
czegokolwiek by nie było w indeksie — czyli kreacja, nie odnalezienie.

**Co poprawia.** Naming, burza mózgów, projektowanie (wariant „analogiczny do
X, ale bez skojarzenia z Y"), **negative requirements** w generowaniu (filtr
na etap wektora zamiast post-filtera tekstowego), a w wariantie
medyczno-chemicznym na własnym grafie: analog leku spełniający relacyjne
„nie-ograniczenia".

**Metryki.** preferencja ludzka vs „przepisz zapytanie + filtr" (realny
alternatywny baseline, który kosztuje LLM-call i zazwyczaj nie trafia);
pokrycie ograniczeń (ile rozwiązań spełnia `≤ τ` przy zachowaniu celu).

**Fazy.** (1) solver + wizualizacja na 2 wymiarach (sanity check), 2 dni;
(2) integracja z indeksem (rozwiązanie jako zapytanie), 2 dni; (3) test A/B
preferencji na 50 zadaniach naming/design, 4 dni.

**Ryzyka.** rozwiązanie może leżeć poza częścią wspólną ograniczeń — wtedy
retriever zwraca szum; mitygacja: sprawdzić `OMD` i `applicability` dla
rozwiązania przed wysłaniem do indeksu.

## 12. Samosprzeczność tekstu i wrażliwość na kolejność rozumowania

**Prymitywy:** P7, P6, P4.

Dwa małe, tanie, a niedoceniane zastosowania diagnostyczne:

**(a) Detektor „mowy-zegara"** — dokument i jego własna negacja: jeśli
`cos(v_doc, T_ant(v_claim))` jest niepokojąco wysokie, dokument jest wewnętrznie
sprzeczny (spin, greenwashing, „wyjątek od wyjątku" w umowie). Metryka: AUC
na korpusach z adnotacją sprzeczności/greenwashingu vs baseline NLI (który
liczy pary zdan, a nie spójność całego dokumentu i jest ~100× droższy).

**(b) Test nieprzemienności rozumowania** — pary relacji, dla których `C_rs ≈ 0`
są **goldem**, że kolejność zmienia znaczenie. Zadanie dla LLM/embeddera:
„antonim synonimu X vs synonim antonimu X — co to jest?". Metryka:
wrażliwość na kolejność jako funkcja rozmiaru modelu i po fine-tuning'u.
To darmowy, algebraicznie uzasadniony benchmark kompetycji
składalnościowych — i jednocześnie lokalizator miejsc, w których modele
mylą się w rozumowaniu wielokrokowym (patrz też `LLM_USE_CASES.md` §5.2).

**Fazy.** (a) 3 dni, (b) 3 dni; oba nie wymagają niczego poza `apply`,
`decode` i `commutation_C.csv`, więc mogą być **pierwszym eksperymentem**
weryfikującym, czy operator jest w ogóle wystarczająco dobry.

---

## Priorytety (wartość / wysiłek / pewność)

| pomysł | wysiłek | wartość | ryzyko naukowe |
|---|---|---|---|
| 12 (diagnostyka a/b) | 1 tyg. | średnia, ale kalibruje resztę | bardzo małe |
| 1 (klasteryzacja relacyjna) | 3 tyg. | wysoka, udokumentowana wada baseline'ów | małe |
| 2 (dedup typowany) | 2–3 tyg. | wysoka (dane treningowe) | małe |
| 4 (entropia orbitowa) | 2 tyg. | wysoka (abstain to produkt) | średnie (F4 na zdaniach) |
| 7 (CI-gate / odcisk) | 1,5 tyg. | wysoka operacyjnie | bardzo małe |
| 3 (augmentacja wektorowa) | 2 tyg. | wysoka (embedery PL) | średnie (zależy PCE) |
| 9 (abstrakcyjność z `λ`) | 2 tyg. | wysoka naukowa, niska koszt | małe |
| 10 (dryf bez dopasowania) | 3 tyg. | wysoka naukowa | średnie |
| 11 (synteza z ograniczeniami) | 1,5 tyg. | średnia–wysoka, efektowna | małe |
| 6 (pseudonimizacja `T⁻¹`) | 1 tyg. badawczo | potencjalnie duża | **duże** — zrobić jako first probe |
| 5 (ripple via holonomia) | 3 tyg. | wysoka (neresolved problem) | duże (mapowanie stanu modelu) |
| 8 (wyrównanie bez słownika) | 3 tyg. | wysoka, ale wymaga 2. grafu | średnie |

**Kolejność, którą bym przyjął:** 12 → 1 → 2 → 7 → 4 → 9 → 3 → 11 → 10 → 6
(probe) → 5 → 8.

## Uwagi wspólne (warunki wstępne, których nie da się obejść)

- **Domena wektorów.** Operator jest szkolony na **jednostkach leksykalnych**.
  Pomysły 1, 2, 4, 11, 12(a) działają na **zdaniach/dokumentach** — wymagają
 wymaga przećwiczenia F4 z `shadow vectors` = wektorami zdań z docelowego
  indeksu (to dokładnie mechanizm „shadow manifold", nie nowa praca
  architektoniczna). Przed jakąkolwiek budową: zmierzyć `OMD` dla wektorów
  zdań — jeśli jest niskie, pomysły „zdaniowe" odrzucić albo przećwiczyć F4.
- **Zależność od F4.** `decode` i `applicability` powstają w etapie F4.
  Pomysły 5–10 (bez dekodera/aplicability) da się robić na samym operatorze,
  nawet jeśli F4 nie doszedł.
- **Metryka decyduje.** W każdym z powyższych punktem wyjścia jest ten sam
  benchmark: `PCE(k)`, `ISOT`, `OMD`, QR-Recall (`core/metrics/`) — bez
  niego żaden z pomysłów nie jest falsyfikowalny.
- **Czego tu nie ma, a było w `LLM_APPLICATION_IDEAS.md`:** steered
  retrieval, agent przepisywania zapytań, weryfikator twierdzeń, eksplorator,
  cache — czyli wszystko, co „poprawia wyszukiwanie". Tu jest to, co poprawia
  **klasteryzacja, dane treningowe, niepewność, prywatność, CI, pomiary
  znaczenia i diagnostykę modeli**.

## Kotwice literaturowe (sprawdzone w arXiv, październik 2026)

- SemDeDup (2023), FairDeDup (2024) — dedup semantyczny i jego wady.
- *Is Cosine-Similarity of Embeddings Really About Similarity?*; *Problems with
  Cosine as a Measure of Embedding Similarity*; *Intuitive Contrasting Map for
  Antonym Embeddings*; *Cosine similarity of antonym embeddings trained with
  different contexts* — antonimy blisko siebie w cosine (pomysł 1, 2, 12).
- Semantic entropy (Farquhar et al., Nature 2024); *Beyond Semantic Entropy:
  Boosting LLM Uncertainty Quantification with Pairwise Semantic Similarity*
  (2025); *FASE* (2026) — entropia semantyczna bez typów relacji (pomysł 4).
- *Why Does New Knowledge Create Messy Ripple Effects in LMs?* (2024);
  *Efficiently Quantifying and Mitigating Ripple Effects in Model Editing*
  (2024); *CLaRE-ty Amid Chaos* (2026) — nierozwiązany problem ubocznych
  skutków edycji (pomysł 5).
- *Sentence Embedding Leaks More Information than You Expect* (2023);
  *Against All Odds: … Multilingual Embedding Inversion Attacks* (2024);
  *Embedding Inversion via Conditional Masked Diffusion LMs* (2026);
  *Denoising-Aware Inversion* (2026) — inwersja embeddingów, w tym
  zaszumionych (pomysł 6).
- *Diachronic word embeddings and semantic shifts: a survey* (2018);
  *Characterizing Linguistic Shifts in Croatian News via Diachronic Word
  Embeddings* (2025) — problem dopasowywania osi (pomysł 10).
- *Verification of Machine Unlearning is Fragile* (2024); *Towards Reliable
  Forgetting: A Survey on Machine Unlearning Verification* (2025) — sąsiedni
  problem: weryfikacja „co model już nie wie" (naturalne rozszerzenie 12a).
- *Kernelized Activation Steering* (2026); *Depth-Wise Activation Steering for
  Honest Language Models* (2025) — steering jako dodawanie wektora: brak
  odwracalności i brak struktury (kontrast dla P1+P3).
