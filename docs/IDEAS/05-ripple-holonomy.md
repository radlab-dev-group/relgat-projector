# 5. Holonomia jako probe na uboczne skutki edycji wiedzy (ripple effects)

**Teza w jednym zdaniu:** poprawna edycja wiedzy w modelu musi zachować
**lokalną płaskość** połączenia wokół redagowanej encji — a `ΔHD` (zmiana
defektu holonomii) wykrywa szkody uboczne **bez generowania setek probe'ów**,
czyli za ułamek kosztu obecnych metod.

**Prymitywy:** P7 holonomia `HD` · P4 `decode` · P5 `applicability` · P6 `C_rs`
**Wysiłek:** 3 tygodnie · **Ryzyko naukowe:** duże (patrz §7) — zrobić
najpierw jednodniowy test mapowania przestrzeni
**Status:** problem nierozwiązany w literaturze; tu jest największa możliwa
wartość w całym zestawie, ale i największa niepewność

---

## 1. Problem

Edycja wiedzy w zamrożonym modelu (ROME, MEMIT i następcy) zachowuje się
dobrze na edytowanym fakcie i **źle w sąsiedztwie**: po wbiciu „Polemarch
napisał Traktat o architekturze" model potrafi zgubić, że Polemarch był
kupcem, albo zacząć przypisywać mu inne dzieła. To dobrze udokumentowane:

- *Why Does New Knowledge Create Messy Ripple Effects in LLMs?* (2024) —
  opis zjawiska i analiza przyczyn;
- *Efficiently Quantifying and Mitigating Ripple Effects in Model Editing*
  (2024) — kwantyfikacja przez **generowanie**;
- *CLaRE-ty Amid Chaos: Quantifying Representational Entanglement to Predict
  Ripple Effects in LLM Editing* (2026) — próba predykcji przez splątanie
  reprezentacji.

Wspólna słabość wszystkich: koszt. Żeby zmierzyć szkody uboczne, trzeba
**wygenerować** tysiące promptów kontrolnych (wszyscy sąsiedzi edytowanej encji
× wszystkie relacje). Dla regularnego CI na modelach to bariera nie do
przejścia, dlatego w praktyce nikt tego nie mierzy — i dlatego problem trwa.

Druga słabość: metryki edycji (edit success, fluency, generalization) są
**lokalne wokół edytowanego faktu**. Szukają odpowiedzi „czy nowa informacja
weszła", a nie „czy struktura dookoła się nie zapadła".

## 2. Mechanizm i formalizacja

Traktujemy wiedzę nie jako zbiór faktów, tylko jako **połączenie** (connection):
wektory mają sens lokalny, a relacje transportują znaczenie. Edycja faktu to
**lokalna zmiana transportu**. Dobra edycja zmienia jeden kierunek; zła edycja
zmienia transport w całym otoczeniu, czyli **krzywiznę**.

Mierzalne: dla czwórki (diamentu) `A -r-> C -q-> D <-s- B <-p- A` holonomia

    H = T_p⁻¹ ∘ T_s⁻¹ ∘ T_q ∘ T_r

powinna spełniać `H(x_A) ≈ x_A`, jeśli dwie drogi między A i D opisują ten sam
przejście znaczeniowe. Defekt `HD = ‖H(x_A) − x_A‖` jest więc **miarą spójności
lokalnej, niezależną od odpowiedzi modelu**.

Probe: dla promienia 2 wokół encji `e` objętej edycją liczymy

    ΔHD(e)   = HD_po(e)  − HD_przed(e)
    Δappl(e) = |a_po(v, r) − a_przed(v, r)|  uśrednione po relacjach otoczenia
    Δconf(e) = zmiana ufności dekodera dla relacji SPOZA ścieżki edycji

Sygnał diagnozy: **duży `ΔHD` przy małej zmianie samej edytowanej krawędzi**.
Czyli: fakt wszedł (edytowana krawędź się poprawiła), ale okolica się
rozjechała. To dokładnie ten stan, którego nie widać w edit-success ani w
fluency — bo obie mierzą tylko edytowany fakt.

## 3. Algorytm

Kluczowa część to **mapowanie stanu modelu na przestrzeń operatora** — bez
niego nie ma pomiaru. Trzy warianty, rosnącej trudności:

*Wariant A (najprostszy, do testu go/no-go).* Nie ruszamy stanu wewnętrznego
modelu, tylko **przestrzeń wyjściową**: edytujemy model, potem dla ustalonego
zestawu probe'ów (zdania opisujące fakty z otoczenia encji) liczymy embeddingi
**przed i po** edycji, tym samym embederem. Wtedy `HD` liczymy na embeddingach
zadań, a transport `T_r` bierzemy z wytrenowanego operatora. `ΔHD` mierzy, czy
edycja nie popsuła **geometrii opisów**, co jest pośrednim, ale twardym
sygnałem.

*Wariant B (właściwy).* Dopełnienie Procrustesem warstwy ukrytej modelu do
przestrzeni operatora na stałym zestawie probe'ów, przed edycją; ta sama
macierz transformacji stosowana po edycji. `ΔHD` liczymy w przestrzeni
znormalizowanej. To jest to, czego chce badać *CLaRE-ty*, ale przez krzywiznę
zamiast splątania.

*Wariant C (jeśli A i B zawiodą).* Operator trenowany **osobno na
przestrzeni wewnętrznej** modelu docelowego (nowy trening, ten sam graf) i
`ΔHD` liczony między dwoma operatorami wytrenowanymi na dwóch wersjach modelu.
Droższe, ale bez ukrytych założeń.

```python
flow = RelationFlow.load(ckpt_op, device="cuda")

def local_hd(probe_vecs, diamonds, radius=2):
    V = align(probe_vecs)                       # wariant A: tożsamość; B: Procrustes
    return holonomy_defect(flow.model.operator, V, diamonds_in_neighbourhood(radius))

hd_before = local_hd(V_before, D)
edit_model(...)                                  # ROME / MEMIT
hd_after  = local_hd(V_after,  D)

score = hd_after - hd_before                     # im większy, tym więcej kolateralu
```

Zestaw probe'ów: dla encji `e` bierzemy wszystkie (sąsiedzi, relacja) z
promienia 2 w Słowosieci, renderujemy zdania szablonem i embedujemy. Typowo
50–200 probe'ów na encję, **bez generowania LLM** — to jest źródło oszczędności.

## 4. Metryki i protokół

| metryka | definicja | cel |
|---|---|---|
| **AUC kolateralu** | czy `ΔHD` oddziela edycje z dużymi szkodami od edycji czystych | > 0.75 |
| Spearman(`ΔHD`, true ripple) | korelacja rangowa z oficjalnym pomiarem ripple (generowanym) | > 0.5 |
| **koszt vs dokładność** | ile promptów trzeba wygenerować, by dorównać `ΔHD` | nasz koszt: **0** generowań |
| `Δconf` relacji spoza ścieżki | osobna metryka: czy dekoder widzi zmianę znaczeń w otoczeniu | korelacja z `ΔHD` |
| AUC dla „ta edycja zepsuje sąsiadów" w CI | praktyczna bramka | użyteczna > 0.8 |

Protokół ewaluacji: 100 edycji (50 czystych, 50 wstrzykniętych tak, by psuły
sąsiadów — np. edycje w gęstym fragmencie grafu) + oficjalny pomiar ripple jako
gold. Różnica względem literatury: **gold istnieje i jest tani**, bo strukturę
okolicy znamy z grafu, a nie z generowania.

## 5. Fazy

| faza | czas | artefakt | decyzja |
|---|---|---|---|
| 5a | 1 dzień | wariant A: `ΔHD` na embeddingach przed/po dla 10 edycji | **go/no-go**: czy `ΔHD` w ogóle drga? |
| 5b | 3 dni | 100 edycji, korelacja `ΔHD` z ripple generowanym | kontynuować / porzucić |
| 5c | 1 tydzień | wariant B (dopasowanie Procrustesem) + porównanie z wariantem A | wybrać metodę |
| 5d | 1 tydzień | ewaluacja na benchmarku ripple + raport | publikowalny wynik |

## 6. Warunki, w których pomysł **nie** zadziała

- jeśli edycja ROME/MEMIT zmienia warstwę FFN w sposób niemal ortogonalny do
  kierunku, w którym `T_r` obraca — `ΔHD` będzie zerowe, a szkody realne;
- jeśli przestrzeń embedera zdan nie jest w ogóle zgodna z przestrzenią
  operatora (brak wspólnego podprzestrzenia) — wtedy wariant A jest bez
  znaczenia i trzeba iść w wariant C;
- jeśli otoczenie encji w Słowosieci jest zbyt rzadkie (brak czwórek w promieniu
  2) — `HD` nie ma czego mierzyć; sprawdzić licznik dostępnych diamentów **przed**
  jakąkolwiek pracą (`QuadrupleSampler.sample_batch` zwraca `None`).

Ten trzeci punkt jest konkretny i łatwy do sprawdzenia: policz, ile encji ma co
najmniej jeden diament w promieniu 2. Jeśli < 30% encji, pomysł zawęzić do
encji gęstych i tak też go opisać.

## 7. Ryzyka

| ryzyko | ocena | mitygacja |
|---|---|---|
| mapowanie stanów (wariant B) nie jest stabilne | **wysokie** | zacząć od wariantu A; C jako plan awaryjny |
| `ΔHD` reaguje na szum numeryczny, nie na wiedzę | średnie | baseline „ten sam model, dwa przebiegi" — wyznaczyć poziom szumu i mierzyć ponad nim |
| brak czwórek w otoczeniu | średnie, sprawdzalne w godzinę | zawęzić zakres do encji gęstych |
| edytowalny model musi być open-weight | pewne | wymagać: model 1–7B open-weight + implementacja ROME |

## 8. Test go/no-go

10 edycji na małym modelu open-weight + 50 probe'ów na edycję. Policzyć
`ΔHD` dla edycji i dla **10 edycji zerowych** (puste ROME, ta sama procedura).
Jeśli rozkłady `ΔHD` dla edycji i dla edycji zerowych się nie rozdzielają —
sprawdzian jest bezwartościowy i nie warto inwestować dalej. Całość: jeden
dzień, w tym pół dnia na odpalenie modelu.

## 9. Punkty odniesienia

- *Why Does New Knowledge Create Messy Ripple Effects in LLMs?* (2024).
- *Efficiently Quantifying and Mitigating Ripple Effects in Model Editing* (2024).
- *CLaRE-ty Amid Chaos* (2026) — najbliższy konkurent: predykcja przez
  splątanie reprezentacji; nasza różnica to **krzywizna zamiast splątania** i
  zero generowania.
- W repo: `core/metrics/geometry.py` (`holonomy_defect`),
  `core/sampler/quadruples.py` (`QuadrupleSampler`, `Diamond`),
  `core/loss/curvature.py`.
