# 12. Diagnostyka: samosprzeczność tekstu i wrażliwość na kolejność rozumowania

**Teza w jednym zdaniu:** dwa tanie testy bez etykiet — (a) czy tekst jest
bliski własnej negacji (spójność wewnętrzna) i (b) czy model rozumie, że
`ant∘syn ≠ syn∘ant` (nieprzemienność jako gold) — działają na samym `apply`,
`decode` i `commutation_C.csv`, więc mogą być **pierwszym eksperymentem** w
katalogu i od razu mówią, czy operator jest wystarczająco dobry do reszty.

**Prymitywy:** P7 spójność · P6 `C_rs` · P4 `decode` · P1 `apply`/`inverse`
**Wysiłek:** 3 dni (a) + 3 dni (b) · **Zależności:** minimalne
**Status:** **zaimplementowany** (0.4.1) — `relgat_projector/core/diagnostics/` +
`python -m relgat_projector_apps.diagnostics.consistency_report --best`;
dwujęzyczny (PL/EN) fixture w `data/consistency/`

---

## Stan realizacji (0.4.1)

Zaimplementowano oba testy jako moduł niezależny od embeddera: jedynym
punktem styku z modelem jest `Transport` (`apply` + `inverse` + `prepare`),
więc te same dane działają z uczniowskim operatorem (`CheckpointTransport`)
albo z bazowym transportem translacyjnym (`AdditiveTransport`) — ten drugi
jest przemienny z definicji i daje separację kolejności oraz holonomię
równe **dokładnie 0**, więc to on jest podłogą, pod którą uczeń musi wyjść.

Dane testowe (`data/consistency/`, generator
`relgat_projector_apps.diagnostics.make_consistency_fixture`):
13 dokumentów (54 tezy, 13 par sprzeczności *z konstrukcji*, nie z adnotacji)
w PL i EN — spójne, samosprzeczne, greenwashing, wyjątek od wyjątku —
plus 84 zadania na kolejność relacji i leksykon 69+69 par źródłowych dla
szóstki relacji leksykalnych.

Pierwszy pomiar (F4, score 0.3361; `ipipan/silver-retriever-base-v1.1`):

| pomiar | wynik |
|---|---|
| separacja kolejności, uczeń | mediana **0.188**, max 0.440 (baza translacyjna: **0.000**) |
| holonomia pętli `r,s,r⁻¹,s⁻¹`, uczeń | średnia **0.120**, max 0.171 |
| `spearman(sep, C_rs)` na 6 relacjach fixture | **−0.45** (znak dobry) |
| `C_rs` w całym grafie | zakres [0.462, 0.636] — 1540 par, 0 z "mocnym" dowodem |
| 12a AUC (PL): S / K / negacja-surface / gold-pary | 0.762 / 0.643 / **0.857** / 0.795 |
| 12a AUC (EN): S / K / negacja-surface / gold-pary | 0.357 / 0.452 / 0.810 / 0.821 |

Drugi pomiar (run kontrolny na tym samym embedderze — `data/same-embedder/`,
graf re-embedded w `ipipan/silver-retriever-base-v1.1`, tak że operator działa
na tych samych wektorach, w których oceniane są dokumenty).  S1 = przepust
nieskalibrowany (przesunięcie jednostki chartu ×41 zostawiało term L2 w
absolutach); S4 = ten sam przepis z naprawioną jednostką L2, referencją cos
zmierzoną z F4 i własnym lr dla `C_rs`, 30 epok (F4 z kontekstem):

| pomiar | S1 — bez napraw (60 ep) | S4 — po naprawie (30 ep) |
|---|---|---|
| `eval/mrr_lift` | **−0.0065** | **+0.0651** |
| `eval/mrr` / `rel_acc_macro` | 0.038 / 0.382 | **0.157 / 0.447** |
| separacja kolejności, uczeń | mediana 0.326, `C_rs` [0.490, 0.535] | mediana **0.306**, `C_rs` **[0.302, 0.935]** |
| holonomia pętli, uczeń | średnia **0.002** — płaska | średnia **0.0003**, max 0.0010 — **wciąż płaska**, patrz wniosek |
| 12a AUC (PL): S / K (uczeń) | 0.762 / 0.643 (baza) → 0.667 / 0.643 | 0.762 / 0.643 (baza) → 0.667 / 0.452 |
| 12a AUC (EN): S / K (uczeń) | 0.357 / 0.452 (baza) → 0.619 / 0.786 | 0.357 / 0.452 (baza) → **0.667 / 0.571** |

Wnioski (zapisane w `CHANGELOG.md` 0.4.1): (1) operator jest silnie
nieprzemienny — to nie jest Defekt, to cecha, którą DistMult nie da;
(2) `C_rs` wskazuje właściwy kierunek na relacjach, które się liczą, ale
w skali całego grafu ma zakres ~0.05 — nadzór nad `C` w `CurvatureLoss`
faktycznie się nie nauczył, i to jest realny, zmierzony defekt jakości;
(3) tania negacja-surface ("Nieprawdą jest, że …") **przewyższa**
relacyjny metryk `S` na tym fixture — do wyprzedzenia, nie do podbijania
własnych liczb; (4) asymetria PL/EN (0.762 → 0.357) jest miarą tego, jak
dużo metoda zależy od przestrzeni — odpowiedź pomiarowa na pytanie o
niezależność od embeddera.

Wniosek z drugiego pomiaru (naprawa jakości, 0.4.2): run kontrolny S1 pokazał,
że przepis portowany na embedder z normą 23.8 zamiast 0.82 **nie działał** —
`mrr_lift` ujemny, holonomia płaska (0.002), `C_rs` przyklejone do prioru —
i winne były *stałe kalibracyjne* (jednostka L2 ×1660 na termach kwadratowych,
temperatura cos 3× za duża), a nie architektura. Po naprawie (jednostka L2
mierzonego chartu, referencja cos z profilu F4, lr dla `C_rs`) ten sam przepis
na tym samym embedderze daje `mrr_lift` **+0.065** (z −0.007), `rel_acc_macro`
0.447 i `C_rs` rozwarstwione do [0.30, 0.94] — struktura komutacyjna się
uczy. Holonomia pętli zostaje bliska zeru (0.0003): operator pozostaje w
praktyce spłaszczony, a to jest następny, niezależny od skali problem
(brak sygnału krzywizny w gradiencie — patrz `CHANGELOG.md` 0.4.2, "co dalej").

## Punkty odniesienia

## 12a. Samosprzeczność dokumentu („detektor mowy-zegara")

### Problem

Wykrywanie sprzeczności działa dziś na **parach zdań** (NLI): bierzemy dwa
zdania i pytamy, czy sobie przeczą. W dokumentach o setkach zdań liczba par jest
kwadratowa, a większość rzeczywistych manipulacji nie jest sprzecznością
zdanie-zdanie, tylko **rozjechaniem się stanowiska w całym tekście**: „jestesmy
zobowiązani" w jednym akapicie, „z zastrzeżeniem, że..." w innym, „w praktyce
nie dotyczy" w trzecim. Zielone mydło (greenwashing), regulaminy z „wyjątkiem od
wyjątku", deklaracje korporacyjne — wszystko to klasy problemów bez taniej
metryki.

### Mechanizm

Dla dokumentu `d` z tezami `c_1..c_m` (tezy wydobyte przez LLM — jedyny droższy
element, a i tak jeden raz na dokument):

    S(d) = Σ_k cos( T_ant(c_k), d )  / m        — im wyższe, tym „dokument jest
                                                 bliski własnej negacji"

Druga, mocniejsza wersja: **spójność wewnętrzna tez**

    K(d) = median_{i≠j} cos( c_i, T_syn(c_j) ) − median_{i≠j} cos( c_i, T_ant(c_j) )

Dokument spójny ma `K` wyraźnie dodatnie (tezy są wzajemnie bliskie synonimicznie, nie
antonimiczne). Dokument kręcący ma `K ≈ 0` — czyli **jego tezy są równie blisko
bycia sobą, co bycia swoimi przeciwieństwami**.

Trzecia wersja (najmniej kosztowna, bez tez): holonomia `HD` na pętli `syn ∘ ant
∘ syn ∘ ant` wokół wektora dokumentu. Wysoka krzywizna = wektor nie leży w
regionie, w którym te relacje są spójne, czyli tekst stoi „pomiędzy" znaczeniami.

### Metryki i baseline

| metryka | gold | baseline |
|---|---|---|
| AUC greenwashing | korpusy z adnotacją (np. deklaracje klimatyczne firm) | NLI para-zdaniowe (100× droższe), fine-tuned klasyfikator |
| AUC „wyjątek od wyjątku" | zbiór fragmentów regulaminów oznaczonych przez prawnika | heurystyka negacji + cosine |
| koszt na dokument | — | nasze: 1 wywołanie LLM (tezy) + kilka `matmul` |
| korelacja `K(d)` z oceną ludzką „tekst kręci" | 100 dokumentów | — |

## 12b. Test nieprzemienności rozumowania

### Problem

Nieprzemienność jest **wbudowana** w nasz operator (generatory `B` rozpinają
algebrę nieprzemienną, a DistMult nie może jej reprezentować — patrz
`docs/GEOMETRY.md`, tabela porównawcza). `C_rs` mówi, **które pary relacji są
nieprzemienne z dowodem z grafu**. To daje darmowy, algebricznie uzasadniony
zestaw testowy dla modeli językowych:

> Dla pary o `C_rs[r,s] < 0.1`: `T_r T_s(x)` i `T_s T_r(x)` to **różne** punkty.
> Które z nich model wskaże, gdy poprosimy o „`r` i potem `s`"?

### Protokół

1. Z `commutation_C.csv` bierzemy pary: **spójne** (`C > 0.9`) i
   **nieprzemienne** (`C < 0.1`) — to gold, bez adnotacji.
2. Dla każdej pary i każdego `x` z próbki: dwa wektory-docelowe
   `u = compile([r,s])·x` i `w = compile([s,r])·x`.
3. Pytanie do modelu (tekstowo albo przez embedding odpowiedzi): „weź `x`,
   zastosuj `r`, potem `s`". Odpowiedź modelu embedujemy i liczymy, czy bliżej
   jej do `u` czy do `w`.
4. Wynik: **wrażliwość na kolejność** = jak często model trafia przy parach
   nieprzemiennych, w porównaniu z parami przemiennymi (tam kolejność nie ma
   znaczenia i model nie może się pomylić „przez kolejność").

### Metryki

| metryka | definicja | oczekiwanie |
|---|---|---|
| **order-sensitivity gap** | acc(pary nieprzemienne) − acc(pary przemiennie) | modele językowe: gap ujemny lub zerowy — czyli nie rozumieją, że kolejność zmienia wynik |
| zależność od rozmiaru modelu | gap vs liczba parametrów | hipoteza: gap rośnie z rozmiarem — testowalne i ciekawe |
| zależność od długości łańcucha | gap dla `k = 2, 3` | maleje z `k` — lokalizator miejsca, gdzie rozumowanie się rozsypuje |
| zgodność z `C_rs` | korelacja między „model uważa, że kolejność ma znaczenie" a `1 − C_rs` | dowód, że `C_rs` mierzy coś realnego dla rozumowania |

### Dlaczego to ma wartość

To **jedyny** w tym zestawie pomysł, który produkuje wiedzę o modelach językowych
bez etykiet i bez kosztownych benchmarków — i jednocześnie jest testem samego
operatora: jeśli `C_rs` nie koreluje z tym, co robią modele, to albo `C_rs` jest
szumem, albo modele są ślepe na kolejność (drugie jest bardziej prawdopodobne i
samo w sobie warte opisania).

## Fazy (wspólne)

| faza | czas | artefakt |
|---|---|---|
| 12a-1 | 2 dni | `S(d)`, `K(d)` na 500 dokumentach + histogram (sanity) |
| 12a-2 | 3 dni | korpusek gold (greenwashing / regulaminy) + AUC vs baseline |
| 12b-1 | 2 dni | generator par z `C_rs` + zadania dla modelu |
| 12b-2 | 2 dni | pomiar na 3–5 modelach (w tym małe open-weight) + krzywa rozmiar |

## Ryzyka

| ryzyko | mitygacja |
|---|---|
| wydobywanie tez dokumentu jest zawodne | test wariantu bez tez (`HD` na wektorze dokumentu) — porównanie pokaże, ile wnosi dekompozycja |
| `ant` na zdaniach ≠ „negacja tezy" | walidacja na 100 parach ręcznie; ewentualnie osobna relacja „negacja" uczona w F4 |
| `C_rs` ma dużo wartości ~0.5 (brak świadków w grafie) | test dotyczy tylko skrajnych wartości; `0.5` raportować jako „brak dowodu" (`docs/GEOMETRY.md` §4) |
| odpowiedź modelu trudno porównać z `u`/`w` | alternatywny protokół: model wskazuje między **tekstowymi** opisami obu wyników, bez porównywania wektorów |

## Test go/no-go (najtańszy w całym zbiorze, 3 dni)

Weź 30 par relacji o `C_rs` skrajnych, `k = 2`, `x` z 50 encjami, i zapytaj 3
modele (w tym najmniejszy open-weight). Sprawdź, czy `order-sensitivity gap`
**w ogóle różni się między parami przemiennymi i nieprzemiennymi**. Wynik
ujemny też jest publikowalny („modele nie rozróżniają kolejności relacji
semantycznych"), a przy okazji kalibruje wszystkie pozostałe pomysły.

## Punkty odniesienia

- `docs/GEOMETRY.md` §4 — odczyt `C_rs`, w tym przypadek `D4` (brak świadków →
  `C` pozostaje 0.5 zamiast zmyślać strukturę).
- NLI i detekcja sprzeczności — baseline 12a; różnica: para zdań vs cały
  dokument, koszt O(m²) vs O(m).
- Badania nad składalnością w modelach językowych — 12b jest testem
  konkretnego, algebrycznie zdefiniowanego aspektu składalności.
- W repo: `core/diagnostics/consistency.py` (implementacja 12a/12b),
  `core/operator/compiled.py` (`compile` → jeden `matmul`),
  `handlers/wiazka_checkpoint.py` (`load_commutation`),
  `core/metrics/geometry.py` (`holonomy_defect` na kwadratach grafu).
