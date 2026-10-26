# Szczegółowe specyfikacje pomysłów nieoczywistych (1–12)

Rozwinięcie zbiorcze: `NON_OBVIOUS_IDEAS.md` (tam jest krótki opis każdego
pomysłu i tabela priorytetów). Ten katalog zawiera po jednym pełnym
opisie na pomysł — z formalizacją, algorytmem na **realnym API tego repo**,
protokołem pomiaru, baseline'ami, ryzykami i **testem go/no-go**.

Struktura każdego pliku jest jednakowa, więc można czytać wyrywkowo:
*Problem → Mechanizm → Algorytm → Metryki → Fazy → Ryzyka → Test go/no-go →
Punkty odniesienia*.

## Spis

| # | pomysł | plik | wysiłek | ryzyko |
|---|---|---|---|---|
| 1 | klasteryzacja relacyjna (orbity zamiast chmur) | `01-relational-clustering.md` | 3 tyg. | małe |
| 2 | dedup danych treningowych bez zjadania zaprzeczeń | `02-relation-typed-dedup.md` | 2–3 tyg. | małe |
| 3 | augmentacja wektorowa embederów z grafu | `03-vector-augmentation.md` | 2 tyg. | średnie |
| 4 | entropia orbitowa (niepewność bez logprobów) | `04-orbit-entropy.md` | 2 tyg. | średnie |
| 5 | holonomia jako probe na ripple effects | `05-ripple-holonomy.md` | 3 tyg. | **duże** |
| 6 | odwracalna pseudonimizacja semantyczna | `06-reversible-pseudonymization.md` | 1 tydz. (sonda) | **duże** |
| 7 | odcisk semantyczny / CI-gate dla modeli | `07-model-fingerprint-ci.md` | 1,5 tyg. | bardzo małe |
| 8 | wyrównanie przestrzeni przez sygnaturę algebry | `08-alignment-by-algebra.md` | 3 tyg. | średnie |
| 9 | abstrakcyjność / wieloznaczność / typowość z `λ` | `09-psycholinguistic-scores.md` | 2 tyg. | małe |
| 10 | dryf semantyczny bez dopasowywania osi | `10-diachronic-drift.md` | 3 tyg. | średnie |
| 11 | synteza pojęcia z ograniczeniami (QP) | `11-constrained-synthesis.md` | 1,5 tyg. | małe |
| 12 | samosprzeczność tekstu + test nieprzemienności | `12-consistency-diagnostics.md` | 5 dni | bardzo małe |

## Prymitywy (odwołania P1–P8 używane w plikach)

| # | prymityw | kod | co daje |
|---|---|---|---|
| P1 | `T_r` odwracalny **dokładnie** | `core/operator/givens.py` → `inverse` | operację można cofnąć bez straty |
| P2 | izometria do skali | `metrics/geometry.py` → `isometry_error` | cosine nie jest niszczony |
| P3 | dial `λ ∈ [0,1]`, `T(0)=I` | `apply(..., lambdas=)` | natężenie sensu jako liczba ciągła |
| P4 | `decode(x,y)` → relacja + ufność | `core/heads/relation_decoder.py` | odwrotny problem: „co tu zaszło" |
| P5 | `applicability(v,r)` + budżet, kalibracja `ECE` | `core/heads/applicability.py`, `metrics/offline.py` | „czy to w ogóle zdefiniowane" |
| P6 | `C_rs` — uczona przemienność | `core/loss/curvature.py`, `load_commutation` | wiedza drugiego rzędu o relacjach |
| P7 | holonomia `HD`, `PCE(k)` | `metrics/geometry.py` | miara spójności niezależna od odpowiedzi |
| P8 | operator ważny poza grafem (`OMD`, shadow) | `metrics/offline.py`, etap F4 | aplikacja do dowolnego tekstu |

## Trzy warunki brzegowe, które decydują o całym zestawie

1. **Domena wektorów.** Operator jest trenowany na **jednostkach leksykalnych**.
   Pomysły pracujące na zdaniach i dokumentach (1, 2, 4, 12a) wymagają etapu
   **F4 z `shadow vectors` = wektorami zdań** z docelowego indeksu — to
   istniejący mechanizm (`--shadow-vectors`, `ShadowSampler`), nie nowa praca
   architektoniczna. **Pierwszy pomiar po tym readme: `OMD` na wektorach zdań.**
   Jeśli jest niskie, pomysły zdaniowe trzeba przećwiczyć albo odrzucić.
2. **Zależność od F4.** `decode` (P4) i `applicability` (P5) powstają w etapie
   F4. Pomysły 3, 6, 7, 8, 9, 10, 11 działają na samym operatorze i są
   osiągalne **nawet jeśli F4 nie doszedł**. Pomysły 1, 2, 4, 5, 12b są od F4
   zależne.
3. **Metryka przed budową.** Każdy plik ma test go/no-go na 0,5–1 dnia. Są
   ułożone tak, żeby **najtańsze testy odsłaniały ryzyka najdroższych
   pomysłów** — test z pliku 12 (3 dni) kalibruje jednocześnie pomysły 1, 2, 4
   i 12b.

## Kolejność, którą rekomenduję

```
tydzień 1        : 12 (diagnostyka, kalibruje wszystko) + test go/no-go z 1
tygodnie 2–4     : 1 (klasteryzacja) → 2 (dedup, ten sam graf)
tydzień 5        : 7 (odcisk / CI) — najtańsza wartość operacyjna
tygodnie 6–7     : 4 (entropia orbitowa) → 9 (miary psycholingwistyczne)
tydzień 8        : 3 (augmentacja) + 11 (synteza z ograniczeniami)
tydzień 9–10     : 10 (dryf) — wymaga drobnej zmiany w trenerze
sonda (równolegle): 6 (pseudonimizacja, 1 tydz.) — decyzja po fazie 6c
potem            : 5 (ripple) i 8 (wyrównanie) — najwyższa wartość, największe ryzyko
```

Pomysły 5 i 8 są na końcu nie dlatego, że są złe — oba mają najwyższą wartość
publikacyjną — ale dlatego, że oba wymagają zasobu zewnętrznego (model
open-weight z edytowalną warstwą; druga ontologia), którego nie da się zdobyć
przy okazji.

## Uwagi wdrożeniowe wspólne dla wszystkich plików

- **`RELgat_NO_TORCH_SCATTER=1`** — wymagaane w środowiskach, gdzie wheel
  `torch_scatter` segfaultuje przy imporcie (patrz `docs/GEOMETRY.md` §6).
- **Liczby na wyjściu treningu**, z których korzystają specyfikacje:
  `commutation_C.csv`/`.npy`, `config.json`, `metrics.json`, `report.json`,
  `rel2idx.json` — wszystkie zapisywane przez `handlers/wiazka_checkpoint.py`.
- **Pułapka `C_rs` w inferencji**: `RelationFlow.commutation()` czyta atrybut
  `curvature_loss`, którego `load()` nie ustawia — zawsze zwróci `None`.
  Macierz wczytuj przez `handlers.wiazka_checkpoint.load_commutation(ckpt)`.
- **Pułapka eksportu**: `RelationFlow.to_retriever_space()` używa zewnętrznego
  `projector`; wyuczona projekcja modelu (`to_retriever`, `export_proj`,
  `export_gate`) to inny obiekt. Przekaż `projector=` jawnie, inaczej export jest
  tożsamością.
- **Pułapka `ISOT` w `evaluate()`**: obecna implementacja liczy izometrię dla
  `rel[0]` (jednej relacji) dla całej partii. Jeżeli metryka `ISOT` jest bramką
  w którymś z pomysłów (3, 10), licz ją osobno per relacja.
- **Kontekst `ε` a odwracalność**: `inverse` nie zawiera członów kontekstu
  (`ctx_W`/`ctx_P`), więc przy włączonym kontekście w F4 zaokrąglenie
  `T⁻¹T = I` **przestaje być dokładne**. Pomysły 6 i 11 wymagają dokładnej
  odwrotności — uruchamiaj je z `use_context=False`.
- **Braki w kodzie, które któreś ze specyfikacji nadrobą**:
  `CompiledGivensOperator.from_product` i `.contract()` podnoszą
  `NotImplementedError` (potrzebne w pomyśle 11 i przy skracaniu łańcuchów),
  a trener nie ma `warmstart_operator` / `freeze_basis` (potrzebne w pomyśle 10).
