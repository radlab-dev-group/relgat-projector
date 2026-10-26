# WIĄZKA / RGT — zastosowanie w modelach LLM

Analiza repo `relgat-projector` i plan wykorzystania wyuczonego modelu
transportu relacyjnego (WIĄZKA / Relational Gauge Transport) w pipeline'ach
LLM. Dokument towarzyszący: `LLM_APPLICATION_IDEAS.md` (10 pomysłów z planami
implementacji).

Data: 2026-10-22. Stan repo: gałąź `feature/wiazka-rgt`, checkpointy trenowane
na Słowosieci (plwordnet-milvus exporter), operator `givens` (obrót + skala).

---

## 1. Co jest wejściem

Dane treningowe (format exporter `plwordnet-milvus --export-relgat-dataset`):

- `lexical_units_embedding.pickle` — **zamrożone** wektory jednostek
  leksykalnych (słowa/formy), klucz: `int node_id` → wektor `R^D_in`
  (w praktyce ~1152 wym. z sentence-transformerów);
- `relation_to_idx.json` — nazwa relacji → id (syn, ant, hyper, hypo,
  meronym, …);
- `relations_triplets.json` — krotki `[from_id, to_id, rel_name]`;
- opcjonalnie `shadow vectors` — **prawdziwe wektory tekstu spoza grafu**
  (z indeksu retrievera), używane w etapie F4, żeby operator był nadzorowany
  poza węzłami grafu.

Kluczowa cecha: graf semantyczny (Słowosieć / WordNet / ontologia) +
zamrożone embeddingi. Model NIE uczy embeddingów słów — uczy się **operatorów
działających na istniejącej przestrzeni wektorowej retrievera**.

## 2. Co jest wyjściem (jaki model)

Wyjściem jest checkpoint `WiazkaModel` (zapisywany przez
`RelationFlow.save` / `wiazka_checkpoint`), czyli:

1. **Operator relacyjny** (`GivensRelationOperator`): każda relacja `r` to
   element grupy podobieństwa `R⁺ × O(D)`:
   `T_r(x) = s_r · exp(θ_rᵀ B) x`, gdzie `B_m` to uczona baza antysymetrycznych
   generatorów (K z so(D)), `θ_r` współrzędne relacji w algebrze Liego,
   `s_r` uczona skala. Własności **z konstrukcji**:
   - dokładna odwrotność `T_r⁻¹` (obrót + dzielenie przez `s_r`);
   - zachowanie geometrii cosine (izometria do skali) — jedyna metryka, którą
     rozumie baza wektorowa, nie jest niszczona;
   - **nieprzemienność** (`r₁∘r₂ ≠ r₂∘r₁`) — niemożliwa dla DistMult;
   - **dial** `λ ∈ [0,1]`: `T_r(0) = I` dokładnie, `T_r(1)` = pełna relacja
     ("40% bardziej abstrakcyjnie", bez retreningu).
2. **Transport-aware GAT** (`TransportGATLayer` × N): udoskonalanie wektorów
   węzłów w przestrzeni operatora; wiadomość `u→v` przez relację `r` to
   `T_r(proj(x_u))`, bramkowany residual (start = tożsamość).
3. **Głowice**: `ApplicabilityHead` (czy relacja w ogóle jest zdefiniowana w
   punkcie `x` + budżet dryfu) i `RelationDecoder` (odwrotne zadanie: która
   relacja mapuje `x` na `y`).
4. **Parametry straty** `C_rs` (uczona macierz przemienności relacji) +
   artefakty: `commutation_C.csv`, `config.json`, `metrics.json`, `report.json`.
5. **Eksport** `to_retriever_space(x)` — wektory z powrotem w przestrzeni
   bazowej (tę samą, co ma Milvus), z bramką `g` startującą od 0.

API inferencji (`RelationFlow`): `refine` → `apply` / `apply_chain` /
`compile` (łańcuch = jeden `matmul`, `O(1)` w długości łańcucha) → `lam` dial →
`applicability` → `decode` → `to_retriever_space`.

Metryki: `PCE(k)` (path compilation error — metryka główna, płaska krzywa w `k`
oznacza prawdziwy operator), `CCE` (cycle closure), `ISOT` (zniekształcenie
cosine), `HD` (holonomia), `OMD` (degradacja off-graph), `QR-Recall`
(delta recall@k po transformacji zapytania — metryka produktowa).

## 3. Co jest uczone (i co NIE)

Uczony jest (curriculum F1→F4, bramki na held-out krawędziach):

- `F1 nodes`: udoskonalenie węzłów (`L_anchor`, `L_edge`) przy zamrożonym
  operatorze;
- `F2 relations`: operator relacji k=1 (`L_con`, `L_isom`, `L_cycle`);
- `F3 composition`: ścieżki `k≥2`, czwórki zamknięte (holonomia), `L_path`,
  `L_curv`, `L_commut` + `C_rs`, kompilacja łańcuchów;
- `F4 shadows`: wektory spoza grafu (`L_shadow`: cycle/commut/pull),
  aplikowalność, dekoder, kontekst (`eps`, hypernetwork `θ(r,x)`).

NIE jest uczony: embeddingi słów (zamrożone), parametry LLM, retriever.
Model jest mały (operator: `num_rel × K` + `K × D × D`; głowice: 2–3 warstwy
MLP) — mieści się na CPU, inferencja to kilkanaście `matmul`.

Sedno unikalności (PLAN.md §1): graf jako **połączenie (connection)** z
transportem wzdłuż krawędzi; kara holonomii = wymuszenie spójności ścieżkowej,
co jest właściwie **właściwością uogólnienia** — jedyny sposób, by wiedzieć,
gdzie `T_{r₂}T_{r₁}(t)` powinno wylądować, gdy `t` nie jest węzłem grafu.

## 4. Gdzie to w ogóle można zastosować (zasięg)

Model operuje w przestrzeni embeddingów używanej przez retrievera. Wszystko,
co łączy LLM z bazą wektorową (RAG, memory, tool-use, agentów), może
wykorzystać:

- transformację zapytania: `t' = T_r(t)` — zmiana znaczenia wektora z
  gwarancją, że dalej leży na "manifoldie" sensownych wektorów;
- kompozycję: `t → r₁ → … → r_k` skompilowaną do jednego operatora;
- test sensowności: `applicability(t, r)` — kiedy relacja nie ma definicji
  (np. antonim rzeczownika własnego) → system odmawia zamiast zwracać szum;
- odwrócenie: `decode(t, t')` — "która relacja tu działała?" →
  automatyczne rozumowanie o intencji zapytania.

## 5. Wykorzystanie w LLM: ograniczanie halucynacji w kontekście grafu relacji

Główna hipoteza: halucynacja w RAG-LLM ma dwie przyczyny, na które ten model
działa punktowo:

(a) **odrzeźbienie od źródła** — LLM generuje twierdzenia, których nie da się
    odwzorować na ścieżki w grafie / kawałkach odzyskanych;
(b) **niekontrolowana ekspansja semantyczna** — model "wypływa" po
    skojarzeniach, których graf nie przewiduje (tutaj wchodzą operatorzy:
    każda ekspansja jest jawnym transportem przez zdefiniowane relacje,
    a nie swobodnym driftem logitsów).

### 5.1 Architektura: RelGAT-Transport jako warstwa między LLM a światem

```
                ┌─────────────────────────────────────────────┐
                │                LLM (frozen)                 │
                │  plan: [retrieve, transform, verify, answer]│
                └──────┬───────────────▲──────────┬───────────┘
             narzędzia │               │ kontekst │ twierdzenia
        ┌──────────────▼───────┐  ┌────┴─────────▼─────────────┐
        │  Retriever (Milvus)  │  │  Graph Constraint Layer    │
        │  t, t' = T_r(t) ...  │  │  (WIĄZKA + LLM-judge)      │
        └──────────────────────┘  └────────────────────────────┘
```

Trzy tryby użycia w pętli generowania:

1. **Steered retrieval (przed generowaniem).** Zamiast wysyłać surowe
   zapytanie, LLM deklaruje *operację semantyczną*:
   `search(t, chain=[hyper, hyper], lam=0.5)` = "szukaj 2 poziomy wyżej
   hierarchii, łagodnie". Transformacja jest:
   - jawną operacją na grafie (każdy krok to relacja z ontologii),
   - kontrolowana przez `applicability` (jeśli `a_r(t) < threshold` —
     np. hiperonimia dla rzeczownika własnego — pipeline odmawia i LLM dostaje
     sygnał "brak definicji", zamiast fałszywych hitów),
   - mierzalna: `delta_recall@k` (QR-Recall) mówi, czy transformacja realnie
     poprawiła ranking przed generowaniem.
   Efekt: LLM generuje na bazie dokumentów, które są **semantycznie
   zgodne z zamierzoną operacją**, a nie z przypadkowym dryfem embeddingów.

2. **Post-hoc verification (po generowaniu).** Każde twierdzenie faktograficzne
   LLM jest dekomponowane na krotki `(podmiot, relacja, orzecznik)` (LLM robi
   to dobrze; to strukturalne, nie semantyczne). Dla każdej krotki:
   - szukamy węzłów grafu najbliższych `podmiot`/`orzecznik`
     (`nearest` w indeksie, po `to_retriever_space`);
   - `decode(v_s, v_d)` mówi, **która relacja** łączy je i z jakim
     zaufaniem; porównujemy z relacją z twierdzenia;
   - dodatkowo: `compile([r])` i sprawdzenie `cos(T_r(v_s), v_d)` — czy
     transport faktycznie trafia w cel;
   - `applicability` filtruje twierdzenia, gdzie relacja nie ma definicji.
   Twierdzenia, które nie mają odwzorowania w grafie (lub mają je przez
   *inny* niż deklarowany operator), oznaczane są jako `unsupported` —
   albo usuwane z odpowiedzi, albo podawane LLM-owi do reformulacji
   ("sformułuj inaczej / oznacz jako niepewne"). To **weryfikacja
   strukturalna, nie semantyczna** — tanio się ją liczy (kilka matmul na
   twierdzenie) i działa bez gold-annotacji.

3. **Constrained decoding / steering (w trakcie generowania).**
   a) *Logit gating*: dla tokenów, które LLM próbuje wygenerować w sekcji
      faktograficznej, liczymy `applicability` najbliższego węzła grafu dla
      relacji sugerowanych przez kontekst; niska aplikowalność → karanie
      tokenów (additive bias w logitsach) lub wymuszenie cytowania źródła.
   b) *Drift budget*: `budget_b(t, r)` z głowicy aplikowalności to
      "pozostały budżet dryfu" — po przekroczeniu (np. łańcuch >3 relacji,
      lub `OMD` lokalny spada) generator ma obowiązek wrócić do źródła
      (retrieve + cytat). To mechaniczne wdrożenie zasady "nie idź dalej niż
      graf".

### 5.2 Dlaczego akurat ten model ogranicza halucynacje (mechanizm)

- **Każdy krok semantyczny jest operatorem z grupy** — nie ma "półśrodka",
  w którym wektor cicho dryfuje: `T_r(λ)` ma dokładnie określoną geometrię
  (izometria do skali), więc `ISOT` i `CCE` można wymuszać jako warunki
  poprawności odpowiedzi, nie jako metryki eval.
- **Spójność ścieżkowa (holonomia = 0)** gwarantuje, że dwa różne sposoby
  dojścia do tego samego znaczenia dają ten sam wektor — LLM nie może
  "obojgać" faktów przez wybór innej drogi rozumowania nad tymi samymi
  relacjami. To odpowiada typowemu mechanizmowi halucynacji: łańcuchowi
  skojarzeń, w którym każdy pojedynczy krok jest "w miarę sensowny", a suma
  wypada daleko od prawdy.
- **`C_rs` jako mapa zależności relacji** — pary o `C_rs > 0.9` można
  kanonicznie sortować (skracanie/caching łańcuchów), a pary o `C_rs ≈ 0`
  to miejsca, gdzie kolejność zmienia znaczenie — czyli dokładnie miejsca, w
  których LLM najczęściej myli się w rozumowaniu wielokrokowym. Można z
  `C_rs` budować listę "stref ryzyka" i wymagać w nich podwójnego
  weryfikowania.
- **Dekoder relacji** zamienia "czy to halucynacja?" w "który operator tu
  działał i z jakim zaufaniem?" — pytanie, na które model odpowiada
  kalibrowanym prawdopodobieństwem (ECE jest metryką treningową), a nie
  subiektywną oceną LLM-judge'a. LLM-judge zostaje jako *offline*
  walidacja, nie w pętli inferencji (PLAN.md §6.5).

### 5.3 Plan wdrożenia (fazy)

**Faza 0 — benchmark halucynacji (1 tydzień).** Zestaw eval: 500–2000
zapytań RAG z gold-źródłami (np. subset HotpotQA/2WikiMultihopHQ przełożony
na Słowosieć, albo własny korpus z ontologią). Pipeline baseline:
vanilla RAG + LLM. Metryki: factuality (twierdzenia z weryfikacją
strukturalną), hallucination rate (frakcja unsupported twierdzeń),
faithfulness. Wynik Fazy 0 = odniesienie.

**Faza 1 — steered retrieval (2 tygodnie).**
- Integrujemy `RelationFlow.load` jako tool LLM (`search(t, chain, lam)`);
- budujemy adapter `applicability` → "odmów z powodem";
- mierzymy: `delta_recall@k` przed/po na gold-parach, potem end-to-end
  factuality. Sukces: delta_recall stabilnie > 0 dla `hyper`, `syn`,
  `ant`; factuality rośnie o ≥5 pkt względem Fazy 0.

**Faza 2 — post-hoc verification (2 tygodnie).**
- Dekompozycja twierdzeń na krotki (few-shot, deterministycznie);
- weryfikator strukturalny (`decode` + `compile` + `applicability`);
- dwa warianty: (a) filtrowanie (usuwanie unsupported), (b) steering
  (LLM dostaje listę unsupported i reformuluje);
- metryka: spadek hallucination rate przy utracie <2 pkt useful-ness
  (nie usuwamy poprawnych odpowiedzi).

**Faza 3 — constrained decoding (2 tygodnie, eksperymentalne).**
- Additive bias w logitsach sterowany `applicability`/`budget` (wymaga
  dostępu do logitsów — LLM open-weight);
- alternatywnie bez logitsów: "drift budget" jako reguła w pętli agenta
  (po N krokach transformacji → wymuszone retrieve+citat);
- mierzymy koszt: latencję (cel: <50 ms na twierdzenie w F2, <2 ms na token
  w F3a) i spadek general quality (PPL / win-rate vs baseline).

**Faza 4 — produkt (1 tydzień).**
- Wrapper `GraphGuard` (tool + weryfikator + budget) jako biblioteka;
- raport: krzywa factuality vs latencja, ablacje (bez applicability,
  bez decode, bez C_rs gating), dokumentacja "stref ryzyka" z `C_rs`.

Ryzyka i mitygacje: (1) pokrycie ontologii — jeśli zapytanie dotyczy
dziedziny spoza Słowosieci, weryfikator nie ma węzłów; mitygacja:
`applicability ≈ 0` dla wszystkich relacji → tryb "brak grafu, standardowe
RAG" (graceful degradation); (2) dekompozycja twierdzeń jest słabym ogniwem;
mitygacja: dekomponujemy tylko twierdzenia z entity znanymi z kontekstu RAG
(restricted entity linking, nie NER z surowego tekstu); (3) koszt —
operator to 1–2 matmul D×D na wektor, przy D=256 to <0.1 ms na CPU;
wąskie gardło to dekompozycja LLM-owa (1 raz na odpowiedź, nie na token).

## 6. Podsumowanie

Model z tego repo nie jest "lepszym embeddingiem" — jest **algebrą
operacji semantycznych w przestrzeni embeddingów**. Dla LLM to daje coś,
czego nie da czysty RAG ani czysty KG: każda zmiana znaczenia wektora
zapytania/twierdzenia jest (i) jawną operacją, (ii) zweryfikowalną
(strukturalnie), (iii) kalibrowaną (applicability/budget), i (iv) skompilowaną
do O(1) w produkcji. To dokładnie interfejs potrzebny, by halucynacja stała
się wykrywalnym stanem ("ten krok nie ma odwzorowania w grafie"), a nie
statystycznym szumem.
