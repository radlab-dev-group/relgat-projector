# WIĄZKA / RGT — 10 pomysłów zastosowań + plany realizacji

Pomysły na wykorzystanie wyuczonego modelu transportu relacyjnego
(operator `T_r = s_r · exp(θ_rᵀB)x` w grupie podobieństwa, łańcuchy
kompilowane do jednego `matmul`, dial `λ`, aplikowalność, dekoder relacji).
Każdy pomysł: **co/ile**, **jak** (konkretnie, co budujemy), **metryki**,
**fazy** (z orientacyjnym nakładem), **ryzyka**.

Wspólna baza do wszystkich pomysłów (zbudować raz):

- `GraphGuard` — cienka biblioteka: `RelationFlow.load` + adaptery
  (embedding tekst→wektor, wektor→odpowiedź) + API:
  `transform(t, chain, lams)`, `is_defined(t, r)`, `explain(t, t')`,
  `budget(t, r)`, `shorten(chain)` (z `C_rs`);
- **benchmark geometryczny**: PCE(k), CCE, ISOT, HD, OMD, QR-Recall na
  Słowosieci (już w repo: `core/metrics/geometry.py`, `offline.py`,
  `query_rewriting.py`) — każdy pomysł zaczyna się od baseline'u;
- **benchmark produktowy** per pomysł (zdefiniowany poniżej).

Wszystkie pomysły są niezależne; sugerowana kolejność realizacji wg
stosunku wartości do wysiłku: 3 → 1 → 8 → 5 → 2 → 6 → 9 → 4 → 10 → 7.

---

## 1. RAG z "przyciskami semantycznymi" (steered retrieval)

**Co:** zapytanie użytkownika + intencja ("szerszy kontekst", "kontrast",
"konkretniej") → LLM wybiera łańcuch relacji i `λ` → retriever dostaje
`T_chain(t)` zamiast `t`.

**Jak:** tool `search(t, intent)` mapuje intencje na łańcuchy
(`szerszy` → `[hyper, hyper, lam=0.5]`; `kontrast` → `[ant]`;
`konkretniej` → `[hypo, lam=0.7]`) — mapa intencji→łańcuch jest
konfigurowalna i weryfikowalna `C_rs`. `applicability < 0.5` → fallback
do surowego `t` + notatka w logu.

**Metryki:** `delta_recall@k` (QR-Recall) per intencja; end-to-end:
odpowiedź lepsza wg LLM-judge (blind, 2×100 zapytań); latencja
dodatkowa (<5 ms).

**Fazy:** (1) mapa intencji + tool, 3 dni; (2) A/B na benchmarku RAG,
1 tydzień; (3) kalibracja progów applicability, 2 dni.

**Ryzyka:** intencje, których nie ma w ontologii (mitygacja: fallback);
użytkownik nie mówi intencji jawnie (mitygacja: LLM klasyfikuje intencję
z kontekstu, z konfigurowalnym "domyślnie nic").

## 2. Weryfikator twierdzeń strukturalny (anti-hallucination)

**Co:** każda odpowiedź LLM → dekompozycja na krotki `(s, r, d)` →
weryfikacja w grafie (`decode` + `compile` + `applicability`) →
oznaczenie `supported / contradicted / unsupported` → filtrowanie lub
wymuszenie cytowania.

**Jak:** dekompozycja tylko dla entity występujących w kontekście RAG
(restricted linking — tanio i bez NER); `contradicted` gdy `decode` daje
relację `r'` z `C_rs[r, r'] < 0.1` i `cos(T_{r'}(v_s), v_d) > 0.8`
(graf mówi "to jest co innego"); próg `unsupported` z ECE-kalibrowanego
confidence.

**Metryki:** hallucination rate (gold: 500 twierdzeń z etykietami
supported/unsupported), spadek useful-ness <2 pkt (nie wyrzucamy
poprawnych); koszt <50 ms/twierdzenie.

**Fazy:** (1) dekompozycja + weryfikator offline, 1 tydzień; (2)
zestaw gold (500 twierdzeń, 2× annotator, IAA), 1 tydzień; (3) integracja
w pętli (tryb: filter / flag / rewrite), 1 tydzień.

**Ryzyka:** dekompozycja myli się (mitygacja: tylko twierdzenia z
entity z kontekstu; reszta przechodzi z flagą `unverified`); pokrycie
ontologii (mitygacja: tryb "brak grafu").

## 3. Interaktywna eksploracja wiedzy (query space explorer)

**Co:** UI: zapytanie → wykres sąsiedztwa, na którym użytkownik klika
relacje i suwak `λ`; każdy klik = `T_r`; widać na żywo, dokąd "idzie"
zapytanie (top-k odzyskanych po transformacji).

**Jak:** `compile(chain)` po każdym kliku (O(1)); `applicability` jako
alpha/kolor krawędzi; `decode` między dwoma punktami UI = "co tu
właściwie zrobiłeś?" (wyjaśnienie); eksport łańcucha do RAG (pomysł 1).

**Metryki:** usability (5 użytkowników, zadania typu "znajdź
kontrast do X"); technicznie: ISOT/OMD na wektorach generowanych
suwakiem (czy `λ=0.3` dalej leży na manifoldie).

**Fazy:** (1) CLI demo (graf + top-k), 2 dni; (2) UI (streamlit), 3 dni;
(3) testy użycia, 3 dni.

**Ryzyka:** niskie — to showcase + narzędzie debugowania słownika
(relacje, które "nie działają", widać od razu przez applicability).

## 4. Automatyczne przerabianie zapytań (query rewriting agent)

**Co:** LLM-agent, który przed retrievem sam decyduje: "to zapytanie
potrzebuje najpierw uogólnienia, potem zawężenia" — i *dowodzi*
przekształcenia (łańcuch + `λ` + uzasadnienie z `C_rs`).

**Jak:** agent dostaje `explain` (dekoder) i `budget`; strategia
planowania: BFS po łańcuchach długości ≤3, scoring =
`applicability · delta_recall` (szacowany na próbkach gold); `C_rs > 0.9`
pozwala kanonicznie sortować łańcuch (cache kompiLOWANYCH operatorów —
`commutation_C.csv` → tablica cache'ów).

**Metryki:** recall@10 agent vs surowe zapytanie (benchmark zapytań
wieloznacznych); frakcja zapytań, gdzie agent poprawił ranking; koszt
(liczba transformacji na zapytanie).

**Fazy:** (1) planner BFS + cache operatorów, 1 tydzień; (2) integracja
z LLM (tool calling), 3 dni; (3) eval + ablacje (bez C_rs, bez budget),
3 dni.

**Ryzyka:** planner może "luzować" zapytanie do punktu, w którym
odzyskuje zbyt wiele (mitygacja: hard limit `λ` i długości łańcucha +
`OMD` lokalny jako hamulec).

## 5. Kompletnie uzupełnianie / naprawianie ontologii

**Co:** wykrywanie braku krawędzi i jej etykiety: (a) dla pary węzłów
spoza grafu `decode` proponuje relację; (b) dla krawędzi o niskim
`cos(T_r(v_s), v_d)` — relacja jest "zła" lub brakuje innej; (c)
holonomia na czwórce z jedną brakującą krawędzią → wyznaczenie
`T_brak = T_⁻¹T_qT_p` i test, czy operator ma sensowną interpretację
(skalowanie przez istniejące relacje).

**Jak:** pipeline batch: wszystkie pary top-k (cosine > 0.75, bez krawędzi)
→ `decode` → sortowanie po confidence; raport: propozycje krawędzi
(z typem) + propozycje "krawędź X powinna być relacją Y, nie Z" +
czwórki z holonomią > prógu (kandydaci na nową relację). Człowiek zatwierdza.

**Metryki:** precision/recall propozycji vs gold-etykiety (subset
Słowosieci z celowo usuniętymi krawędziami — 500 usunięć); liczbę
"nowych relacji" potwierdzonych przez leksykografów.

**Fazy:** (1) pipeline decode-batch, 2 dni; (2) symulacja braków + gold,
1 tydzień; (3) interfejs review (CSV + UI), 2 dni.

**Ryzyka:** decode jest szkolony na istniejących relacjach — propozycja
"nowej" relacji to w praktyce mieszanka istniejących (mitygacja: raportuje
współrzędne θ w bazie B — widoczne, czy to kombinacja, a nie nowość).

## 6. Semantyczne "przesunięcia" treści (parafraza / styl / abstrakcja)

**Co:** przekształcenie całego dokumentu: "napisz to bardziej formalnie /
dla laików / kontrastowo". Wymiar wektorowy: każdy wektor segmentu
przechodzi `T_r(λ)` przed LLM-owym parafrazowaniem (LLM dostaje
"przesunięty" kontekst), albo transformowany wektor jest sam
rekonstruowany (retrieval augmentacji parafrazy).

**Jak:** `formal` → `[hyper, lam=0.3]` + filtrowanie segmentów przez
applicability; `laik` → `[hypo, lam=0.5]`; per-segment dial (nie na
cały dokument naraz — mniejszy dryf). Walidacja: PPL po transformacji +
LLM-judge "czy styl się zmienił, czy treść została".

**Metryki:** styl (klasyfikator stylu, accuracy zmiany); wierność
(entropia twierdzeń: wspólna frakcja twierdzeń przed/po, cel >0.9);
QR-Recall dla segmentów.

**Fazy:** (1) pipeline segment→transform→parafraza, 3 dni; (2)
benchmark styl+wierność (200 dokumentów), 1 tydzień; (3) strojenie `λ`,
2 dni.

**Ryzyka:** rotacje nie "skracają" pojęcia (ograniczenie operacyjne,
GEOMETRY.md §7) — mitygacja: `s_r` + opcjonalny człon low-rank, jeśli
PCE(1) nie spada; dryf kumulacyjny na długich tekstach — mitygacja:
per-segment + OMD jako hamulec.

## 7. Weryfikacja spójności bazy wiedzy / detekcja sprzeczności

**Co:** audyt KG: (a) krawędzie sprzeczne (para `syn` + `ant` między tymi
samymi węzłami, albo łańcuch o `HD` > progu zwracający "błąd holonomii");
(b) pary relacji, które *powinny* komutować (według domeny) a `C_rs`
mówi inaczej — flaga do review ontologii; (c) węzły, dla których
applicability jest niska dla relacji, które "powinny" istnieć
(np. rzeczownik bez hiperonimów).

**Jak:** batch over graph: holonomia na wszystkich czwórkach (sampler z
repo), `C_rs` vs tabela oczekiwań domenowych (konfig), applicability
per węzeł×relacja → raport CSV + heatmapa. Zero LLM — czysta geometria.

**Metryki:** odnalezienie wstrzykniętych błędów (benchmark: 100
sprzeczności wstawionych w kopię Słowosieci — recall@próg); czas
audytu pełnego grafu.

**Fazy:** (1) skan holonomii + raport, 3 dni; (2) benchmark wstrzykniętych
błędów, 3 dni; (3) heatmapa + interfejs, 3 dni.

**Ryzyka:** czwórki są rzadkie w wordnetach (PLAN.md §6.2) — mitygacja:
czwórki cieniowe (shadow squares) + uczona `C_rs` zamiast twardych
testów; fałszywe alarmy z rzadkich relacji (mitygacja: próg minimum
częstotliwości relacji w raporcie).

## 8. Kalibrowany "drift monitor" dla pipeline'ów RAG (operacyjny)

**Co:** produkcja: każdy request RAG loguje łańcuch transformacji i
współczynniki; monitor liczy na żywo: średnie `applicability`, budżet
dryfu, OMD na próbkach, rozkład `λ` — i alarmuje, gdy pipeline "pływa"
(np. ktoś zmienił embeddera, ontologię, albo model zaczął nadużywać
transformacji).

**Jak:** middleware (10 linijek) wokół `RelationFlow.apply*`: metryki do
metrystyki (prometheus/wandb); progi z metryk treningowych (ISOT<0.05,
ECE<0.08 — już zdefiniowane w `docs/GEOMETRY.md`); alert + automatyczny
fallback do surowego zapytania, gdy próg przekroczony N razy z rzędu.

**Metryki:** detekcja wstrzykniętego "złamanego embeddera" (test:
zamiana embeddera w pipeline → czas do alaru); false-alarm rate.

**Fazy:** (1) middleware + metryki, 2 dni; (2) progi + fallback, 1 dzień;
(3) chaos-test (3 scenariusze degradacji), 2 dni.

**Ryzyka:** niskie; to ubezpieczenie, nie feature. Wartość: bez tego
wszystkie inne pomysły cicho się degradują po zmianie embeddera.

## 9. Relational arithmetic — syntetyczne dane i eval dla LLM

**Co:** generowanie par (zapytanie, oczekiwana transformacja) do
trenowania/oceny LLM w rozumowaniu relacyjnym: `T_r∘T_s⁻¹` daje nową
operację bez treningu (GEOMETRY.md §4) — stąd nieskończony strumień
zadani typu "X jest do Y tym, czym A do B" z *dokładnym* gold (operator
jest deterministyczny).

**Jak:** generator: losowe łańcuchy + odwrócenia → pary (t, t');
tekstualizacja (LLM opisuje t, t' naturalnym językiem); gold = łańcuch.
Użycia: (a) SFT/RLHF data dla rozumowania relacyjnego, (b) eval
"analogical reasoning" LLM-ów (nowy benchmark), (c) distylacja małego
modelu klasyfikatora relacji (zamiast dużego LLM-judge'a).

**Metryki:** jakość danych (LLM-judge: frakcja par semantycznie
spójnych, cel >0.9); wyniki LLM na benchmarku przed/po SFT.

**Fazy:** (1) generator + tekstualizacja, 3 dni; (2) pipeline QA (judge),
3 dni; (3) eksperyment SFT/eval na 2 modelach, 1 tydzień.

**Ryzyka:** tekstualizacja wprowadza szum (gold jest wektorowy, opis
tekstowy przybliżony) — mitygacja: judge ocenia spójność opisów, nie
współrzędne; rozmiar przestrzeni relacji ograniczony do tych z ontologii.

## 10. "Relational cache" — przyspieszenie i determinizm RAG

**Co:** cache kompiLOWANYCH operatorów + wektorów transformowanych,
sterowany przez `C_rs`: łańcuchy z parami `C_rs > 0.9` są kanonicznie
sortowane przed cache'owaniem (więcej trafień); łańcuchy z `C_rs ≈ 0`
nie są upraszczane (kolejnoŚĆ ma znaczenie). Dodatkowo: `applicability`
jako warunek trafienia cache (wynik transformacji dla `a_r < 0.3` nie
jest cache'owany — za mała pewność).

**Jak:** wrapper `CacheLayer(RelationFlow)`: key = (kanoniczny łańcuch,
`hash(t)`, `λ`-bucket); tablica `C_rs` decyduje o redukcji klucza;
statystyki trafień; TTL po zmianie embeddera/ontologii (integracja z
pomysłem 8).

**Metryki:** cache hit rate (cel >40% na realnym ruchu zapytań);
determinizm: bitwise identity wyników dla identycznego łańcucha
(własność testowana, nie mierzona); spadek latencji p95.

**Fazy:** (1) cache + kanonizacja z `C_rs`, 2 dni; (2) benchmark ruchu
(symulacja z realnych logów), 3 dni; (3) TTL + integracja z monitorem,
1 dzień.

**Ryzyka:** kanonizacja przy `C_rs ≈ 0.9` (strefa graniczna) — mitygacja:
twardy próg 0.95 do redukcji, poniżej = pełny łańcuch w kluczu;
`λ`-bucketing (kwantyzacja 0.05) może utracić fine-grained dial —
mitygacja: buckety 0.01 dla drogi interaktywnej (pomysł 3).

---

## Kolejność i zależności

```
GraphGuard (baza) ──┬──> 3 (explorer, showcase)  ──> 1 (steered RAG)
                    ├──> 8 (drift monitor)        ──> 1,4 (ubezpieczenie)
                    ├──> 2 (weryfikator)          ──> 4 (agent rewriting)
                    ├──> 5 (kompletnie KG)
                    ├──> 6 (przesunięcia treści)
                    ├──> 7 (audyt spójności)
                    ├──> 9 (syntetyczne dane/eval)
                    └──> 10 (relational cache)
```

**Sprint 1 (2 tyg.):** baza + 3 + 8 (szybki, namacalny efekt +
bezpieczeństwo operacyjne).
**Sprint 2 (3 tyg.):** 1 + 2 (główna wartość produktowa: RAG z
przyciskami + anti-hallucination).
**Sprint 3 (3 tyg.):** 5 + 7 (wartość "dla ludzi z ontologią"), 10
(przyspieszenie), 4 (agent).
**Sprint 4 (2 tyg.):** 6 + 9 (warsztat treści + dane/eval).

## Wspólne założenia i otwarte pytania

- Embedder: wszystkie pomysły zakładają, że embedder zapytań jest w tej
  samej przestrzeni, co trening (Słowosieć). Zmiana embeddera = retrain
  (pomysł 8 to wykrywa, ale nie naprawia).
- Pokrycie domen: Słowosieć = język ogólny. Dla domen medycznych/
  prawniczych: trening na ontologii domeny (SNOMED, UMLC) — pipeline
  identyczny, zmienia się dataset.
- Gold-etykiety: pomysły 2 i 5 wymagają ~500 etykiet (2× annotator,
  ~2 dni). Pozostałe działają na metrykach geometrycznych (bez gold).
- Otwarte: czy kontekst `ε` (F4) włączać w produkcji — zwiększa
  wieloznaczność relacji, ale CCE przestaje być tautologią; decyzja po
  Faza 1 pomysł 1 (A/B z `ε` on/off).
