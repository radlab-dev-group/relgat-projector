# WIĄZKA — Relacyjna Wiązka Transportowa (ang. *Relational Gauge Transport, RGT*)

Szczegółowy plan przebudowy procesu uczenia `relgat-projector`.
Cel: relacje semantyczne jako **operatorzy w tej samej przestrzeni co wektory
z retrievera**, komponowalne (`N -> r_1 -> ... -> r_k -> K`), odwracalne,
stabilne numerycznie i niekurczące geometrii i — co najważniejsze — **spójne
niezależnie od ścieżki**, tak aby działając relacjami na wektorze tekstu
spoza grafu zawsze trafiać w sensowny punkt przestrzeni.

Robocza nazwa metody: **WIĄZKA** / **RGT**.

---

## 0. Diagnoza stanu obecnego (dlaczego obecnego schematu nie da się dowieźć do (3))

Wejściem jest graf semantyczny (Słowosieć / WordNet / ontologia) z zamrożonymi
wektorami węzłów. Obecny pipeline: `RelGAT` (udoskonalenie węzłów) -> scorer
(`DistMult`/`TransE`) pełniący jednocześnie rolę operatora transformacji ->
`ProjectionHead` z powrotem do wymiaru wejściowego.

Błędy blokujące (naprawić w Fazie 0, zanim cokolwiek mierzymy):

- `relgat_projector/handlers/models/relgat.py:17` — brak `import pickle`
  (`NameError` przy wczytywaniu wektorów węzłów).
- `relgat_projector/core/model/model.py:216` — `get_config()` odwołuje się do
  `self._config`, który nigdzie nie jest ustawiony -> `save_pretrained()`
  i `load_from_pretrained()` padają.
- `relgat_projector/trainer/relgat_projector.py:~300` (`upper_is_better = False`)
  oraz `:708` (`_on_eval_end`) — MRR jest metryką „im więcej tym lepiej”, ale
  śledzenie „best” i early stopping działają dla kryterium minimalizacji.
  Model jest wybierany w pobliżu **najgorszego** wyniku, a training wcześniej
  kończy się dokładnie wtedy, gdy model zaczyna być dobry.
- `RelGATModel.forward` transformuje relacją **obie** strony (`src` i `dst`)
  i liczy scorer na obu po transformacji. Dla DistMult daje to
  `<s*r, r, d*r>` zamiast `<s, r, d>`, czyli inny model niż ten, który jest
  deklarowany i niż operator użyty potem w `transform()`. Scoring i transformacja
  muszą korzystać z **tej samej** definicji operatora.
- `RelGATDataset._prepare_edge_dataset` buduje topologię GAT wyłącznie z
  krawędzi treningowych, a ewaluacja i inferencja działają na pełnym grafie.
  Rozdziel: (a) topologia widoczna dla GAT (wiadomo, że to nie etykieta) vs
  (b) zbiór krawędzi podlegający predykcji. Obecnie te dwie rzeczy są pomieszane.
- `EdgeDataset._precompute_negative_candidates` materializuje
  `list(ws - {dst})` dla **każdej** krawędzi -> O(|E|·|V|) pamięci, a przy
  samplingu `random.choice` daje negatywy prawie zawsze trywialnie łatwe.
- Brak ścieżek `k >= 2` w danych treningowych. Operator złożony
  `f_{r_2}(f_{r_1}(x))` **nie jest w ogóle nadzorowany** — dziś to, co dzieje
  się po nałożeniu dwóch relacji, jest interpolacją, nie wyuczoną semantyką.
- `transform_from_vectors` nie kontroluje normy ani odległości od „manifoldu”
  wektorów -> po 2-3 relacjach wektor wypada z obszaru, w którym cosine w
  wektorowej bazie danych cokolwiek znaczy.

Wniosek: trzeba zmienić **postać operatora relacji** i **sposób nadzorowania
kompozycji**. To jest sedno tego planu.

---

## 1. Idea przewodnia — to jest ta unikalna część

### 1.1 Framing: graf semantyczny to wiązka z połączeniem (connection)

Zamiast traktować relację jako „wektor/translację” albo „diagonalne mnożenie”,
traktujemy graf jako **wierzchołki z przestrzenią włókna `R^D` w każdym
wierzchołku i transportem wzdłuż krawędzi**:

    T_{r @ x} : R^D -> R^D      transport relacji r zaczepiony w węźle x

Trzy konsekwencje, które są osią całej metody:

1. **Operator jest odwracalny i prawie izometryczny z konstrukcji**
   (`T` jako obrót + skalowanie). Dzięki temu cosine — czyli jedyna metryka,
   którą rozumie wektorowa baza danych — **nie jest niszczony** przez
   transformację. To jest dokładnie warunek, którego brakuje, żeby `t'` dało
   się sensownie wyszukać.
2. **Składanie relacji jest domknięte**: iloczyn dwóch operatorów to operator
   tej samej klasy. Ścieżka `r_1..r_k` daje się **skompilować do jednego
   operatora** i wykonać jako jeden `matmul` (koszt stały, niezależny od długości
   ścieżki i od grafu).
3. **Ścieżkowa niezależność ma nazwę i metrykę**: jeśli transport nie zależy od
   wyboru ścieżki między `A` i `B`, to krzywizna połączenia jest zerowa.
   Zatem: **karanie krzywizny (holonomii) = wymuszanie, żeby `r_1..r_k` działało
   spójnie na dowolnym wektorze, także spoza grafu.**

To ostatnie jest osią unikalności. W literaturze o embeddingach grafów
(kompozycja relacji) spotyka się nadzorowanie złożenia na ścieżkach, ale
**wymuszanie spójności ścieżkowej jako warunku całkowalności połączenia i
użycie tego do przenoszenia operatorów na wektory spoza grafu — nie jest
stosowane**. Dla użytkownika efekt jest namacalny: `t -> r_1 -> r_2` daje ten
sam wynik niezależnie od tego, którą drogą „doszliśmy” do opisu relacji, więc
operator można zastosować do tekstu, którego w grafie nigdy nie było.

### 1.2 Postać operatora: obrót + skalowanie (grupa podobieństwa)

Relacja `r` to element grupy podobieństwa `R^+ x O(D)` (najmniejsza grupa,
która zachowuje kształt przestrzeni cosine):

    T_r(x) = s_r * Rot_r(x),         Rot_r ∈ O(D),  s_r > 0

Parametryzacja **bez** jawnej macierzy `D x D` — łańcuch obrotów Givensa
(standard numeryczny na `O(D)`, tu użyty jako struktura relacji):

    Rot_r = G_M(theta_{r,M}) ... G_2(theta_{r,2}) G_1(theta_{r,1})

- `G_i` — obrót w płaszczyźnie `(a_i, b_i)`; zbiór płaszczyzn `M` jest **wspólny
  dla wszystkich relacji** i wyuczony raz (indeksy jako wagi twarde/EM lub
  trzymane jako bufor).
- kąty `theta_{r,i} ∈ R^{num_rel x M}` — współrzędne relacji w algebrze.
- Koszt nałożenia operatora: `O(M)` operacji na wektor (zamiast `O(D^2)`).
- **Odwrócenie jest dokładne i darmowe**: `T_r^{-1} = G_1(-theta_1) ... G_M(-theta_M)`
  oraz `1/s_r`. Zero błędu numerycznego, zero dodatkowych parametrów.
- **Płynna skala relacji (dial)**: `T_r(lambda)` = te same obroty z kątami
  `lambda * theta`. `lambda=0` to tożsamość, `lambda=1` pełna relacja.
  Pozwala pytać o „tekst w 40% bardziej anonimowy”, „lekko uogólniony”.
- **Nieprzemienność jest naturalna**: obroty w różnych płaszczyznach nie
  komutują, więc `r_1 o r_2 != r_2 o r_1` (czego DistMult — diagonalny, a więc
  przemienny — nie jest w stanie wyrazić; to jest fundamentalna przyczyna, dla
  której obecny operator nie nauczy się składania).
- `s_r` (log-skalarka) daje efekty typu „hiperonim zwęża pojęcie”,
  „antonimia zachowuje normę” — bez dodawania osobnych mechanizmów.

Warianty `DistMult` i `TransE` zostają jako baseline do ablacji (ten sam
interfejs, jedna flaga).

### 1.3 Operator zależny od kontekstu: `T_{r @ x}` (prawdziwe połączenie)

Relacja „negacja” czy „synonimia” działa inaczej na czasownikach, rzeczownikach
i w różnych domenach; w Słowosieci to samo `ant` łączy pary różnego typu.
Dlatego współczynniki operatora są **modyfikowane przez punkt zaczepienia**:

    theta(r, x) = theta_r + eps * P(relu(W x))        (W: K x D, P: K x M, eps << 1)

- `theta_r` — część „globalna” relacji (to, czego uczy się dziś).
- druga część — mała hiper-sieć (koszt `K*D + K*M` parametrów, `K ~ 32`)
  produkująca korektę kątów zależną od wektora wejściowego.
- `eps` wyuczone, start `0` — model sam decyduje, ile kontekstu potrzebuje
  (bez ryzyka zniszczenia stabilnego startu).
- Konsekwencja projektowa: `T_{r @ x}` przestaje być dokładnym obrotem, więc
  **cykl `T^{-1}(T(x))` przestaje być tożsamy** i staje się sensowną metryką
  oraz regularyzatorem (patrz 2.5).

To jest element, którego nie ma żadne znane mi KG-embeddingowe rozwiązanie:
transport zależny od punktu bazowego, z dokładnym (w granicy `eps -> 0`)
odwróceniem i z **naukową mierzalną miarą** jego niezupełności.

### 1.4 Aplikowalność i budżet dryfu

Operator musi umieć powiedzieć „tej relacji tu nie da się zastosować”
(negacja nazwy własnej, hiponim 6 poziomów w głąb). Dołączona głowica:

    a_r(x) ∈ [0, 1]      prawdopodobieństwo, że relacja r jest zdefiniowana w x
    b_r(x) ∈ [0, 1]      „budżet”: ile jeszcze transformacji zniesie otoczenie x

- Nadzór: krawędzie prawdziwe (1) vs pary niedozwolone wg typów z ontologii i
  losowe pary (0). Metryka: AUC + kalibracja (ECE).
- Zastosowanie w łańcuchu: `p(ścieżki) = prod a_{r_i}(x_{i-1})` — pipeline
  wyszukiwania wie, kiedy przestać ufać transformowanemu zapytaniu, i może
  wrócić do oryginalnego `t`. Bez tego feature'a produkuje „ciche” błędy,
  których nie da się odfiltrować.

### 1.5 Odwrotny problem: „jaka to relacja?” (dekoder relacji)

Głowica `g(x, y) -> (lambda_1..lambda_num_rel, confidence)` odzyskuje współrzędne
transportu z `x` do `y`. Daje to trzy możliwości produktowe, dzisiaj niedostępne:

- **automatyczne przerabianie zapytania**: zamiast ręcznego wybierania `r`,
  model podpowiada, która relacja mapuje zapytanie użytkownika na zbiór trafień;
- **uzupełnianie grafu**: zaproponuj krawędź (relacje) między dwoma synsetami;
- **arytmetyka relacji**: `r_a o r_b^{-1}` jako nowy operator (np.
  „przeciwieństwo do hiponimii”), z walidacją na podzbiorze gold.

---

## 2. Funkcja straty (osiem składowych, cztery etapy)

Niech `x_v = Refined(base_v)` (widok węzła w przestrzeni operatora),
`pi : D -> D_out` projekcja do przestrzeni embedera/retrievera.

### 2.1 Wierność transportu (krawędź, k=1)

    L_edge = sum_(A,r,B) [ w_d * (1 - cos(T_r(x_A), x_B)) + w_g * BCE(a_r(x_A), 1) ]

Zastępuje obecna parę „ranking + cosine + MSE” dla k=1; ranking zostaje w
postaci kontrastu (2.4), bo to on optymalizuje to, co mierzymy w retrieverze.

### 2.2 Składanie ścieżek (k = 2..K_max) — nadzór wprost

Losowanie ścieżek z grafu (bez powrotu do wierzchołka poprzedniego, bez tej samej
relacji dwa razy z rzędu, ograniczone głębokością `K_max = 4`):

    L_path = sum_{{A,r_1..r_k,B}} 1 - cos( T_{r_k} o ... o T_{r_1}(x_A), x_B )

Wariant „compiled”: ten sam cel liczony na operatorze skompilowanym (patrz 3.3),
żeby ścieżka była równoważna jednemu `matmul` także numerycznie.

### 2.3 Kara krzywizny / holonomii — serce metody

Dla każdej napotkanej czwórki `A -r-> C <-s-> B`, domykanej relacjami `p, q`:

    H = T_q^{-1} o T_p^{-1} o T_r o T_s        (obrócenie wokół A)
    L_curv = w_curv * || H(x_A) - x_A ||^2     (tylko dla czwórek domkniętych w grafie)

+ **wyuczona macierz komutacji** `C_{rs} ∈ [0,1]` (para relacji, `num_rel^2`)
  i jej miękkiej wersja dla par niedomkniętych (czwórki „cieniowe”, 2.6):

    L_commut = w_com * C_{rs} * || T_r(T_s(x)) - T_s(T_r(x)) ||^2
             + w_anti * (1 - C_{rs}) * margin( || T_r(T_s(x)) - T_s(T_r(x)) || )

Drugi składnik jest istotny: metodzie nie chodzi o to, żeby **wszystko** było
przemienne (to sprowadza do DistMult), tylko żeby model był **spójny**: relacje,
które komutuja (synonimia x negacja), komutuja; a które nie — niech nie komutuja
w sposób przypadkowy, lecz kontrolowany. `C` jest jednocześnie **artefaktem
interpretowalnym**: mapa przemienności relacji słownikowych.

### 2.4 Kontrast w przestrzeni retrievera (in-batch + false-negative masking)

    L_con = InfoNCE( pi(T_r(x_A)), { pi(x_j) }_{j in batch}, target = pi(x_B) )

Temperatura kalibrowana; maskowanie fałszywych negatywów na podstawie
sąsiedztwa w grafie (jeżeli `j` jest realnym celem relacji `r` od `A`, nie jest
negatywem) — to eliminuje główne źródło szumu obecnego `neg_cosine_weight`.

### 2.5 Spójność cyklu i izometryczność (stabilność wyszukiwania)

    L_cycle  = || T_{r @ x}^{-1}( T_{r @ x}(x) ) - x ||          (odwrócenie approximate)
    L_isom   = relu( |cos(u,v) - cos(T_r(u), T_r(v))| - delta )  (próbki par u,v)
    L_iso-norm = | ||T_r(x)|| / ||x|| - s_r |                    (kontrola normy wektora)

`L_isom` to formalizacja wymogu „po transformacji dalej mam sensowne cosine”:
operator może przesunąć znaczenie, ale nie może go rozmyć.

### 2.6 Manifold cieniowy (wektory spoza grafu) — walidacja produktu

Poszukujemy wektory prawdziwych tekstów z indeksu retrievera (nie sa węzłami
grafu) i traktujemy je jako **węzły cieniowe**:

- `L_shadow_cycle`, `L_shadow_commut` liczone na cieniach (spójność operatora
  poza grafem — to jest dokładnie test generalizacji na `t` użytkownika),
- `L_pull` — koherencja z projekcja: `T_r(pi(z)) ≈ pi(T_r(z))` (operator
  niemal przemienny z projekcja między przestrzeniami, inaczej eksport do
  bazy wektorowej psuje transformacje),
- `L_anchor` — węzeł po refiningu nie ucieka z okolicy swojej bazy:
  `1 - cos(x_v, base_v) < margin`, bo inaczej tracony jest sens startowy.

### 2.7 Wagowanie i stabilność

- grupowanie wag po etapach + **maska włączana etapami** (`w_* = 0` do startu etapu);
- normalizacja składników do rzędu wielkości (running EMA skali każdego lossa)
  — bez tego `L_curv` i `L_con` nie idą w parze;
- `L2`/`weight_decay` wyłączony dla kątów i `s_r` (niszczy strukturę obrotu),
  włączony dla `W`, `P` i głowic;
- gradient clipping per-grupa (operator vs GAT vs głowice);
- bf16 dozwolone, wyjątkowo FP32 dla kumulacji kątów (rotacje są wrażliwe).

---

## 3. Architektura i kod

### 3.1 Nowe moduły

    relgat_projector/core/operator/__init__.py
    relgat_projector/core/operator/base.py           # Protocol RelationalOperator
    relgat_projector/core/operator/givens.py         # WIĄZKA: Givens + skala + dial
    relgat_projector/core/operator/context.py        # hiper-sieć theta(r, x), eps
    relgat_projector/core/operator/compiled.py       # kompilacja ścieżki -> 1 operator
    relgat_projector/core/operator/builtin_distmult.py   # baseline
    relgat_projector/core/operator/builtin_transe.py     # baseline
    relgat_projector/core/geometry/givens_kernel.py  # fwd/inv, cuda kernel lub wektoryzacja
    relgat_projector/core/geometry/planes.py         # nauka zbioru płaszczyzn
    relgat_projector/core/sampler/paths.py           # random walk ścieżek k>=2
    relgat_projector/core/sampler/quadruples.py      # czwórki do krzywizny i komutacji
    relgat_projector/core/sampler/negatives.py       # degree-aware + type-aware + filtered
    relgat_projector/core/sampler/shadow.py          # węzły cieniowe z indeksu
    relgat_projector/core/loss/transport.py          # L_edge
    relgat_projector/core/loss/path.py               # L_path
    relgat_projector/core/loss/curvature.py          # L_curv, L_commut
    relgat_projector/core/loss/contrast.py           # InfoNCE + maskę
    relgat_projector/core/loss/stability.py          # L_cycle, L_isom, L_anchor
    relgat_projector/core/heads/applicability.py     # a_r(x), b_r(x)
    relgat_projector/core/heads/relation_decoder.py  # g(x, y) -> lambda
    relgat_projector/core/metrics/geometry.py        # HD, PCE, CCE, ISOT
    relgat_projector/core/metrics/offline.py         # OMD, QR-Recall, APL/ECE
    relgat_projector/trainer/stages/                 # stage1..stage4 + kurator wag
    relgat_projector/inference/relation_flow.py      # API: apply/compile/invert/decode

`RelGATModel` dostaje `operator_type: {"givens","distmult","transe"}` i deleguje
do `RelationalOperator`; `transform_from_vectors` zachowuje sygnaturę i staje
się cienkim wrapperem (bez łamania istniejących wywołań).

### 3.2 Zmiany w istniejących miejscach

- `core/model/model.py`: ustawić `self._config`; usunąć transformacje `dst`
  w scoringu; jeden kod ścieżki dla `forward` i `transform` (wspólny
  `apply_operator`); `single_gat_step` z residuum i `L_anchor`.
- `core/model/layer.py`: (a) **transport-aware agregacja** zamiast czystego
  attention: wiadomość `u -> v` to `T_{r}(proj(x_u))`, a nie `proj(x_u) * alpha`
    — GAT zaczyna być spójnym uogólnieniem transportu (faza 1 i 2 dzielą
    tę samą centralną operację); (b) `residual + LayerNorm`; (c) dropout tylko na
    wyjściu, nie na wagach attention.
- `dataset/relgat_dataset.py`: rozdzielić `topology_edges` (pełny graf, do
  GAT/transportu) od `supervised_edges` (podział train/eval); `Generator`
  lokalny zamiast `random.seed` globalnego; `num_workers > 0` albo wyłączyć
  `pin_memory`; próbki cieniowe ładowane strumieniowo.
- `dataset/edge.py`: usunąć `_precompute_negative_candidates`; negatywy
  `p(i) ~ deg(i)^{3/4}` (filtrowane po `a_r` i po sąsiedztwie relacji `r`),
  ewentualnie `p ~ 1/deg` jako wariant do ablacji; negatywy typu-tożsamego
  (ten sam POS/domena) jako trudne — w słownikach semantycznych to najczęstszy
  błąd modelu.
- `trainer/relgat_projector.py`: `upper_is_better` jako parametr; `best` po
  złożonej metryce modelu (MRR + PCE + AUC aplikacji); checkpoint „best”
  niezależnie od `save_every_n_steps`; naprawić `int(None)` dla
  `early_stop_patience`; Adam -> AdamW z grupami wag.
- `core/loss/multi_objective_loss.py`: zostaw jako ścieżkę „legacy” (flaga),
  nowe starty idą przez `trainer/stages`.

### 3.3 Kompilacja ścieżek (feature produktowy)

`compile_path([r_1..r_k])` łączy łańcuchy obrotów i **redukuje** z powrotem do
`M` obrotów (najmniejsze kwadraty na bazie/płaszczyznach, jeden `lstsq`
`M x M`, lub po prostu trzymanie łańcucha — i tak O(kM)). Wynik: jeden operator.

Efekt: w produkcji `t -> r_i..r_k -> t'` to jeden `matmul` + jedna skalarka,
bez wywołań do grafu i bez przechodzenia po węzłach pośrednich. Łatwo
pamiętać podłączenia `C` (przemienne relacje można kanonicznie posortować,
ściąć łańcuch i cache'ować) — to zamienia koszt długich łańcuchów w O(1).

### 3.4 Interfejs docelowy

```python
conn = RelationalConnection.load(ckpt)              # łącznie z rel2idx i projekcją
v  = conn.refine(base_vec)                          # (1) ulepszona reprezentacja
v2 = conn.apply(v, rel_id=ID_SYN)                   # (3) jeden krok
v3 = conn.apply_chain(v, [ID_NEG, ID_SYN])          # (3) łańcuch
op = conn.compile([ID_NEG, ID_SYN])                 # (3) skrócenie do 1 operatora
w4 = conn.apply(v, ID_HYP, lam=0.4)                 # (3) dial: „lekko uogólnij”
p  = conn.applicability(v, ID_NEG)                  # (4) czy to w ogóle ma sens
r_hat, conf = conn.decode(v, w3)                    # (5) „jaka to relacja?”
out = conn.to_retriever_space(w3)                   # eksport do bazy wektorowej
```

---

## 4. Etapy uczenia (curriculum) i kryteria przejścia

| Etap | Dane | Aktywne składowe | Kryterium przejścia |
|---|---|---|---|
| **F0 Naprawy** | — | — | zielone testy dymne, `save/load` round-trip, MRR > random |
| **F1 Węzły** | krawędzie (topologia) | `L_anchor`, `L_edge` (bez operatora wolnych: operator zamrożony do tożsamości) | cosine `x_v` vs `base_v` > 0.90 i MRR k=1 > baseline |
| **F2 Relacje (k=1)** | krawędzie + negatywy | `L_edge`, `L_con`, `L_isom`, `L_cycle` | PCE(k=1) <= PCE baseline; ISOT < 0.05 |
| **F3 Składanie** | ścieżki k=2..4 + czwórki | + `L_path`, `L_curv`, `L_commut` | przyrost PCE(k) < 0.10 na krok; HD < 0.15 |
| **F4 Cienie i kontekst** | wektory tekstów + `eps` on | + `L_shadow_*`, `L_pull`, aplikowalność, dekoder | OMD/MRR(węzeł) >= 0.85; APL AUC > 0.85, ECE < 0.08 |

Kolejność nie jest przypadkowa: `eps` (kontekst) włączamy na końcu, bo przy
kontekście `T^{-1}` nie jest dokładne i nie chcemy tego mieszać z uczeniem
samej struktury obrotu.

---

## 5. Metryki (część nowa — to też jest deliverable)

Klasyczne (zostają): `MRR`, `Hits@k`, `eval/loss`.

Nowe, geometryczne (bez nich nie da się udowodnić, że (3) działa):

- **PCE — Path Compilation Error**: `1 - cos( T_compiled(x), target )` dla
  `k = 1..4`; raportujemy **krzywa**, nie jedna liczbę. Kluczowa metryka
  całości projektu.
- **HD — Holonomy Defect**: średnie odchylenie transportu pętlowego od tożsamości na
  domkniętych czwórkach; łącznie z mapą `C_{rs}` jako artefakt interpretowalny.
- **CCE — Cycle Closure Error**: `||T^{-1}(T(x)) - x||` na węzłach i cieniach.
- **ISOT — Isometry Error**: `|cos(u,v) - cos(Tu, Tv)|` — miara „czy
  wyszukiwanie cosinusowe po transformacji wciąż ma sens”.
- **OMD — Off-Manifold Degradation**: `MRR_węzłów` / `MRR_textow_cieniowych`.
  Mierzy dokładnie to, co interesuje użytkownika: jak bardzo model gorszy się na
  prawdziwych tekstach niż na węzłach grafu.
- **APL / ECE**: jakość i kalibracja głowicy aplikowalności.
- **QR-Recall@k (query rewriting)**: dla par gold (np. `ant` ze Słowosieci,
  `synset_domain_topic`, plus zestaw oceniany przez LLM, żeby nie uczyć się na
  własnym modelu) — Recall po transformacji zapytania vs Recall bez transformacji
  i vs baseline „doklejenie słowa kluczowego”.

Zestaw ablacji (przeglądalny w jednym run-table):
`operator ∈ {givens, distmult, transe}` x `ścieżki k>=2 {on,off}` x
`L_curv {on,off}` x `kontekst eps {on,off}` x `cienie {on,off}`.
Bez `L_curv` spodziewamy się PCE(k) wyraźnie gorsze przy k>=2 — to jest hipoteza
główna, która ten plan testuje.

---

## 6. Potwierdzone ryzyka i środki zaradcze

1. **Rotacje mogą być „zbyt sztywne”** dla relacji nieodwracalnych w sensie
   semantycznym (hiponimia realnie zawęża zakres pojęcia, a nie obraca wektora).
   Środek zaradczy: `s_r` +
   mały składnik `low-rank` dopuszczony poza `O(D)` (`T = s_r Rot (I + U V^T)`,
   `rank(U) = 4`) — włączany tylko jeśli PCE(k=1) nie spada poniżej progu.
2. **Krzywizna wymaga czwórek**, a Słowosieć jest rzadka w czwórki.
   Środek zaradczy: czwórki cieniowe (2.3) + nauka `C_{rs}` zamiast twardego
   wymuszenia; mierzyć HD osobno na czwórkach prawdziwych i cieniowych.
3. **Kontekst (`eps`) psuje dokładne odwracanie.** Środek zaradczy: start `eps=0`,
   `L_cycle` jako miękka kara, walidacja CCE na każdym kroku F4.
4. **Osem składowych straty to za dużo na stabilne uczenie.** Środek zaradczy:
   etapy z maską wag, EMA-normalizacja składników, logowanie składników
   osobno, jeden seed i jedna zmienna na eksperyment (matryca ablacji).
5. **Brak gold do walidacji semantyki transformacji.** Środek zaradczy:
   `ant`/`syn`/`hyper` ze Słowosieci jako walidacja + opcjonalnie LLM-judge
   w trybie offline (nie w lossie), z osobnym raportem zgodności.
6. **Wyciek danych przez topologie.** Środek zaradczy: jawne rozdzielenie
   `topology_edges` / `supervised_edges` w Fazie 0 + raport o pokryciu
   (ile krawędzi ewaluacyjnych jest „widocznych” w sąsiedztwie).

---

## 7. Co jest nowe, a co nie (uczciwa deklaracja)

- **Nie jest nowe**: parametryzacja `O(D)` przez obroty Givensa (numeryka),
  obroty/płaszczyzny jako operatory relacji (RotatE i rodzina), InfoNCE,
  self-adversarial negative sampling, hypernetworki, flow matching.
- **Nowe — i to jest wartość planu**:
  1. traktowanie grafu semantycznego jako **połączenia z transportem zależnym
     od punktu bazowego** i budowa operatora relacji jako elementu grupy
     podobieństwa (obrót + skala) **dokładnie odwracalnego**;
  2. **kara holonomii/krzywizny + wyuczona macierz przemienności `C_{rs}`** jako
     sposób wymuszenia spójności składowania relacji — zamiast heurystycznego
     nadzorowania długich ścieżek;
  3. użycie tej spójności do **przenoszenia operatorów na wektory spoza grafu**
     (prawdziwe teksty) z mierzalną degradacją (OMD) — czyli produkt
     „zmień znaczenie wektora i szukaj”;
  4. **kompilacja łańcucha relacji do jednego operatora** (`O(1)` w produkcji)
     ze strukturalnym skracaniem przez `C_{rs}`;
  5. **dial gęstości relacji** (`lambda`) i **głowica aplikowalności/budżetu
     dryfu** jako element interfejsu, nie dodatek;
  6. **dekoder relacji** `g(x, y)` — odwrotny problem, umożliwiający
     automatyczne przerabianie zapytań.

Żaden z punktów 1-6 nie jest dziś dostępny w dostępnych metodach embeddingów
grafowych ani w tym repo; kombinacja 1+2+3 daje właśnie to, czego opis
użytkownika wymaga: `t' = f_{r_i..r_k}(t)` w tej samej przestrzeni, z gwarancją
geometrii cosine i możliwością weryfikacji.

---

## 8. Plan wykonania (kamienie milowe)

- **M0 (0.5 dnia)** — F0: naprawy blokujące + testy dymne + runda referencyjna
  (baseline `transe`/`distmult`, te same hiperparametry) jako baza odniesienia.
- **M1 (1 dzień)** — `RelationalOperator` + `givens.py` (fwd, inv, dial) +
  testy właściwości: `T^{-1}T = I`, `||T|| = s_r`, `ISOT = 0`, nieprzemienność
  na przykładach. Bez trenowania.
- **M2 (1.5 dnia)** — GAT transport-aware + `L_anchor` + rozdzielenie topologi;
  zamknięcie F1.
- **M3 (2 dni)** — sampler negatywów (degree/type/filtered), `L_edge`, `L_con`,
  `L_isom`, `L_cycle`; F2; pierwsza tabela wyników vs baseline.
- **M4 (2-3 dni)** — ścieżki `k>=2`, czwórki, `L_path`, `L_curv`, `L_commut`,
  `C_{rs}`, kompilacja ścieżek; F3; **krzywa PCE(k)** jako główny artefakt.
- **M5 (1-2 dni)** — cienie, `L_pull`, aplikowalność, `eps`/kontekst, dekoder;
  F4; OMD + APL + QR-Recall.
- **M6 (1 dzień)** — API inference, eksport, README, dokument `GEOMETRY.md`
  (metody + odczyt `C_{rs}`), czyszczenie ścieżek legacy.

Razem ok. 8-10 dni pracy. Każdy kamień milowy kończy się mierzalnym
artefaktem i uruchamialnym `training_scripts/run-*.sh`, więc można zatrzymać
się w dowolnym momencie bez utraty użyteczności.

---

## 9. Hipotezy do falsyfikacji

- **H1**: operator z grupy podobieństwa (obrót+skala) składa się wyraźnie
  lepiej niż diagonalny/translacyjny przy tym samym budżecie parametrów
  (PCE(k>=2) spada >= 25%).
- **H2**: `L_curv` redukuje degradację przy łączaniu relacji i **poprawia
  OMD** (generalizację na wektory spoza grafu), choć nie jest nadzorowany
  danymi z tekstów.
- **H3**: `C_{rs}` jest interpretowalna (pary typu synonimia x negacja blisko
  1, hiponimia x meronimia blisko 0) i pozwala skracać łańcuchy bez utraty
  PCE.
- **H4**: kontekst operatora (`eps`) pomaga tam, gdzie relacja jest
  wieloznaczna typologicznie (POS/domena), i jest widoczny w APL.
- **H5**: `L_anchor` + `L_isom` są konieczne, żeby MRR w przestrzeni retrievera
  (po `to_retriever_space`) nie byl gorszy niż w przestrzeni GAT.
