# 3. Augmentacja wektorowa dla uczenia embedderów (darmowa superwizja z grafu)

**Teza w jednym zdaniu:** augmentacją kontrastową może być **sam operator**
`T_r(v)` — z konstrukcji izometria do skali, więc pary pozytywne mają
identyczną geometrię, a negatywy są trudne **znaczeniowo** (`ant`, łańcuchy
nieprzemienne), a nie losowo.

**Prymitywy:** P1 · P2 `ISOT` · P3 dial `λ` · P6 `C_rs`
**Wysiłek:** 2 tygodnie · **Zależności:** działaNie na samym operatorze —
**bez F4**, wystarczy `T_r`

---

## 1. Problem

Augmentacja kontrastowa dla modeli zdań opiera się na manipulacji **tekstem**:
tłumaczenie wsteczne, utrata tokenów, permutacja zdań, parafraza LLM. Każda z
nich ma wspólną wadę: **przestawia normę i rejestr wektora**, czyli deformuje
dokładnie tę wielkość, potem mierzymy cosine'em. Efekt: model uczy się
niezależności na augmentacji, a my i tak kalibrujemy próg podobienstwa osobno,
bo rozkład cosine po augmentacji „pływa".

Drugi problem: **trudne negatywy**. W InfoNCE losowy negatyw jest z definicji
łatwy, więc po kilku epokach gradient zanika; standardem jest przeszukiwanie
trudnych negatywów (ANN, mining) — drogie i ryzykujące fałszywe negatywy.

Dla języków o mniejszej zasobności (polski) dochodzi trzecia bolączka: pary
parafraz kosztują pieniądze albo generowanie LLM, a tańsze zasoby (Słowosiec)
są wykorzystywane wyłącznie do zadań leksykalnych.

## 2. Mechanizm

Mamy operator `T_r = s_r · exp(θ_rᵀ B)` z trzema własnościami, których nie da
żadna augmentacja tekstowa:

1. **Izometria do skali (P2)**: `cos(Tu, Tv) = cos(u, v)` dokładnie (błąd
   mierzony jako `ISOT`, w referencyjnym biegu ~0.058 i to jest **górna granica
   błędu numerycznego**, nie deformacja semantyczna). Zatem augmentacja nie
   zmienia struktury podobienstw — zmienia tylko to, co ma zmienić.
2. **Dial `λ` (P3)**: siła augmentacji jako parametr ciągły. `λ → 0` daje parę
   trywialną, `λ = 1` pełne przesunięcie. Krzywą `λ` można stroić jak
   temperature w InfoNCE.
3. **Negatyw znaczeniowo trudny**: `T_ant(v)` jest **blisko** w sensie tematu i
   **daleko** w sensie znaczenia — czyli definicją trudnego negatywu,
   wygenerowaną bez przeszukiwania bazy.

Konstrukcja partii dla `v`:

    pozytywy    : { T_syn(v, λ) , T_hyper(v, λ ≤ 0.4) }
    trudne neg. : { T_ant(v) }  ∪  { T_r T_s v : C_rs[r,s] < 0.1 }
    proste neg. : losowe v' z batcha (jak zwykle)

Druga rodzina negatywów jest subtelna i **nowa**: dla par relacji, które wg
`C_rs` nie komutują, `T_r T_s(v)` i `T_s T_r(v)` to dwa **różne** punkty tej
samej orbity — więc oba są poprawnymi, trudnymi negatywami dla siebie
nawzajem. `C_rs` działa więc jako **generator trudnych negatywów bez etykiet**.

## 3. Algorytm

**Wariant A — augmentacja (darmowe pary, bez dekodera i bez F4):**

```python
op = build_operator("givens", dim=D, num_rel=R, num_generators=16)
# ...wczyj operator z checkpointu...

def positives(v, rel_syn, lam=0.7):
    return op.apply(v, torch.full_like(v[:,0], rel_syn), v.new_full((v.shape[0],), lam))

def hard_negatives(v, rel_ant, C, k=4):
    # relacje niekomutujące z ant → przesunięcia „obok" znaczenia
    idx = torch.nonzero(C[rel_ant] < 0.1).flatten()[:k]
    outs = []
    for r in idx:
        outs.append(op.apply(v, v.new_full((v.shape[0],), int(r))))
    return torch.stack(outs, 1)          # [B, k, D]
```

Pary idą do InfoNCE dokładnie jak pary z tłumaczenia wstecznego. **Nie potrzebny
żaden dekoder tekstu** — pracujemy wyłącznie w przestrzeni wektorów, więc kosztem
dodatkowym jest jeden `matmul` D×D na przykład.

**Wariant B — regularizer na istniejący embeder (ciekawszy naukowo):**

Kara za niezgodność geometrii przestrzeni z geometrią ontologii:

    L_flat = Σ_{diament A}  ‖ H_embed(v_A) − v_A ‖² ,  H_embed = „transport po
    relacjach grafu wyznaczony przez podobieństwo w przestrzeni embedera"

Intuicja: przestrzeń ma być **płaska tam, gdzie płaska jest ontologia**. To
regularizer, nie nowa architektura — można go dopiąć do istniejącego przebiegu
jako składnik straty, startując z wagą 0.

## 4. Metryki

| metryka | baseline | komentarz |
|---|---|---|
| nDCG@10 / MRR na zadaniach retrieval (podzbiety PL-MTEB lub STS-PL) | tłumaczenie wsteczne, kontrast bez augmentacji | główna teza |
| **stabilność norm** | std(‖v‖) po augmentacji | back-tłumaczenie psuje, `T_r` nie — wykres obowiązkowy |
| **`ISOT` pary po augmentacji** | ~0 dla naszej, > 0 dla tekstowej | dowodzi, że augmentacja nie deformuje |
| krzywa trudnych negatywów | bez negatywów znaczeniowych | pokazuje wartość rodzin `C_rs` |
| kalibracja progu cosine | osobne strojenie vs jedno dla wszystkich | pokazuje zysk operacyjny |

**Test kontrolny (ważny):** przebieg z negatywami **losowymi** zamiast `C_rs`
przy tej samej liczbie negatywów. Jeśli różnica znika, działa sama augmentacja,
a nie trudne negatywy — i to też jest wynik wart odnotowania.

## 5. Fazy

| faza | czas | artefakt |
|---|---|---|
| 3a | 2 dni | sampler par z operatora + testy (orbita, `λ = 0` → tożsamość) |
| 3b | 3 dni | InfoNCE na małym embederze (miniLM wielojęzyczny lub równoważny) |
| 3c | 1 tydzień | ewaluacja + ablacje: bez `ant`, bez `C_rs`, losowe negatywy |
| 3d | 4 dni | wariant B (regularizer `L_flat`) — eksploracyjnie |

## 6. Ryzyka

| ryzyko | mitygacja |
|---|---|
| operator jest zły tam, gdzie `PCE(2)` rośnie → wadliwe negatywy z łańcuchów | dopuścić tylko relacje z `PCE(1)` i `PCE(2)` poniżej progu (metriki per relacja są już liczone w `evaluate`) |
| `T_syn(v)` może być za bliski `v`, więc pozytywy trywialne | stroić `λ`; raportować histogram `cos(v, T_syn(v, λ))` |
| uogólnianie operatora wytrenowanego na hasłach na wektory zdań | w tym wariancie to **zaleta**, nie wada: uczymy embeder, więc operator działa jako nauczyciel; mimo wszystko zacząć od zdań krótkich i mierzyć `OMD` |
| wariant B może pogorszyć zadania spoza ontologii | waga startowa 0 + walidacja na zadaniach nieleksykalnych |

## 7. Test go/no-go

Weź 1000 wektorów zdań, zrób augmentację `T_syn(λ=0.7)` i **policz rozkład
cosine para–para** w porównaniu z augmentacją tłumaczeniem wstecznym (dostępną
za darmo dla 200 zdań). Jeśli nasza augmentacja daje **węższy** rozkład (mniejszą
wariancję) przy porównywalnym średnim przesunięciu — teza o niepsuciu geometrii
jest udowodniona za jeden dzień pracy.

## 8. Punkty odniesienia

- SemDeDup (2023) — ten sam kontekst, inne podejście (usuwanie zamiast
  wzmacniania).
- *Is Cosine-Similarity of Embeddings Really About Similarity?* — argument, że
  deformacja geometrii przez augmentację jest problemem realnym.
- W repo: `core/sampler/negatives.py` (stopniowo-aware negatywy, punkt wyjścia),
  `core/loss/contrast.py` (InfoNCE z maską fałszywych negatywów),
  `core/metrics/geometry.py` (`isometry_error`, `pce_curve`).
