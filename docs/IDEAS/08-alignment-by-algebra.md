# 8. Wyrównanie dwóch przestrzeni bez słownika — przez sygnaturę algebry

**Teza w jednym zdaniu:** dwa pojęcia z dwóch **nieporównywalnych** przestrzeni
(dwa języki, dwa embedery, dwie ontologie) oznaczają to samo, jeżeli ich
**zachowanie transportowe** jest zgodne — a to daje się zmierzyć bez wspólnego
ziarna słownikowego, bo zmienniczą jest algebra, nie współrzędne.

**Prymitywy:** P6 `C_rs` · P4 `decode` · P1 `T_r` · P5 `applicability`
**Wysiłek:** 3 tygodnie · **Zależności:** **drugi graf** (druga ontologia) — bez
niego pomysł nie istnieje

---

## 1. Problem

Porównanie dwóch przestrzeni wektorowych to zawsze problem mapowania. Dwie
szkoły:

1. **Procrustes na ziarnie** (VecMap i rodzina): potrzebny słownik
   startowy `~5k–50k` par, potem iteracyjne dosztukowywanie. Dla języków bez
   dobrego słownika porównawczego albo dla **dwóch ontologii tej samej
   dziedziny** ziarno nie istnieje.
2. **Bez nadzoru, przez kształt rozkładu** (Sinkhorn/OT): działa dla języków o
   podobnej typologii i wymaga założenia, że oba rozkłady są „takie same z
   kształtu". Znane przypadki porażki na językach fleksyjnych — a polszczyzna
   taka jest.

W obu przypadkach dopasowujemy **geometrię**. Tymczasem to, co naprawdę
znaczy pojęcie w grafie relacyjnym, to nie jego współrzędne, tylko **sposób, w
 jaki reaguje na relacje**: jak wygląda `T_hyper(pojęcie)`, czy `ant` jest w
ogóle zdefiniowane, czy para relacji komutuje. Ta informacja jest
**niezmiennicza względem dowolnego odwracalnego przekształcenia liniowego**
przestrzeni — więc odporna na dokładnie ten problem, który blokuje obie szkoły.

## 2. Mechanizm i formalizacja

Dla pojęcia `x` w przestrzeni z operatorem `T` definiujemy **sygnaturę
transportową**:

    sig(x) = [
      (cos(T_r x, x))_r            // jak bardzo każda relacja go „rusza"
      (argmax_r cos(T_r x, y))_y∈N // struktura najbliższych względem relacji
      C_rs[r, ·] ważony udziałem r w sąsiedztwie x
      (applicability(x, r))_r      // typowanie: co jest tu zdefiniowane
    ]

Kluczowe: `sig(x)` jest **niezmiennicza względem obrotu** `x → Qx`, o ile
operator jest transformowany zgodnie (`T → Q T Q⁻¹`), a tak właśnie zmienia się
operator przy zmianie bazy. Nie musimy więc **znać** przekształcenia między
przestrzeniami, żeby porównywać sygnatury.

Wyrównanie = problem przypisania między dwoma zbiorami sygnatur:

    min_{π}  Σ_i  d( sig_I(x_i), sig_II(π(x_i)) )

rozwiązywany algorytmem Węgra / Sinkhornem na macierzy kosztów. Bez gradientów,
bez treningu, bez ziarna.

Drugi, mocniejszy wariant: **wyrównanie relacji, nie pojęć** — dopasowujemy
permutację słowników relacji między dwiema ontologiami (która relacja w PL
odpowiada której w EN / w SNOMED). To samo narzędzie, inny zbiór: macierz
kosztów liczona na `θ_r` i `C_rs`. Produkt: mapa `rel_PL → rel_EN`, z
wskaźnikiem pewności i — co cenne — z klasą „**brak odpowiednika**", gdy
żadna sygnatura nie pasuje.

## 3. Algorytm

```python
flowA = RelationFlow.load(ckpt_pl); flowB = RelationFlow.load(ckpt_en)

def signature(flow, V, rel_ids, k=16):
    rows = []
    for r in rel_ids:
        rows.append(cos(flow.apply(V, r), V, dim=-1))          # „ruszenie"
    rows.append(applicability_matrix(flow, V, rel_ids))        # P5
    nb = ann_neighbors(V, k)                                   # struktura lokalna
    rows.append(relation_profile_over_neighbors(flow, V, nb))  # P4 na sąsiadach
    return normalize(torch.stack(rows, -1))                    # [N, F]

cost = cdist(signature(flowA, A), signature(flowB, B))         # [N, M]
pi   = hungarian(cost)  # albo sinkhorn(cost, eps) dla miękkiemu przypisaniu
```

Prostowanie skali: `sig` musi być niezależna od globalnej skali wektorów —
normalizujemy każdy blok osobno i raportujemy wrażliwość wyniku na dobór wag
bloków (to jest nasza główna niewiadoma inżynierska, nie naukowa).

## 4. Metryki

| metryka | definicja | baseline |
|---|---|---|
| **top-1 / top-k accuracy bez ziarna** | zgodność przypisania ze „złotym" słownikiem trzymanym w rezerwie | VecMap bez ziarna, Sinkhorn na surowych wektorach |
| **krzywa: ile ziarna potrzebujemy** | accuracy vs liczba użytych par (0, 100, 1k) — nasza powinna wystartować wysoko przy zerze | to główny rysunek |
| F1 mapowania relacji | dopasowanie słowników relacji | dopasowanie po nazwach (string), po histogramie częstości |
| precision klasy „brak odpowiednika" | wśród ogłoszonych luk: ile faktycznie brak | nowa metryka; gold ręczny na 200 przypadkach |
| odporność na skalowanie | accuracy po losowym skalowaniu jednej przestrzeni ×3 | powinna spaść do zera u baseline'ów, u nas nie |

## 5. Warunki brzegowe i uczciwe zastrzeżenie

Pomysł wymaga, by **obie** przestrzenie miały własny graf i własny wytrenowany
operator. Bez drugiej ontologii nie ma czego porównywać — to nie jest metoda na
„dowolne dwa embedery", tylko na „dwa **opisane relacyjnie** zbiory". Trzeba
to napisać wprost, bo inaczej brzmi jak uniwersalne tłumaczenie bez słowników,
czego nie jesteśmy w stanie obiecać.

Realistyczny zakres: PL↔EN (Słowosiec ↔ WordNet, dwa gotowe grafy), dwie wersje
tej samej ontologii (migracja), ontologia firmowa ↔ standard branżowy (SNOMED,
GPRD, ISO-22232 dla kompetencji). Ten trzeci przypadek jest najbardziej dochodowy
i najmniej „publikowalny".

## 6. Fazy

| faza | czas | artefakt |
|---|---|---|
| 8a | 2 dni | dwa treningi operatora (PL, EN) — ten sam kod, dwa grafy |
| 8b | 3 dni | `signature()` + macierz kosztów + Węgier |
| 8c | 1 tydzień | ewaluacja na trzymanym ziarnie PL↔EN + baseline'y |
| 8d | 4 dni | mapowanie relacji + klasa „brak odpowiednika" + raport |

## 7. Ryzyka

| ryzyko | mitygacja |
|---|---|
| sygnatura jest zbyt uboga, by rozróżnić pojęcia bliskie | dodać blok „relacje w sąsiedztwie k=N"; mierzyć separability (silhouette) **przed** przypisaniem |
| różne gęstości grafów (PL rzadszy) d różne skale sygnatur | normalizacja kwantylowa per blok; raport wrażliwości |
| Węgier jest O(N³) | Sinkhorn + wstępne blokowanie ANN |
| „brak odpowiednika" wychodzi zawsze albo nigdy | kalibracja progu na walidacji; raportować krzywą precyzji |

## 8. Test go/no-go

1000 pojęć PL i 1000 EN, sygnatury, macierz kosztów, **bez żadnego ziarna**.
Top-1 accuracy > 15% (przy 1000 klasach losowo daje 0.1%) dowodzi, że sygnatura
niesie informację. Pół dnia po zbudowaniu drugiego operatora.

## 9. Punkty odniesienia

- VecMap i rodzina Procrustes-with-seed — baseline, którego chcemy uniknąć.
- Metody OT/Sinkhorn bez nadzoru — baseline nr 2 i narzędzie, które pożyczymy do
  samego przypisania.
- W repo: `core/operator/givens.py` (`theta`, `basis`, `apply`),
  `core/heads/applicability.py`, `core/sampler/paths.py` (profil relacji w
  sąsiedztwie).
