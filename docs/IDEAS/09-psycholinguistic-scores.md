# 9. Surowce psycholingwistyczne z pokrętła: abstrakcyjność, wieloznaczność, typowość

**Teza w jednym zdaniu:** trzy wielkości, których psycholingwistyka szuka
ręcznym ankietowaniem (abstrakcyjność, wieloznaczność, typowość), dają się
**zdefiniować operacyjnie** przez liczbę kroków transportu, rozrzut
`applicability` i lokalną krzywiznę — a więc policzyć dla **każdego** słowa i
**każdego** zdania w każdym języku, w którym jest graf, bez ani jednej etykiety.

**Prymitywy:** P3 dial `λ` · P5 `applicability` · P7 holonomia/krzywizna
**Wysiłek:** 2 tygodnie · **Zależności:** działa na operatorze; walidacja chce
istniejących norm ludzkich (PL/EN)

---

## 1. Problem

Normy abstrakcyjności (Brysbaek i in. i nastepce) to fundament badań nad
czytelnością, upraszczaniem tekstów, przekładem, a także marker w badaniach
klinicznych (schizofrenia, afazja, spektrum autyzmu — przesunięcie w stronę
abstrakcyjności jest mierzalnym sygnałem). Problem: to **ręczne ankiety**,
kilkanaście tysięcy słów, język po języku, koszt lat pracy, a dla polszczyzny
pokrycie jest skąpe.

Normy wieloznaczności są jeszcze gorsze: zależą od słownika (ile znaczeń?), są
skokowe i nieporównywalne między słownikami. Nie ma dobrej **ciągłej** miary.

Typowości (czy dany egzemplarz jest typowy dla kategorii) w praktyce się nie
mierzy — się po niej zakładamy.

## 2. Definicje operacyjne

### 2.1 Abstrakcyjność = głębokość do nasycenia

Definiujemy `A(x)` jako liczbę kroków `hyper`, po których transport przestaje
**zmieniać sąsiedztwo** — czyli dalsze uogólnianie nie niesie informacji:

    A(x) = min { n : cos( T_hyper(x_n), centroid_top ) > 1 − ε }

Realizacja praktyczna (bez progu): **krzywa nasycenia**

    s(n) = cos( T_hyperⁿ(x), c_top )        rosnie do 1

i `A(x) = s⁻¹(0.95)` — interpolacja na krzywej, wartość **ciagła**, nie liczba
całkowita. To ważne: psycholingwistyka potrzebuje skali ciągłej.

Wariant z dial`em` (ładniejszy matematycznie): zamiast iteracji `T_hyperⁿ`
liczymy `T_hyper(λ)` dla `λ → λ_max` i badamy, dla jakiego `λ` zmiana
`cos(v, T_hyper(λ)v)` przestaje rosnąć. `λ` jest tu **ciąglym odpowiednikiem
liczby kroków**, więc obie definicje dają się porównać i uśrednić.

Dlaczego to jest sensowne, a nie arbitralne: `hyper` **definiuje** porządek
konkret → abstrakt w danym języku. Nie przybliżamy abstrakcyjności — definiujemy
ją przez relację, która nią jest.

### 2.2 Wieloznaczność = rozrzut typowania + lokalna krzywizna

    POLY(x) = H(applicability(x, ·))          // entropia rozkladu „które relacje sa zdefiniowane"
            + β · HD_lokalne(x)               // czy transport zalezy od drogi

Uzasadnienie: słowo wieloznaczne ma **wiele ramek** naraz, więc (a) wiele relacji
jest częściowo zdefiniowanych (rozrzut `applicability`), oraz (b) dwie drogi
transportu prowadzą do **różnych znaczeń**, więc holonomia nie znika. To druga
część jest ciekawsza i nigdzie indziej niedostępna: wieloznaczność mierzona jako
**niezerowa krzywizna przestrzeni znaczeń**.

### 2.3 Typowość = odległość od środka orbity kategorii

    TYP(x) = cos( x, centroid( orbity kategorii(x) ) )

gdzie kategoria to `hyper(x)`. Daje odpowiedź „czy ten egzemplarz jest typowy
dla swojej klasy" bez trenowania czegokolwiek.

## 3. Algorytm

```python
flow = RelationFlow.load(ckpt, device="cuda")
ID_HYPER = flow.rel2idx["hyper"]

def abstraction(v):                       # v: [N, D]
    cur, scores = v.clone(), []
    for n in range(NMAX):
        cur = flow.apply(cur, ID_HYPER)
        scores.append(cos(cur, TOP_CENTROID, dim=-1))     # [N]
    return saturate_point(torch.stack(scores))            # interpolacja -> [N] ciagle

def polysemy(v):
    a = torch.stack([flow.applicability(v, r) for r in range(R)], -1)
    ent = -(a.clamp(1e-9).log() * a).sum(-1) / log(R)
    return ent + BETA * local_curvature(flow, v)          # [N]

def typicality(v, cat_ids):
    return cos(v, centroid_by_category(v, cat_ids), dim=-1)
```

Koszt: `NMAX ≈ 8` aplikacji macierzy `D×D` na cały słownik — sekundy dla
kilkuset tysięcy jednostek. Dla **zdań**: to samo na wektorach zdań (uwaga:
wymaga F4 na zdaniach, inaczej `hyper` na zdaniu nie znaczy tego, co ma znaczyć).

## 4. Walidacja — i tu jest cała wartość

| metryka | gold | cel |
|---|---|---|
| Spearman z normami abstrakcyjnosci | Brysbaek-style (EN), analogie PL jeeli dostępne | ≥ 0.6 — tyle, ile wynosi internal consistency samych norm |
| Spearman dla **rzadkich** słów | podział na kwartyle częstości | pokazać, że nie odtwarzamy częstości, tylko znaczenie |
| Spearman z normami wieloznaczności | WordNet/PL WordNet sense count + normy | ciągła miara > liczenie znaczeń |
| typowość vs ocena ludzka | 200 par (kategoria, egzemplarz) | ≥ 0.6 |

**Kontrola antytrywialnosci (obowiązkowa):** porównać z baseline'em „długość
słowa", „częstość", „liczba współwystąpień". Jeśli nasza miara koreluje z normami
**słabiej niż częstość**, nie mierzy nic poza częstością i artykuł się broni.

**Test przesądzający:** our `A(x)` na słowach, które normy ludzkie **rozstrzygają
spornie** (duża wariancja między ankietowanymi). Jeśli nasza miara też daje
wartości pośrednie — łapiemy realną nieostrość semantyczną. Jeśli daje wartości
pewne — miara jest płytka.

## 5. Zastosowania downstream (to jest „wynik, nie wyszukiwanie")

1. **Sterownik czytelności**: „przepisz na poziom X" z **mierzalnym celem**
   (`A(text) ≤ t`) zamiast prompta „prościej". Różnica: po przepisaniu da się
   udowodnić, że cel został osiągnięty.
2. **Upraszczanie tekstów medycznych i urzędowych** — z raportem przed/po.
3. **Przesiew w badaniach klinicznych** — tani, ciągły, językowo niezależny
   wskaźnik abstrakcyjności zamiast ręcznych norm.
4. **Kontrola stylu odpowiedzi LLM** — pętla zwrotna: mierz `A(output)` i karz za
   przekroczenie celu w RLHF/DPO.
5. **Diagnostyka tłumaczenia** — porównanie `A(źródło)` vs `A(przekład)`:
   spadek = tłumaczenie spłyca.

## 6. Fazy

| faza | czas | artefakt |
|---|---|---|
| 9a | 1 dzień | `abstraction()` + histogram na całym słowniku (sanity: czy rozkład nie jest degenerate) |
| 9b | 3 dni | `polysemy()`, `typicality()` + kontrole antytrywialności |
| 9c | 1 tydzień | walidacja na normach EN (+ PL jeśli dostępne) |
| 9d | 4 dni | sterownik czytelności na 50 tekstach + ewaluacja ludzka |

## 7. Ryzyka

| ryzyko | mitygacja |
|---|---|
| `hyper` w Słowosieci jest rzadkie / urywa się | policzyć rozkład długości ścieżek `hyper` **przed** pracą; jeśli mediana < 3, `A(x)` będzie miało sufit — zgłosić to wprost |
| miara odtwarza częstość słowa | kontrola antytrywialności + regresja z częstością jako kontrolowaną |
| `λ_max` i `ε` arbitralne | strojenie na gold, raport wrażliwości, a nie „dostroenie do ładnego wyniku" |
| zdaniowa wersja wymaga F4 na zdaniach | zacząć od leksykalnej (to i tak nowość dla PL) |

## 8. Test go/no-go

Rozkład `A(x)` na całym słowniku + Spearman z normami EN (tłumaczenie przez
mapowanie PL↔EN z pomysłu 8 albo osobny słownik). Jeśli korelacja < 0.35 przy
kontroli częstości > 0.5 — pomysł nie ma wartości publikacyjnej, ale może mieć
użytkową (sterownik czytelności działa też na mierze własnej).

## 9. Punkty odniesienia

- Brysbaek i in. — normy konkretności; punkt odniesienia i gold.
- *A Grounded Typology of Word Classes* (2024) — dowod, że typologia leksemów
  przez znaczenie jest nadal aktywnym tematem.
- W repo: `core/operator/givens.py` (`apply` z `lambdas` — dial jako krok
  ciągły), `core/heads/applicability.py`, `core/metrics/geometry.py`
  (`holonomy_defect` → lokalna krzywizna).
