# 7. CI-gate dla modeli: relacyjny odcisk palca (regresja zachowania)

**Teza w jednym zdaniu:** współrzędne `θ_r`, macierz `C_rs` i rozkład
`applicability` tworzą **odcisk semantyczny** przestrzeni — dzięki czemu zmiana
embeddera, kwantyzacja albo agresywny fine-tune są wykrywalne jako regresja
**znaczenia**, a nie jako przesunięcie rozkładu, którego nikt nie umie
interpretować.

**Prymitywy:** P6 `C_rs` · P4 `decode` · P5 `applicability` · P3 dial · P7 `OMD`
**Wysiłek:** 1,5 tygodnia · **Zależności:** działa na samym operatorze — bez F4
**Status:** najmniej ryzykowny pomysł w całym zestawie, największa wartość
operacyjna na złotówkę pracy

---

## 1. Problem

Kto utrzymuje produkcyjny retrieval z wieloma indeksami, zna cztery awarie:

1. **nowa wersja embedera** — indeks zbudowany starym, zapytania nowym; cosine
   dalej „coś" zwraca, trafności spadają o 15%, nikt nie dostaje alarmu;
2. **checkpoint skwantyzowany / sprowadzony do fp16** — metryki uśrednione
   wyglądają identycznie, a rzadkie zapytania się sypią;
3. **fine-tune embedera** pod jeden przypadek użycia psie inne;
4. **zmiana ontologii** (nowa Słowosieć, nowe relacje) przesuwa znaczenie
   relacji, ale nie zmienia wymiaru wektora, więc żadna kontrola typu „sprawdź
   kształt tensora" tego nie widzi.

Obecne narzędzia: monitorowanie średnich, MMD, PSI, Procrustes distance między
próbkami starych i nowych wektorów. Wszystkie mają tę samą wadę: **mierzą
przesunięcie rozkładu, a nie zmianę struktury relacji**, i — co gorsza — nie
dadzą się zinterpretować. Alarm „MMD wzrosło o 0.03" nikomu nie mówi, co się
zepsuło i czy to ma znaczenie.

## 2. Mechanizm

Trzy warstwy odcisku, od najtańszej do najdroższej:

**Warstwa 1 — współrzędne relacji `θ_r` (P1, P6).** Tensor `[R, K]` +
`log_scale [R]` + baza `B [K, D, D]`. Dwie przestrzenie z tym samym grafem mają
porównywalne `θ` **tylko po dopasowaniu bazy** — więc nie porównujemy `θ`
bezpośrednio, tylko **macierz odległości kątowych** między relacjami:

    D_Θ[i,j] = kątow(O(θ_i), O(θ_j))   ∈ [0, π]

To niezmiennicze względem obrotu bazy `B` (bo iloczyn skalarny w algebrze jest
niezmienniczy), więc **D_Θ jest odciskiem nie wymagającym wyrównania**. Zmiana
`‖D_Θ' − D_Θ‖` = zmiana *struktury relacji przestrzeni*.

**Warstwa 2 — `C_rs`.** Macierz przemienności. Przesunięcie pary relacji z
`C=0.9` na `C=0.4` znaczy, że przestrzeń zaczęła traktować dwie osy jako
zależne — to jest **zmiana modelu świata**, nie dryf numeryczny.

**Warstwa 3 — profil zadaniami (probe).** Stały zestaw `probe'ów`: pary o
znanej relacji (gold z grafu) + pary kontrolne (bez relacji). Po każdej zmianie
liczymy:

```python
flow = RelationFlow.load(ckpt_new)
rel_id, conf = flow.model.decode(P_a, P_b)
acc_probe  = (rel_id == gold).float().mean()          # „czy synonimia wciąż jest synonimią"
conf_delta = (conf - conf_baseline).abs().mean()
appl_shift = (flow.applicability(P_v, R_ids) - appl_baseline).abs().mean()
omd        = off_manifold_degradation(node_mrr, text_mrr)   # nowa przestrzeń vs indeks
```

Progi **per relacja**, nie globalne — bo praktycznie zawsze psuje się jedna
relacja (np. negacja), a średnia po wszystkich ją maskuje.

Dodatkowo: **sygnał niezgodności indeksu**. `OMD` (stosunek MRR na węzłach do
MRR na wektorach tekstu) rośnie gwałtownie, gdy zapytania idą przez inny
embeder niż ten, którym budowano indeks. To najtańszy detektor awarii nr 1 — i
jest już zaimplementowany (`core/metrics/offline.py`).

## 3. Algorytm bramy CI

```
wejscie: ckpt_new (lub katalog z nowymi wektorami), baseline_ckpt
1. zaladuj oba operatory + C_rs (load_commutation)
2. warstwa 1: D_Θ_old vs D_Θ_new  → Δ_Dtheta
3. warstwa 2: C_old vs C_new      → Δ_Crs  (per para relacji)
4. warstwa 3: probe'y              → acc_probe, conf_delta, appl_shift, OMD
5. per relacja r:
     alarm_jezeli acc_probe[r] < prog_acc[r] - delta_tol
6. raport: tabela per relacja + lista par C_rs, ktore sie rozjechaly
   (to jest gotowa lista „co sie zepsulo" dla czlowieka)
```

Kluczowa cecha użytkowa: **raport jest czytelny**. Nie „wykryto dryf", tylko
„relacja `ant` straciła 12 pkt trafności dekodera; para (`hyper`,`hypo`)
przeszła z `C=0.88` na `C=0.41`". Inżynier wie, gdzie iść.

## 4. Metryki (ewaluacja samej metody)

Sondujemy 10 typów wstrzykniętych regresji:

| wstrzyknięta regresja | oczekiwane zachowanie odcisku |
|---|---|
| wymiana embedera na inny model | `OMD` i `acc_probe` mocno w dół |
| kwantyzacja int8 | `Δ_Dtheta` małe, `appl_shift` średnie |
| fine-tune pod jeden domain | `acc_probe` spada na relacjach niezwiązanych |
| zmiana wersji Słowosieci | `Δ_Crs` duże, `acc_probe` stabilne |
| zamiana indeksu (stare wektory, nowe zapytania) | `OMD` wybuch, reszta bez zmian |
| shuffle wektorów (sanity) | wszystko w dół |
| losowa perturbacja embeddingów (kontrola fałszywych alarmów) | brak alarmu |

Metryka: **AUC wykrycia per typ regresji** oraz **FAF** (false alarms per
tydzień) przy progu dającym AUC 0.9. Baseline: MMD / PSI / Procrustes distance
— porównanie przy **wyrównanym FAF**, bo inaczej baseline wygra przez
nadwrażliwość.

## 5. Fazy

| faza | czas | artefakt |
|---|---|---|
| 7a | 2 dni | definicja `D_Θ` + probe set (500 par gold + 500 kontrolnych) |
| 7b | 3 dni | wstrzykiwarka regresji (10 typów) + rejestr wyników |
| 7c | 3 dni | brama CI + raport czytelny (tabela per relacja) |
| 7d | 2 dni | dashboard / alerty; integracja z istniejącym `report.json` |

## 6. Ryzyka

| ryzyko | mitygacja |
|---|---|
| `θ` nie da się porównywać bez dopasowania bazy | dlatego używamy macierzy odległości kątowych `D_Θ` (niezmiennik), a nie surowych `θ` |
| probe set jest za mały, by wykrywać rzadkie relacje | dobrać próg liczności: relacja z < 20 probe'ów raportowana jako „brak danych", nie jako „OK" |
| fałszywe alarmy na szumie seed'a | baseline dwóch przebiegów **bez zmian** = wyznaczenie poziomu szumu; progi ponad szum |
| koszt w CI | warstwy 1–2 to operacje na macierzach `[R,K]`, `[R,R]` — sekundy; warstwa 3 to 1000 par przez MLP |

## 7. Test go/no-go

Trzy przestrzenie: obecna, druga wersja embedera, ta sama przestrzeń z innym
seedem treningowym. Jeśli odcisk rozdzieli „inny embeder" od „inny seed"
(AUC > 0.9), pomysł działa; jeśli nie, odcisk mierzy szum, nie semantykę.
Pół dnia roboty.

## 7b. Pomiar na operatorze (sekcja 07, 2026-10-30)

Sekcja **07** w `operator_report`
(`relgat_projector/core/diagnostics/fingerprint.py`). Uruchomienie:
`python -m relgat_projector_apps.diagnostics.operator_report --ckpt
wiazka-models/S14_20261029_210326/best --baseline
wiazka-models/S13_20261029_162050/best --section 07`.

Najpierw dwie własności, bez których brama nie ma sensu:

* **samodyf = dokładnie `0.0`** — odcisk jest wyznaczalny powtarzalnie;
* **kąt średni między relacjami 0.346 rad**, `D_Θ` liczona na macierzach
  obrotu, więc niezmiennicza względem zmiany bazy `B` — sprawdzonym
  testem (`test_theta_angle_matrix_is_invariant_to_a_change_of_basis`),
  bo na tym niezmienniku stoi cała warstwa 1.

| wstrzyknięta regresja | `Δ D_Θ` |
|---|---|
| `theta` × 1.01 | 0.1047 |
| `theta` × 1.05 | 0.5190 |
| `theta` × 1.20 | 2.0151 |
| kwantyzacja fp16 | **0.0007** |
| inny przebieg tego samego przepisu (S13 → S14) | **0.7980** (`d_C` = 3.3779) |

Odcisk jest **monotoniczny względem wielkości zmiany** i **głuchy na
fp16** (0.0007, czyli ~150× poniżej nawet 1% drgania) — dokładnie takie
zachowanie chce się w CI: zmiana kwantyzacji nie alarmuje, zmiana algebry
relacji alarmuje. Między tymi dwoma poziomami mieści się też realna różnica
dwóch przebiegów (0.798 = 5,7% średniego kąta), więc próg da się ustawić
powyżej szumu przebiegu i poniżej regresji.

Warstwa 3 (trafność dekodera per relacja) wymaga uczciwej etykiety: probe
to **krawędzie treningowe** checkpointu, macro **0.981** — dla odcisku to
właściwy wybór (ma być własnością *tego* modelu, nie błędem generalizacji),
ale nie wolno go czytać jako 0.476 z `report.json` treningowego. S13 → S14:
** ani jedna relacja nie pogorszyła się** o więcej niż 0.05, mimo `d_C` = 3.38.
"`C_rs` się przesunęło, a zachowanie nie” — to rozróżnienie jest dokładnie
tym, co §3 obiecuje inżynierowi.

Czego nie zmierzono: AUC wykrycia per typ regresji z §4 (potrzeba ≥2
przestrzeni z *różnymi embederami*) ani porównania z MMD/PSI przy
wyrównanym FAF. Do tego potrzeba drugiego embedera — jedna z dwóch rzeczy,
których temu pomysłowi najbardziej brakuje.

## 8. Punkty odniesienia

- *Verification of Machine Unlearning is Fragile* (2024) i *Towards Reliable
  Forgetting* (2025) — dowód, że **walidacja zmian w modelu** jest problemem
  otwartym w pokrewnej dziedzinie (unlearning), a w przypadku embederów w ogóle
  nieweneturowany.
- *Kernelized Activation Steering* (2026), *Depth-Wise Activation Steering for
  Honest Language Models* (2025) — pokaźna rodzina metod interwencji w modele,
  przy całkowitym braku narzędzia, które sprawdziłoby, **czy interwencja nie
  zmieniła reszty znaczeń**. Nasz odcisk jest dokładnie tym narzędziem.
- W repo: `core/operator/givens.py` (`theta`, `basis`, `get_relation_matrices`),
  `handlers/wiazka_checkpoint.py` (`load_commutation`),
  `core/metrics/offline.py` (`off_manifold_degradation`).
