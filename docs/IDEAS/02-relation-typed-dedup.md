# 2. Deduplikacja danych treningowych, która nie zjada zaprzeczeń

**Teza w jednym zdaniu:** deduplikacja semantyczna po progu cosine usuwa razem
z duplikatami **pary kontrastowe** („X działa" / „X nie działa"), czyli dokładnie
te dane, które model uczy obsługi negacji i sprzeczności — typowanie relacją
zamiast progiem naprawia to bez zwiększania budżetu tokenów.

**Prymitywy:** P4 `decode` · P1 odwracalność · P6 `C_rs`
**Wysiłek:** 2–3 tygodnie (z jednym pełnym przebiegiem fine-tuningu)
**Zależności:** F4 na zdaniach; dostęp do GPU na 2 porównywalne przebiegi

---

## 1. Problem

SemDeDup i cała rodzina „dedup semantyczny" działa tak: embeduj, sklasteryzuj
po progu cosine, zostaw jednego reprezentanta. Zysk jest realny (mniej danych,
ten sam efekt), ale cena też: **próg cosine nie odróżnia parafrazy od
zaprzeczenia**. „Lek jest skuteczny" i „Lek nie jest skuteczny" mają wysokie
cosine, więc dedup zostawia jedną z nich — zwykle pierwszą.

Efekt jest dokładnie odwrotny od zamierzonego: korpus po deduplikacji jest
**ubogi w kontrast**, a to kontrast uczy model rozróżniać bieguny. Zjawisko
jest dobrze znane jako „wycięte negacje", ale istniejące poprawki celują w
inny problem — `FairDeDup` (2024) dotyczy rozkładu sprawiedliwościowego w
deduplikacji wizualno-językowej, nie obslugi negacji.

To samo dotyczy deduplikacji **łańcuchów rozumowania** (CoT) i par preferencji:
dwa sprzeczne rozumowania to nie duplikat, to para kontrastowa do DPO.

## 2. Mechanizm i formalizacja

Deduplikacja staje się regułą decyzyjną **po typie krawędzi**, a nie po wartości
podobienstwa. Niech `G = (V, E)` będzie grafem znakowanym zbudowanym jak w
pomyśle 1 (typowanie krawędzi przez `decode`, przefiltrowanym przez
`applicability`).

| krawędź | interpretacja | reguła |
|---|---|---|
| `syn`, conf ≥ τ | duplikat semantyczny | zostaw 1 reprezentanta (orbity `syn` = klasy równoważności) |
| `ant`, conf ≥ τ | **para kontrastowa** | zatrzymaj **obie** i wypisz jako parę augmentacyjną |
| `hyper` / `hypo` | generalizacja / instancja | zatrzymaj ogólnik + `m` instancji, gdzie `m` = min(3, liczba dzieci) |
| brak krawędzi | różne fakty | zatrzymaj obie |

Reprezentanta orbity wybieramy **nie** jako najbliższy centroidzie (to
preferuje zdania „centralne", czyli ogólnikowe), tylko jako **najdalszy od
transportu własnego `hyper`** — czyli najbardziej informacyjny wariant. To mała,
ale istotna róznica: dotychczasowy dobór reprezentanta systematycznie spłaszcza
korpus.

Formalnie: dla orbity `O` i jej reprezentanta `c`:

    c = argmax_{x ∈ O} ‖ T_hyper(x) − mean_{y ∈ O} T_hyper(y) ‖

czyli „zostaw ten wariant, który najbardziej się różni, gdy oba uogólnimy" —
najwięcej treści ponad poziom ogólności.

## 3. Algorytm

```python
flow = RelationFlow.load(ckpt, device="cuda")

# 1. kandydaci przez ANN (tak jak w pomyśle 1)
pairs = ann_candidates(vectors, thresh=0.6, top_k=32)

# 2. typowanie relacją
rel, conf = flow.model.decode(Va, Vb)
appl      = flow.applicability(Va, rel)
ok        = (conf >= 0.7) & (appl >= 0.5)

# 3. reguła decyzyjna per typ
keep, contrast_pairs = set(), []
for (i, j, r, c) in zip(idx_a, idx_b, rel, conf):
    if not ok[i]:  keep |= {i, j};  continue
    if  name(r) == "syn":  keep.add(representative(orbit(i)))
    elif name(r) == "ant": keep |= {i, j};  contrast_pairs.append((i, j, c))
    else:                  keep |= {i, j}          # hyper/hypo/brak

# 4. produkt uboczny: zbiór par kontrastowych do DPO / InfoNCE
dump_jsonl(contrast_pairs, "contrast_pairs.jsonl")
```

Kroki 2–3 są **tanie** wobec kosztu trenowania; cała przewaga nad progiem
cosine bierze się z jednej dodatkowej klasyfikacji na kandydata.

Wariant drugi, bez trenowania modelu: jeśli F4 na zdaniach nie działa, można
zastosować **próg cosine + filtr negacji leksykalnej** jako baseline, a różnicę
miarę potraktować jako dolne przybliżenie zysku.

## 4. Metryki

| metryka | definicja | jak mierzyć |
|---|---|---|
| **negation accuracy delta** | spadek dokładności po zamianie „nie" / antonimie w teście | contrast set w stylu NCT, 500 zdań PL |
| **conflict recall** | ile par sprzecznych zostało zachowanych w korpusie | audyt 300 par na wyjściu dedup |
| NLI probe | F1 klasyfikatora entailment/contradiction trenowanego na zamrożonym embederze po dedup | 2 kolumny: przed / po |
| krzywa uczenia | strata vs liczba tokenów | ta sama kolejka GPU, 2 przebiegi |
| współczynnik redukcji | % usuniętych przykładów | cel: dorównać SemDeDup |

**Kryterium sukcesu:** ta sama redukcja korpusu co SemDeDup **oraz** ≥ 2 pkt
lepiej na teście wrażliwym na negację. Jeśli redukcja jest gorsza, regułę
należy przestroic (parametr `τ`), a nie odrzucać pomysł.

**Baseline'y:** bez dedup, próg cosine, SemDeDup, losowe podpróbkowanie.

## 5. Fazy

| faza | czas | artefakt |
|---|---|---|
| 2a | 3 dni | reguła typująca na korpusie + `contrast_pairs.jsonl` |
| 2b | 4 dni | ewaluacja „na sucho" (conflict recall, audyt par) — bez GPU |
| 2c | 1 tydzień + kolejka | dwa przebiegi fine-tuningu (baseline vs typowana) |
| 2d | 3 dni | test wrażliwy na negację + ablacja (bez par `ant`) |

## 6. Ryzyka

| ryzyko | mitygacja |
|---|---|
| dekoder myli `syn` z `hyper` na zdaniach | próg conf + macierz pomyłek na 200 parach przed właściwym biegiem |
| usunięcie za mało (korpus nie maleje) | reguła i tak daje zysk przez pary kontrastowe; zgłaszamy to jako „dedup + augmentacja" |
| koszt przepuszczenia korpusu przez ANN i dekoder | linearuy w kandydatach; dla 10⁷ zdań: kilka godzin GPU, raz na wersję korpusu |
| para „przeciwieństw" to czasem dwie różne domeny (np. „bank" finansowy / rzeczny) | filtr `applicability`: niska aplikowalność = nie typujemy, zostawiamy oba |

## 7. Test go/no-go

Ten sam co w pomyśle 1 (AUC dekodera `ant` vs `syn` na 100 parach zdań).
Dodatkowo: na istniejącym korpusie zmierzyć, **ile procent par o cosine > 0.9
to zdania sprzeczne** (audyt 100 par ręcznie). Jeśli > 5%, problem jest
mierzalny i pomysł ma uzasadnienie ekonomiczne, zanim cokolwiek wytrenujemy.

## 7b. Pomiar na fiksie (sekcja 01c, 2026-10-29)

Pomysł jest mierzony razem z 01: `clustering_report` dokłada sekcję **01c**
(`relgat_projector/core/diagnostics/dedup.py`). Reguła typująca scala wyłącznie
krawędzie `syn`; baseline'em jest dedup po progu cosine (SemDeDup), a oba
raportowane są **z własną redukcją**, bo bez osi kosztu liczba nie znaczy nic.

Dwie metryki, zawsze obok siebie:

- `reduction` — udział usuniętych pozycji (koszt),
- `conflict_recall` — udział par `ant`, w których **oba**成员 przetrwały (to jest
  to, co próg cosine niszczy).

Test z §7 (`high_cosine_contradiction_share`) na fiksie 01 w przestrzeni
`ipipan/silver-retriever-base-v1.1`: **śród par o cosine > 0.95 aż 28 % to
sprzeczności** i 14 % parafrazy. Próg cosine scala więc sprzeczności
systematycznie — warunek brzegowy pomysłu jest spełniony z nawiązką.

| reguła | redukcja | conflict recall |
|---|---|---|
| próg cos ≤ 0.99 | 0.056 | 0.875 |
| próg cos ≤ 0.97 | 0.556 | **0.062** |
| próg cos ≤ 0.95 | 0.750 | **0.031** |
| `typed:decoder` | 0.000 | 1.000 (nic nie scalone) |
| `typed:transport` | 0.972 | 0.000 (scala wszystko) |

Wniosek jest przykry i dlatego wart zapisania: **pułapka jest zmierzona, ale
obecny operator jeszcze nie daje reguły, która ją omija**. Dekoder na zdaniach
fiksa nie typuje żadnej pary jako `syn` (redukcja 0), a typowanie transportem
`argmax_r cos(T_r a, b)` scala cały korpus w jeden klaster — obie porażki są
spójne z niskim go/no-go 01a (~0.42–0.56). Reguła typująca ma sens dopiero po
treningu zdaniowym; to jest dokładnie ta dziura, którą opisuje §6 ryzyk.

Uruchomienie: `python -m relgat_projector_apps.diagnostics.clustering_report
--ckpt <run>/best --langs pl en`.

## 8. Punkty odniesienia

- SemDeDup (2023) — punkt wyjścia i główny baseline.
- `FairDeDup` (2024) — dowod, że społeczność widzi problemy deduplikacji, ale
  nie ten dotyczący negacji.
- Contrast sets (NCT-style) i DPO — miejsca, gdzie wygenerowane pary
  kontrastowe idą bezpośrednio do użycia.
