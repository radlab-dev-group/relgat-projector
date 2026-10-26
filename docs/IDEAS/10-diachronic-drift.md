# 10. Dryf semantyczny bez wyrównywania osi czasu (współdzielona baza generatorów)

**Teza w jednym zdaniu:** wszystkie slice'e czasowe dzielą **tę samą bazę
generatorów `B`**, więc współrzędne `θ_r(t)` są porównywalne **bez żadnego
dopasowywania osi** — a to rozwiązuje klasyczną przeszkodę analiz diachronicznych
i daje przesunięcie znaczenia **przypisane do konkretnej relacji**, w jednostkach
kąta.

**Prymitywy:** P6 `C_rs` · P1 `θ_r` we wspólnym `B` · P3 dial
**Wysiłek:** 3 tygodnie · **Zależności:** modyfikacja trenera (zamrożenie `B`,
trening per slice) — mała, ale konieczna

---

## 1. Problem

Analiza zmian znaczenia w czasie (diachronic embeddings) rozbija się o jedno:
przestrzenie z dwóch okresów są określone **z dokładnością do obrotu**. Żeby
powiedzieć „słowo `X` przesunęło się", trzeba najpierw obrócić jedną przestrzeń
do drugiej — zazwyczaj Procrustesem na słowniku kotwiczącym albo metodą
ortogonalną z automatycznymi słownikami.

To **klasyczna przeszkoda** tej dziedziny (por. *Diachronic word embeddings and
semantic shifts: a survey*, 2018; zastosowania nowsze, np. analiza zmian w
języku chorwackim 2025): cała teza o przesunięciu znaczy tyle, ile znaczy jakość
**dopasowania**, a dopasowanie jest najlepsze dokładnie dla słów, które się
**nie zmieniły** (bo one najlepiej kotwiczą). Błąd systematyczny wbudowany w
samą metodę.

Druga bolączka: wynik jest **bezimienny**. Mówimy „`praca` przesunęła się o
0.3", ale nie umiemy powiedzieć **w jakim sensie** — czy zmienił się zakres
(hiperonim), walencja, czy skojarzenia. A to właśnie jest interesujące dla
leksykografa, socjologa czy działu compliance.

## 2. Mechanizm

Operator Givensa ma dwie części: **bazę generatorów** `B` (wspólną dla wszyst-
kich relacji) i **współrzędne relacji** `θ_r` (pojedyncze liczby na relację i
generator). Jeśli przy analizie kolejnych okien czasowych **zamrozimy `B`** i
będziemy uczyć wyłącznie `θ_r`, `log_scale_r` i `C_rs`, to:

    θ_r(t₁)  i  θ_r(t₂)    są w TEJ SAMEJ bazie

czyli **różnica `Δθ_r = θ_r(t₂) − θ_r(t₁)` jest wielkością fizyczną**, nie
umową: kąt obrotu w ustalonej płaszczyźnie, mierzony w radianach. Zero
dopasowywania osi, zero słownika kotwiczącego, zero założenia o kształcie
rozkładu.

To nie jest trik implementacyjny — to **zmiana paradygmatu pomiaru**: zamiast
mierzyć przesunięcie punktu (co wymaga układu odniesienia), mierzymy zmianę
**operatora** (której układ odniesienia nie dotyczy, bo trzymamy go wspólny).

Interpretacja trzech sygnałów:

| sygnał | co znaczy zmiana |
|---|---|
| `‖Δθ_r‖` | jak bardzo zmieniło się **znaczenie relacji** `r` w czasie |
| `ΔC_rs` | zmiana **porządku świata**: dwie osy, które były niezależne, przestały być (albo odwrotnie) |
| `Δ log_scale_r` | zmiana „siły" relacji — np. hiponimia zaczęła mocniej zawężać |
| `ΔHD` | zmiana **spójności** systemu pojęciowego w danym okresie (miara polaryzacji dyskursu) |

Przykład interpretowalnego wyniku, którego nie da się uzyskać z cosine: „relacja
`przyczyna` w latach 2015–2025 obróciła się o 0.42 rad w płaszczyźnie
generatora `B₇` i przestała komutować z relacją `odpowiedzialność`
(`C` z 0.86 → 0.31)". To jest **teza merytoryczna**, nie chmura słów.

## 3. Algorytm i wymagana zmiana w kodzie

Obecnie trener tworzy operator od zera (`WiazkaTrainer.__init__` →
`build_operator`). Potrzebna jest opcja **zamrożenia bazy i kontynuacji**:

```python
# WiazkaConfig: + warmstart_operator: str | None, + freeze_basis: bool
op = build_operator("givens", dim=D, num_rel=R, num_generators=K)
sd = torch.load(warm_ckpt / "model.pt")["model"]
op.load_state_dict({k[len("operator."):]: v for k, v in sd.items()
                    if k.startswith("operator.")})
if freeze_basis:
    op.basis_raw.requires_grad_(False)      # B zamrożone — to jest oś odniesienia
# uczymy TYLKO theta, log_scale, (i C_rs w stracie straty)
```

Wymagane drobne zmiany:

1. `WiazkaConfig`: `warmstart_operator`, `freeze_basis` (i propagacja do
   `build_operator` / `WiazkaModel`);
2. upewnienie się, że **optimizér obejmuje `C_rs`** (należy do `CurvatureLoss`
   — w repo jest na to regres test: `test_loss_owned_parameters_are_optimised`);
3. zapis `θ` i `C_rs` do `metrics.json`/`report.json`, żeby dało się różnicować
   między oknami.

Kolejność prac: **ten sam seed, ta sama kolejność danych, zamrożone `B`** —
inaczej różnica `Δθ` będzie zawierać szum inicjalizacji, a nie dryf. To warunek
konieczny, nie zalecenie.

## 4. Metryki

| metryka | gold | cel |
|---|---|---|
| korelacja `‖Δθ_r‖` z adnotacjami leksykografów | 30 relacji × 2 okna, ocena „czy znaczenie się zmieniło" | Spearman ≥ 0.5 |
| **AUC wykrycia suchego przesunięcia** | symulacja: w oknie `t₂` przebudowujemy 5% krawędzi jednej relacji | ≥ 0.9 |
| porównanie z Procrustes: **ile kotwic potrzeba** | AUC vs liczba kotwic (0, 100, 1000) | nasza przy zerze ≥ baseline przy 1000 |
| stabilność seed'a | `Δθ` przy dwóch seedach, ten sam brak zmian | szum **poniżej** sygnału suchego |
| `ΔC_rs` vs reality | ręczna ocena „czy te dwie osy się rozjechały" | AUC ≥ 0.7 |

**Ostatnia pozycja jest testem uczciwości**: jeśli szum seed'a jest tej samej
wielkości co sygnał suchy, **wszystkie** wnioski o dryfie są bezwartościowe. To
trzeba zmierzyć jako pierwsze, nie jako ostatnie.

## 5. Fazy

| faza | czas | artefakt |
|---|---|---|
| 10a | 1 dzień | `freeze_basis` + `warmstart_operator` w trenerze + test, że `B` się nie zmienia |
| 10b | 1 dzień | pomiar szumu seed'a (2 przebiegi, ten sam slice) — **go/no-go** |
| 10c | 1 tydzień | korpus 3–5 okien (np. 2010–2025) + treningi |
| 10d | 1 tydzień | ewaluacja: symulowane przesunięcia + adnotacje + porównanie z Procrustes |

## 6. Ryzyka

| ryzyko | mitygacja |
|---|---|
| szum seed'a ≥ sygnał suchy | 10b przed czymkolwiek; jeśli tak — pomysł umiera i to dobrze, że szybko |
| zamrożone `B` jest zbyt ubogie dla nowych okien (nowe słowa, nowe relacje) | raportować `PCE` per okno; jeśli rośnie, `B` straciło pokrycie — zgłosić jako granicę metody |
| dryf w oknie wynika ze zmiany **źródła danych**, nie z języka | kontrola: ten sam korpus, dwie połówki losowe — suchy `Δθ` musi być bliski zera |
| rozkład krawędzi per okno rzadki | agregacja: okna 3-letnie, bootstrap po krawędziach |

## 7. Test go/no-go (dzień drugi, zanim cokolwiek zbudujemy)

Dwa treningi na **tym samym** oknie z różnymi seedami, `B` zamrożone.
Policzyć `‖Δθ‖`. Potem trening na drugim oknie, policzyć `‖Δθ‖` między oknami.
Jeśli stosunek (różnica międzyoknowa) / (różnica seedowa) < 2 — suchy dryf nie
jest wykrywalny i nie ma o czym mówić.

## 8. Punkty odniesienia

- *Diachronic word embeddings and semantic shifts: a survey* (2018) — diagnoza
  problemu dopasowywania osi.
- *Characterizing Linguistic Shifts in Croatian News via Diachronic Word
  Embeddings* (2025), *From cart to truck: meaning shift through words in
  English in the last two centuries* (2024) — aplikacje, które ciągle używają
  Procrutesa.
- W repo: `core/operator/givens.py` (`basis_raw`, `theta`, `log_scale`),
  `trainer/wiazka_trainer.py` (`WiazkaConfig`, `build_operator`),
  `core/loss/curvature.py` (`commutation()`).
