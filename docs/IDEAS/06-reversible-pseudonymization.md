# 6. Odwracalna pseudonimizacja semantyczna (prywatność z kluczem i pokrętłem)

**Teza w jednym zdaniu:** wektor `T_r(λ)·v` publikujemy zamiast `v` — `λ` jest
pokrętłem anonimowości, `r` wybiera **kierunek** uogólnienia (tożsamość →
rola → osoba), a dokładna odwrotność `T_r⁻¹` jest **kluczem** do zgodnego z
prawem odszyfrowania; otrzymujemy krzywą prywatność–użyteczność, której szum
dajeć nie potrafi.

**Prymitywy:** P1 `inverse` (dokładne) · P3 dial `λ` · P5 `applicability`
**Wysiłek:** 1 tydzień jako **sonda badawcza**, potem decyzja
**Ryzyko naukowe:** **duże** — to jedyny pomysł w zestawie, który może nie
wypalić, i dlatego jest opisany jako eksperyment, nie jako produkt

---

## 1. Problem

Wektory embedera wyciekają treść. Ataki invertujące odtwarzają **całe zdania**
z samych embeddingów:

- *Sentence Embedding Leaks More Information than You Expect: Generative
  Embedding Inversion Attack to Recover the Whole Sentence* (2023);
- *Against All Odds: Overcoming Typology, Script, and Language Confusion in
  Multilingual Embedding Inversion Attacks* (2024) — działa także poza
  angielskim, czyli także dla polszczyzny;
- *Embedding Inversion via Conditional Masked Diffusion Language Models* (2026);
- *Denoising-Aware Inversion: Revealing Privacy Risks in Noise-Protected Text
  Embeddings* (2026) — **szum nie ratuje**: atakujący nauczy się szumu odszewać.

Ten ostatni punkt jest kluczowy dla naszej branży: standardowa odpowiedź
„dodaj szum do wektora / zkwantyzuj" jest **łamana**, bo szum jest
nierozróżnialny i ma znaną strukturę. Potrzebna jest transformacja, która:

1. realnie psuje odtwarzanie (nie tylko „trochę zasłania"),
2. **zachowuje użyteczność** zadania, dla którego wektor wysyłamy (wyszukiwanie,
   klasyfikacja, klasteryzacja),
3. daje się **cofnąć w sposób kontrolowany i audytowalny**, bo w systemach
   prawnych (RODO: pseudonimizacja ≠ anonimizacja) uprawniony podmiot musi
   odzyskać dane.

Warunku 3 nie spełnia żadna metoda z literatury o ochronie embeddingów — bo
wszystkie są traconośne z definicji.

## 2. Mechanizm i formalizacja

Publikujemy `ṽ = T_r(λ)·v`, gdzie `T_r` to nasz operator, a `r`, `λ` są kluczem.

Trzy własności, które robią tu robotę:

1. **Izometria do skali (P2) zachowuje użyteczność.** Dla dowolnej pary
   `cos(T_r u, T_r v) = cos(u, v)` — więc wyszukiwanie cosine, klasteryzacja i
   klasyfikator liniowy działają **identycznie** jak na oryginalnych wektorach,
   tyle że w obróconej bazie. To przewaga nad szumem, który degraduje
   współczynnik podobieństwa wszędzie.
2. **Uogólnianie jako mechanizm prywatności.** Nie każda relacja nadaje się
   jednakowo. `hyper` przesuwa wektor w stronę **klasy nadrzędnej**:
   `Kowalski → kierownik → osoba`. To jest semantyczny odpowiednik
   **k-anonimowości**: `λ` steruje tym, jak głęboko w hierarchii lądujemy, a
   liczba elementów orbity w promieniu `ε` to de facto rozmiar równoważności.
   `ant` albo `syn` **nie dają** prywatności (to obrót w tej samej warstwie
   hierarchii) — więc dobór `r` nie jest obojętny i trzeba go mierzyć.
3. **Odwrotność jest kluczem.** `inverse` w operatorze Givensa jest
   **dokładne z konstrukcji** (`exp(−Ω)/s`, patrz `givens.py`), nie przybliżone.
   Zatem mamy schemat: `publikuj T_r(λ)v`, `odszyfruj T_r⁻¹`, a kluczem jest
   tensor `θ_r` (i `log_scale_r`). Kto ma `θ_r`, ten odzyskuje; kto nie ma, ten
   widzi obrócony wektor o tej samej normie.

Protokół doboru `λ` — i to jest główny wynik eksperymentu:

    λ* = min { λ : atak invertujący na {T_r(λ)v} osiąga BLEU/ROUGE ≤ β
                             ∧ AUC ataku linkage ≤ 0.55 }

dla siatki `λ ∈ {0, 0.1, …, 1}` i dla każdej relacji z osobna. Raport:
**krzywa prywatność–użyteczność per relacja**. Tabela, której nie da się
otrzymać dla metod szumowych, bo tam obie osie degradują równocześnie.

Granica bezpieczeństwa: `applicability(v, hyper)` mówi, czy da się jeszcze
uogólniać. Jeśli niska, oznacza to „brak dalszego stopnia ogólności" — czyli
**nie da się już zwiększyć prywatności** i trzeba zastosować inną metodę (np.
usunięcie rekordu). To użyteczny sygnał brzegowy, którego nie wymyślisz z
szumu.

## 3. Algorytm eksperymentu

Krok 1 — **odtworzenie ataku** (bez tego nie ma pomiaru prywatności):
implementujemy attack-in-the-middle w wersji uproszczczonej — decoder (mały
LLM lub T5) uczony na `(wektor → zdanie)` na danych treningowych, potem testuje
na wektorach ochrony. Uproszczenie jest uczciwe, o ile raportujemy **redused
setup**: atakujący zna architekturę i embeder, nie zna `θ_r`.

Krok 2 — **siatka ochrony**:

```python
flow = RelationFlow.load(ckpt, device="cuda")
for r in ["hyper", "hypo", "syn", "ant"]:
    for lam in np.arange(0, 1.01, 0.1):
        v_prot = flow.apply(V, r, lam=lam)
        bleu   = inversion_attack(v_prot)          # odzysk tekstu
        auc    = linkage_attack(v_prot, V_meta)    # rozpoznanie autora/encji
        util   = downstream_mrr(v_prot, index)     # użyteczność zadania
        emit(r, lam, bleu, auc, util)
```

Krok 3 — **weryfikacja odwracalności** (klucz działa?):

```python
back = flow.model.inverse(v_prot, rel_ids)
assert (back - V).abs().max() < 1e-4      # CCE — dokładnie, nie przybliżone
```

Krok 4 — **test ataku „wielu wektorów"** (największe ryzyko tego pomysłu):
jeśli atakujący widzi **dużo** wektorów objętych tą samą ochroną, struktura
orbity może pomóc odgadnąć `θ_r` (to klasyczny atak kryptoanalityczny na
szyfr strumieniowy z powtarzającym się keystreamem). Test: trenujemy dekodera
na 10⁴, 10⁵, 10⁶ wektorów chronionych i sprawdzamy, czy BLEU rośnie z liczbą
próbek. **Jeśli rośnie znacznie — pomysł upada** i to jest dokładnie to, co
chcemy wiedzieć po pierwszym tygodniu.

## 4. Metryki

| metryka | definicja | cel |
|---|---|---|
| recovery BLEU/ROUGE | odzysk tekstu przez atak | ≤ 0.1 przy `λ*` |
| AUC ataku linkage | rozpoznanie encji/osoby po wektorze | ≤ 0.55 |
| **k-anonymity semantyczne** | mediana liczby orbity w promieniu `ε` | ≥ 20 |
| ΔMRR użyteczności | spadek jakości wyszukiwania po ochronie | ≤ 2 pkt |
| krzywa P–U | **główny wynik**: prywatność vs użyteczność per relacja | nasza linia pod baseline'ami |
| wzrost BLEU z N | czułość na atak wieloupróbkowy | **płasko** — inaczej pomysł odpada |

**Baseline'y:** brak ochrony, szum gaussian (dopasowany do tej samej
użyteczności), kwantyzacja scalar, losowa rotacja **bez klucza** (traconośna),
rotacja z kluczem (bez właściwości semantycznych — pokazuje, co daje **dobór
relacji**, a nie sam obrót).

Ten ostatni baseline jest najważniejszy: pokazuje, czy wartość dodana bierze
się z **algebry** (możliwość uogólnienia, interpretowalność, `applicability`),
czy tylko z faktu, że obrót jest odwracalny. Musimy umieścić różnicę.

## 5. Fazy

| faza | czas | artefakt | decyzja |
|---|---|---|---|
| 6a | 3 dni | atak invertujący (uproszczony) + baseline bez ochrony | czy atak w ogóle działa na naszej przestrzeni |
| 6b | 3 dni | siatka `λ × relacja` + krzywe P–U | czy `λ` w ogóle coś daje |
| 6c | 2 dni | **test wielu wektorów** (ryzyko główne) | go/no-go |
| 6d | 1 tydzień | pełna ewaluacja + baseline'y + raport | decyzja: publikować / porzucić |

## 6. Ryzyka

| ryzyko | ocena | mitygacja / test |
|---|---|---|
| atak wieloupróbkowy odzyskuje `θ_r` z orbity | **wysokie** | faza 6c; jeśli BLEU rośnie z N — pomysł porzucić |
| obrót zachowuje za dużo struktury (izometria = zaleta dla atakującego) | wysokie | to fundamentalne napięcie: ta sama własność, która chroni użyteczność, chroni informację |
| „klucz" to duży tensor, nie praktyczny w RO DO | średnie | kluczem może być sam `θ_r` (R×K liczb) — mały, możliwy do pieczętowania |
| atakujący zna `r` i `λ` (jawny protokół) | pewne w modelu Kerckhoffsa | wtedy ochrona = `θ_r`; raportować oba modele zagrożeń osobno |
| brak realnego partnera z danymi wrażliwymi | średnie | eksperyment na danych syntetycznych + dane publiczne z adnotacjami PII |

## 7. Test go/no-go

Trzy dni, jeden skrypt: atak invertujący na `V` (bez ochrony) i na
`T_hyper(λ=0.8)V`. Jeśli BLEU spada z np. 0.45 do < 0.15 **przy ΔMRR ≤ 2 pkt**,
jest co badać. Jeśli nie spada wcale — kierunek jest martwy i szkoda tygodnia.

## 7b. Pomiar na operatorze (sekcja 06, 2026-10-30)

Sekcja **06** w `operator_report`
(`relgat_projector/core/diagnostics/pseudonymization.py`). Uruchomienie:
`python -m relgat_projector_apps.diagnostics.operator_report --ckpt
wiazka-models/S14_20261029_210326/best --section 06`. Prywatność mierzona
atakiem linkage (model zagrożeń z §3: atakujący zna korpusem i architekturę,
nie zna `theta_r`); brak w tej chwili dekodera invertującego, więc BLEU z §4
nie jest jeszcze liczbowo dostępne.

Promień anonimowości liczony z kwantyla rozkładu kosinusów korpusem
(`epsilon = 0.0258` dla 1%), **nie** jako stała: w przestrzeni o średnim
cosinie par ~0.95 sztywne `epsilon = 0.05` wrzuca do kuli 1963 z 2048
wektorów i metryka przestaje cokolwiek znaczyć.

| ramię (krok `synonimia`, λ = 1) | AUC ataku linkage | MRR (użyteczność) | `cce` (klucz) |
|---|---|---|---|
| klucz semantyczny `T_r(λ)` | **0.955** | **0.990** | **4.3·10⁻⁷** |
| losowa rotacja z kluczem | 0.088 | 0.100 | — |
| szum gaussowski (dopasowany krok) | 0.578 | 0.113 | — |

Trzy wnioski, wszystkie niezgodne z tezą lub ją zawężające:

1. **Klucz działa dokładnie** — `cce = 4·10⁻⁷` dla każdego `λ`. To jest
   część twierdząca pomysłu i jako jedyna przechodzi bez zastrzeżeń.
   Wymusiła ona zresztą poprawkę w kodzie: `inverse()` odwracał wyłącznie
   pełny krok, więc opublikowanie `T_r(λ) v` nie miało w ogóle klucza.
2. **Prywatności nie ma wcale.** `T_r(λ)` zachowuje użyteczność (MRR 0.990)
   i w zamian zachowuje tożsamość: AUC ataku spada z 0.990 do 0.955 na
   całym zakresie `λ`. Ramiona, które atak psują (rotacja 0.088, szum 0.578),
   psują też wyszukiwanie (MRR 0.100 i 0.113). **Krzywa prywatność–
   użyteczność nie istnieje — jest pionowa.** To dokładnie napięcie z §6,
   wpisane tam jako ryzyko „wysokie”: izometria chroni użyteczność dokładnie
   tak samo dobrze, jak chroni tożsamość, bo dla `cos` te dwie rzeczy to ta
   sama wielkość.
3. **Warunek 1 z §2 jest więc spełniony, a warunek 2 nie**: `hyper` miał
   wnosić prywatność przez uogólnienie, ale `T_r` realizuje obrót, nie
   przejście po hierarchii — orbita encji pozostaje jednoelementowa w sensie
   kosinusowym (`k`-anon śr. 0 przy `epsilon` mierzonym).

**Co to oznacza dla pomysłu:** jako schemat prywatności jest do odrzucenia
w obecnym kształcie, i §6 miało rację, prosząc o ten test w pierwszym
tygodniu. Wartość, która zostaje, jest inna i mniejsza: *dokładnie
odwracalna, izometryczna transformacja wektorów* nadaje się do
współdzielenia indeksu między instytucjami z kluczem (synchronizacja bez
ujawniania `theta_r`), tylko nie do anonimizacji.

## 8. Punkty odniesienia

- Wszystkie cztery prace o inwersji embeddingów (2023, 2024, 2026, 2026) —
  definiują model zagrożenia; **żadna nie proponuje transformacji odwracalnej
  z kluczem**, wszystkie są traconośne.
- k-anonimowość (Sweeney) — pojęcie, które przenosimy do przestrzeni semantycznej.
- Kerckhoffs — założenie, że algorytm jest jawny, a kluczem są parametry.
- W repo: `core/operator/givens.py` (`inverse`, `apply` z `lambdas`),
  `core/metrics/geometry.py` (`cycle_closure_error` — weryfikacja klucza),
  `core/heads/applicability.py` (brak dalszego stopnia uogólnienia).
