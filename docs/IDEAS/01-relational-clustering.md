# 1. Klasteryzacja relacyjna — skupienia jako orbity, nie jako chmury

**Teza w jednym zdaniu:** zamiast grupować wektory po odległości, grupuj je po
**typie relacji**, która między nimi zachodzi — synonimia spaja, antonimia
odpycha, a liczba i nazwa klastrów bierze się z algebry relacji, nie z
parametru `k`.

**Prymitywy:** P4 `decode` · P6 `C_rs` · P1 odwracalność · P5 `applicability`
**Wysiłek:** ~3 tygodnie · **Zależności:** F4 z `shadow vectors` = zdaniami,
jeśli klasterujemy tekst, a nie hasła słownika

---

## 1. Problem

Cosine umieszcza „bateria jest świetna" i „bateria jest fatalna" blisko siebie,
bo oba zdania dzielą temat, rejestr i szkielet leksykalny. To nie jest wada
konkretnego embedera ani przypadek — embedding uczy się grupować **współwystę-
powanie**, a antonimy w języku celowo występują w identycznym kontekście.
Zjawisko jest opisane wprost w literaturze (m.in. *Is Cosine-Similarity of
Embeddings Really About Similarity?*, *Problems with Cosine as a Measure of
Embedding Similarity*, *Intuitive Contrasting Map for Antonym Embeddings*).

Skutek praktyczny: każdy pipeline, który **grupuje, a potem streszcza lub
odszumia**, zlewa sprzeczności w jedno zdanie zbiorcze. Widoczne w agregacji
opinii, topic modelingu korpusów prawnych i społecznych, deduplikacji danych
treningowych (patrz pomysł 2) oraz w zbieraniu „podobnych zgłoszeń".

Dlaczego znane obejścia nie wystarczają:

- **cosine + NLI** — NLI ocenia relację dla pary, więc koszt to O(N²) wywołań
  modelu generującego; na korpusie 10⁶ tekstów jest nieprzechodnie;
- **usuwane negacje („nie")** — nie działa na antonimach leksykalnych
  (świetny/fatalny) ani na negacji implicytnej („zawiodło oczekiwania");
- **fine-tuning embeddera na parach contrastive** — wymaga etykiet, których nie
  ma w skali, i psuje geometrię dla pozostałych zastosowań.

## 2. Mechanizm i formalizacja

Relacja jest elementem grupy działającym w przestrzeni wektorów. Definiujemy
zatem grupowanie nie przez metrykę, lecz przez **orbitę podgrupy dozwolonej**:

    O(x) = { T_g x : g ∈ G },   G ⊂ ⟨R⟩ — podgrupa relacji „tolerowanych"

gdzie `R` to słownik relacji (`syn`, `hyper`, `hypo`, `ant`, ...), a `G` to
wybór, co uznajemy za „to samo pojęcie" — np. `G = {syn}` albo
`G = {syn, hyper(λ ≤ 0.4)}`. Przyciąganie i odpychanie pochodzą z **tej samej
algebry**, nie z dwóch niespójnych modeli.

Przynależność i konflikt:

    i ~ j  ⟺  decode(v_i, v_j) ∈ G  ∧  conf(i,j) ≥ τ_attr
    i ⊥ j  ⟺  decode(v_i, v_j) = ant ∧  conf(i,j) ≥ τ_rep

Kluczowa, niedoceniana zaleta: ponieważ `T_r` ma **dokładną odwrotność**,
spójność jest symetryczna z konstrukcji. Jeżeli `T_syn(v_i) ≈ v_j`, to
`T_syn⁻¹(v_j) ≈ v_i` — bez osobnego nadzorowania, bez drugiej wyceny modelu.
Podejścia NLI/metryczne tej gwarancji nie mają i muszą ją wymuszać dodatkowym
liczeniem albo arbitralną kolejnością par.

Dodatkowo `C_rs` (P6) mówi, **które relacje wolno mieszać wewnątrz klastra**:
jeśli `C_rs[syn, hyper] ≈ 1`, orbita po `syn` i po `hyper` jest spójna i klaster
może być szerszy; jeśli `≈ 0`, mieszanie łańcuchów tworzy klaster pozorny.
`C_rs` staje się więc **parametrem strukturalnym** klasteryzacji, a nie
ciekawostką diagnostyczną.

## 3. Algorytm

**Krok 0 — redukcja kandydatów.** `decode` to MLP o wejściu `2D`; pełne N² to
za dużo. Kandydaci z ANN (wskaźnik: `cos > 0.55`, top-50 na wektor). Dla 20k
wektorów daje ~10⁶ par: GPU sekundy, CPU minuty. Koszt jest **liniowy w liczbie
krawędzi**, nie kwadratowy w N — i degraduje monotonicznie (mniej krawędzi =
słabsza znakowość, ale brak błędów systematycznych).

**Krok 1 — typowanie krawędzi** (batchowo, bez pętli Pythona):

```python
from relgat_projector.inference import RelationFlow
from relgat_projector.handlers.wiazka_checkpoint import load_commutation

flow = RelationFlow.load(ckpt, device="cuda")
C    = load_commutation(ckpt)                      # macierz C_rs [R, R]

va, vb = torch.stack(A), torch.stack(B)            # [P, D_op] — przestrzeń operatora
rel_id, conf = flow.model.decode(va, vb)           # logits → id, conf [P]
appl       = flow.applicability(va, rel_id)        # [P] — czy relacja jest tu zdefiniowana
keep       = (conf >= tau_attr) & (appl >= tau_appl)
sign       = torch.where(rel_id == ID_ANT, -1.0, 1.0)
W          = sign * conf                            # graf znakowany, rzadki
```

Uwaga wdrożeniowa: wektory muszą być w **przestrzeni operatora** (`flow.refine`
/ `model.base_to_op`), a nie surowymi wektorami embedera. Tekst spoza grafu:
`flow.refine(v_raw)` i dalej jak wyżej.

**Krok 2 — klasteryzacja grafu znakowanego.** Macierz `W` o znakach ±1·conf.
Warianty: correlation clustering (PVQ/LPCC) albo etykietowanie rozprzestrzeniane
z regułą Heidera (`+ z + → +`, `+ z − → −`). Drugi wariant jest lepszy na
start, bo **nie wymaga podania `k`** i zwraca węzły nieprzypisane zamiast na
siłę wciskać je do najbliższego klastra.

**Krok 3 — oś sporu z nazwą.** Dla par klastrów o dużej gęstości krawędzi
ujemnych liczymy transport między ich centroidami i odczytujemy typ relacji o
największym udziale. Zamiast spekulować, „o co im chodzi" (PCA na różnicy
centroidów), dostajemy **nazwaną oś sporu** — bo relacja pochodzi ze słownika.
To jedyny moment w całym pipeline’zie grupowania, w którym wymiar ma etykietę.

**Krok 4 — walidacja wewnętrzna bez etykiet.** W obrębie klastra liczymy
holonomię `HD` na czwórkach zbudowanych z węzłów klastra (sampler z repo).
Jeśli wewnątrz „grupy" transport nie jest spójny ścieżkowo, klaster jest
pozorny. To darmowa metryka jakości, której **żaden** clustering metryczny nie
potrafi podać.

## 4. Metryki

| metryka | definicja | oczekiwany wynik |
|---|---|---|
| **contradiction-purity (CP)** | odsetek klastrów bez żadnej pary `ant` o `conf ≥ τ` | nasz: > 0.9; cosine-kmeans: znacznie niżej — to **metryka docelowa** |
| ARI / NMI | zgodność z etykietami ludzkimi | nie gorsze niż baseline |
| liczba klastrów przy tym samym ARI | mniej = lepiej (brak rozbijania przez antonimy) | spadek o 20–40% |
| krzywa CP–ARI | główny rysunek raportu: X = ARI, Y = CP | nasza linia dominuje |
| `HD` wewnątrz klastra | spójność ścieżkowa | niska w dobrych klastrach; korelacja z oceną ludzką |

**Baseline'y:** cosine-kmeans, HDBSCAN na cosine, dedup po progu cosine
(SemDeDup-style), oraz **NLI + cosine** jako kosztowny oracle (górna granica
bez naszego modelu — pokazuje, ile zysku daje sam typowany graf).

## 5. Dane ewaluacyjne

Trzy korpusy po 500–1000 tekstów, dwóch adnotatorów, liczyć Cohena kappę:

1. opinie produktowe (etykiety: pozytywna / negatywna / mieszana),
2. wątki dyskusyjne (stanowisko: za / przeciw / umiarkowane),
3. zgłoszenia serwisowe (ten sam problem / inny problem).

Etykieta `ant` z dekodera walidowana **osobno** (100 par, ocena ludzka) — nie
wolno jej mieszać z „złym klastrlem", bo to dwa różne błędy.

## 6. Fazy

| faza | czas | artefakt |
|---|---|---|
| 1a | 1 dzień | **test go/no-go** (patrz §8) |
| 1b | 3 dni | `scripts/edge_typing.py` — batch `decode` + `applicability` + zapis grafu (parquet) |
| 1c | 4 dni | klasteryzacja znakowana + 4 baseline'y + CP/ARI/krzywa |
| 1d | 1 tydzień | korpusy + adnotacja + pierwsza ewaluacja |
| 1e | 3 dni | nazwana oś sporu + walidacja `HD` wewnątrz klastrów + raport |

## 7. Ryzyka

| ryzyko | mitygacja / test |
|---|---|
| dekoder trenowany na hasłach, nie na zdaniach — poza słownikiem szumi | F4 z `shadow vectors` = zdaniami z docelowego indeksu; bez tego zakres = tylko jednostki leksykalne |
| dekoder halucynuje relację dla par bez związku | filtr `applicability`; **klasa „brak krawędzi"** zamiast słabej wagi; audyt 100 par |
| słownik Słowosieci nie ma relacji „oceny" | zacząć od `ant` (jest w grafie); relacje oceniające dopisać dopiero z etykietami |
| N > 10⁵ — koszt ANN i klasteryzacji | dwustopniowo: cosine-kmeans jako blokowanie wstępne, znakowanie wewnątrz bloków |
| correlation clustering jest NP-trudny | aproksymacje (LPCC/PVQ); dla dużych N — lokalnie, z blokowaniem |

## 8. Test go/no-go (najtańszy eksperyment w całym zestawie)

500 zdań opinii z indeksu + 100 par oznaczonych ręcznie (syn / ant / neutralne).
Policzyć **AUC `conf` dekodera** w rozróżnianiu `ant` vs `syn`.

- AUC > 0.8 → pomysł 1 stoi, a razem z nim pomysły 2, 4 i 12a;
- AUC ≈ 0.6 → cała rodzina wymaga F4 na zdaniach, zanim cokolwiek zbudujesz.

Jeden dzień pracy, a odpowiada na pytanie, które warte jest więcej niż miesiąc
budowy. **To powinien być pierwszy eksperyment po przeczytaniu tego zbioru.**

## 9. Punkty odniesienia

- *Is Cosine-Similarity of Embeddings Really About Similarity?*; *Problems with
  Cosine as a Measure of Embedding Similarity*; *Intuitive Contrasting Map for
  Antonym Embeddings* — problem znany i **nienaprawiony zmianą metryki**.
- SemDeDup (2023) — ten sam korzeń problemu w kontekście danych treningowych.
- Heider (równowaga strukturalna), correlation clustering — narzędzia, które
  odziedziczamy za darmo, gdy mamy graf znakowany.
- Kod: `relgat_projector/core/heads/relation_decoder.py`,
  `relgat_projector/core/metrics/geometry.py` (`holonomy_defect`),
  `relgat_projector/handlers/wiazka_checkpoint.py` (`load_commutation`).
