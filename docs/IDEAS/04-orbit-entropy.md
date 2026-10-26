# 4. Niepewność bez logprobów: entropia orbitowa (typed semantic entropy)

**Teza w jednym zdaniu:** entropia semantyczna grupuje odpowiedzi modelem
„to samo / nie to samo", a powinna rozróżniać **trzy** stany — zgodność
(`syn`), realny konflikt (`ant`) i wypowiadanie się poza opisem (niska
`applicability`) — a ten trzeci stan jest dokładnie tym, co mylnie raportujemy
jako wysoką niepewność.

**Prymitywy:** P4 `decode` · P5 `applicability` + kalibracja `ECE` · P7
spójność · działa bez logprobów (API)
**Wysiłek:** 2 tygodnie · **Zależności:** F4 na zdaniach (dekoder i applicability)

---

## 1. Problem

Metoda entropii semantycznej (semantic entropy, Farquhar i in., Nature 2024)
oraz jej warianty (`Beyond Semantic Entropy: … Pairwise Semantic Similarity`,
2025; `FASE`, 2026) działają tak: próbkujemy `k` odpowiedzi, grupujemy je w
klasy równoważności (próg cosine albo NLI), liczymy entropię na klasach.
Uzasadnienie jest zdrowe: „nie wiem" to nie to samo co „powiem inaczej".

Model binarny (równoważne / nierównoważne) myli jednak dwa zupełnie różne
przypadki:

1. **pięć odpowiedzi w jednej orbicie `syn`** — model jest pewien, tylko
   gadatliwy → niska niepewność, **odpowiedz podajemy**;
2. **dwie odpowiedzi w orbitach połączonych `ant`** — model widzi dwa sprzeczne
   stany świata → wysoka niepewność, **odmawiamy**;
3. **`k` odpowiedzi o niskiej `applicability`** — model opisuje coś, czego nie
   da się opisać dostępnymi relacjami (np. dziedzina spoza wiedzy, pytanie
   bez sensu, prompt injection) → to **nie jest niepewność, to brak pokrycia**;
   poprawną reakcją nie jest „nie wiem", tylko „nie mam o tym wiedzy / pytanie
   poza zakresem".

Przypadek 3 jest dziś systematycznie mylony z 2, bo metryka binarna nie ma
trzeciego wyjścia. To poważny błąd projektowy w systemach z abstainem: model
odmawia odpowiedzi tam, gdzie powinien odpowiedzieć ostrożnie, i odpowiada tam,
gdzie powinien zgłosić brak pokrycia.

## 2. Mechanizm i formalizacja

Dla `k` próbek `v_1..v_k` odpowiedzi budujemy graf znakowany (jak w pomysłach
1 i 2) i definiujemy **podział na orbity**:

    V = O_1 ⊎ O_2 ⊎ … ⊎ O_m      gdzie i ~ j  ⟺  decode(v_i,v_j)=syn ∧ conf ≥ τ

Entropia orbitowa:

    H_orb = − Σ_{t=1..m} p(O_t) log p(O_t),   p(O_t) = |O_t| / k

Trzy sygnały, nie jeden:

| sygnał | wzór | interpretacja | decyzja |
|---|---|---|---|
| `H_orb` | jw. | rozrzut po znaczeniach | wysoka → abstain |
| **konflikt** `κ` | `max` conf nad krawędziami `ant` między orbitami | sprzeczność, nie różnorodność | `κ > τ_κ` → **odmowa z uzasadnieniem** |
| **pokrycie** `α` | średnia `applicability(v_i, r)` | czy w ogóle umiemy to opisać | `α < τ_α` → **„poza zakresem"** |

Wartość progów bierze się z kalibracji, nie z doświadczenia: `ECE` jest już
metryką treningową głowicy aplikowalności (`core/metrics/offline.py`,
`calibration_ece`), więc `τ_α` ustawiamy z krzywej kalibracji zamiast zgadywać.
To różnica między „odmawiam, bo tak wyszedł próg" a „odmawiam, bo przy tej
konfidencji 12% odpowiedzi jest błędnych".

**Objaśnienie dla użytkownika** jest darmowe i nie jest wymyślone przez LLM:
*„odmawiam, bo próbki 2 i 5 są w relacji sprzeczności z ufnością 0.83"*. To
zdanie jest **opisem faktu geometrycznego**, nie generatem — co ma znaczenie
zarówno dla użycia, jak i dla audytu.

## 3. Algorytm

```python
flow = RelationFlow.load(ckpt, device="cuda")

def orbit_entropy(answers: list[torch.Tensor], id_syn: int, id_ant: int):
    V = torch.stack(answers)                       # [k, D] — przestrzeń operatora
    V = flow.refine(V) if raw else V
    rel, conf = flow.model.decode(V.unsqueeze(1).expand(-1,k,-1).reshape(-1, V.shape[-1]),
                                  V.unsqueeze(0).expand(k,-1,-1).reshape(-1, V.shape[-1]))
    rel, conf = rel.view(k,k), conf.view(k,k)
    appl = torch.stack([flow.applicability(V, r) for r in range(R)]).mean(0)  # [k]

    # orbity = składowe spójne po krawędziach syn
    orbits = connected_components(adjacency=(rel == id_syn) & (conf >= TAU))
    H  = orbit_entropy_from_sizes([len(o) for o in orbits])
    kap = conf[(rel == id_ant)].max() if (rel == id_ant).any() else 0.0
    a  = appl.mean()
    return decide(H, kap, a)          # ODPOWIEDZ / ODMOWA / POZA_ZAKRESEM
```

Koszt dla `k = 10`: 45 par do dekodera — ułamek procenta kosztu wygenerowania
samych odpowiedzi. Metoda nadaje się więc do użycia w pętli produkcyjnej, w
przeciwieństwie do entropii opartej na NLI (45 wywołań modelu generującego).

## 4. Metryki

| metryka | definicja | baseline |
|---|---|---|
| **AUROC poprawności** | zdolność rozdzielenia odpowiedzi poprawnych od błędnych | semantic entropy, SelfCheckGPT, entropia logprobów |
| **risk-coverage** | dokładność przy odrzuceniu `x%` najgorszych | jw. — główny rysunek |
| **konflikt-recall** | przy abstainie: ile przypadków **naprawdę sprzecznych** zostało wyłapanych | nowa metryka — binarne metody nie mają jak jej zoptymalizować |
| **precision „poza zakresem"** | wśród zgłoszeń „poza zakresem": ile pytań faktycznie było poza wiedzą | nowa metryka; gold z budowy zadania |
| `ECE` decyzji | kalibracja samej decyzji odmowy | pokaże, czy progi są uczciwe |

**Gold do konflikt-recall** budujemy sztucznie: dla części pytań wymuszamy
sprzeczne odpowiedzi (temperatura + prompt „podaj przeciwną tezę") i sprawdzamy,
czy metoda rozpoznaje konflikt, a nie tylko różnorodność słów. To jedyny uczciwy
sposób zmierzenia różnicy wobec entropii binarnej.

**Test przesądzający:** zbiór pytań, na których model jest **pewny, ale
gadatliwy** (długie parafrazy tej samej odpowiedzi). Entropia semantyczna
podwyższa niepewność (błędnie), nasza nie. Jeśli ten test przechodzimy, mamy
 przewagę, której nie da się podrobić dostrojeniem progu.

## 5. Fazy

| faza | czas | artefakt |
|---|---|---|
| 4a | 1 dzień | test go/no-go (poniżej) |
| 4b | 3 dni | `orbit_entropy()` + testy jednostkowe na sztucznych próbkach |
| 4c | 4 dni | 3 benchmarki QA + baseline'y (sem-ent, SelfCheckGPT, logprob-ent) |
| 4d | 3 dni | kalibracja `τ_α` z `ECE`, risk-coverage, raport |

## 6. Ryzyka

| ryzyko | mitygacja |
|---|---|
| dekoder nie działa na zdaniach | F4 z `shadow vectors` = zdaniami; bez tego pomysł odpada — to główne ryzyko |
| `k` zbyt małe, by entropia miała sens | krzywa `k ∈ {3,5,10,20}` — pokazać, przy jakim `k` metoda stabilizuje |
| odpowiedzi wieloczęściowe (jedna odpowiedź = kilka faktów) | entropia per zdanie/fakt, nie per odpowiedź; agregacja przez medianę |
| kalibracja `applicability` słaba na zdaniach | policzyć `ECE` osobno; jeśli > 0.1, sygnału 3 nie używać i raportować tylko `H_orb` + `κ` |

## 7. Test go/no-go (1 dzień)

200 pytań, `k = 8` próbek, temperatura 0.9. Dla każdej odpowiedzi ręcznie
oznaczyć: (a) czy poprawna, (b) czy orbity dekodera zgadzają się z Twoim osądem
(„to jest to samo znaczenie?"). Jeśli zgodność orbit z osądem ludzkim < 75%,
entropia orbitowa nie ma fundamentu i nie warto budować reszty.

## 8. Punkty odniesienia

- Farquhar i in., *Semantic entropy is a hallucination signal…* (Nature 2024).
- *Beyond Semantic Entropy: Boosting LLM Uncertainty Quantification with
  Pairwise Semantic Similarity* (2025) — bezpośredni poprzednik; ich krok to
  podobieństwo par, nasz krok to **typ relacji**.
- `FASE` (2026), *Towards Transparent AI Grading* (2025) — dowód, że rodzina
  metod jest aktywna i że warto zaproponować wariant typowany.
- W repo: `core/heads/relation_decoder.py`, `core/heads/applicability.py`,
  `core/metrics/offline.py` (`calibration_ece`, `applicability_auc`).
