# 11. Synteza pojęcia z ograniczeniami relacyjnymi (QP zamiast szukania)

**Teza w jednym zdaniu:** ponieważ `T_r` jest **odwracalne i liniowe**, zdanie
„znajdź to, co jest przeciwieństwem X w rejestrze Y, ale nie kojarzy się z Z"
przestaje być promptem, a staje się **układem ograniczeń** z rozwiązaniem
zamkniętym — czyli kreacją pojęcia, którego może w ogóle nie być w indeksie.

**Prymitywy:** P1 `T_r` liniowe/odwracalne · P3 dial (ciagłość rozwiązań) ·
P5 `applicability` (walidacja rozwiązania)
**Wysiłek:** 1,5 tygodnia · **Zależności:** operator + dowolny indeks wektorów

---

## 1. Problem

Wyszukiwanie cosine odpowiada na jedno pytanie: „co jest blisko `q`?". Cała
reszta jest wymuszana na siłę:

- „co jest **przeciwieństwem** X?" — trzeba znać X i jego antonim z góry;
- „co jest podobne do A, ale **nie** podobne do B?" — post-filtr albo
  wymyślne odejmowanie wektorów, bez żadnej gwarancji;
- „co jest w tym samym rejestrze co Y, ale w innej kategorii?" — poza zasięgiem.

Wymaganie negatywne („nie podobne do Z") jest praktycznie najczęstszą potrzebą w
zadaniach projektowych (naming, design, dobór analogii, dobór przykładów w
few-shot), a **żadna** metoda wektorowa go nie obsługuje natywnie — bo cosine
jest miarą, a nie logiką.

## 2. Mechanizm i formalizacja

Szukamy `v` spełniającego ograniczenia relacyjne:

    maks   cos( T_r v , a )                      — „cel: r-względnie blisko a"
    war.   cos( T_s v , b ) ≤ τ                  — „nie za blisko b w sensie s"
           ‖v‖ = 1

Ponieważ `T_r` jest liniowe, `cos(T_r v, a) = vᵀ T_rᵀ a / ‖T_r v‖`, a dla
operatora z grupy podobieństwa `T_rᵀ T_r = s_r² I` (izometria do skali!), więc

    cos(T_r v, a) = cos(v, T_r⁻¹ a)      — dokładnie!

Ograniczenie relacyjne **redukuje się do zwykłego ograniczenia cosine wobec
obróconego wektora** `a' = T_r⁻¹ a`, `b' = T_s⁻¹ b`. Całe zadanie to:

    maks   vᵀ a' / ‖v‖‖a'‖
    war.   vᵀ b' / ‖v‖‖b'‖ ≤ τ

czyli **maksymalizacja iloczynu skalarnego przy jednym ograniczeniu
półprzestrzennym na jednostkowej sferze**. Rozwiązanie zamknięte:

- jeśli `cos(a', b') > τ` — ograniczenie jest **niesprzeczne z celem** i
  optimum leży na brzegu: `v* ∝ a' − cos(a',b')·b'` (składowa `a'` prostopadła
  do `b'`), znormalizowana;
- w przeciwnym razie `v* = a'/‖a'‖` (ograniczenie nieaktywne).

Dla kilku ograniczeń (`k > 1`) to samo: `v*` to rzut `a'` na dopełnienie
rozpiętości `b'_1..b'_k`, z korektą spełniającą nierówności — czyli mały problem
własny w wymiarze `k+1`. **Zawsze wymiary rzędu kilku**, więc koszt
zaniedbywalny. To jest prawdziwa przewaga strukturalna: izometria operatora
sprowadza egzotyczne ograniczenie do trywialnej geometrii na sferze.

Wariant wielocelowy (wiele „celów" i „zakazów") = problem kwadratowy o macierzy
`G = Σᵢ λᵢ a'_i a'_iᵀ − Σⱼ μⱼ b'_j b'_jᵀ`, rozwiązania z wektora własnego.
Wymiar problemu = liczba ograniczeń, nie wymiar przestrzeni.

## 3. Algorytm

```python
flow = RelationFlow.load(ckpt)

def synthesize(goal, forbid, r="syn", s="ant", tau=0.55, lam=None):
    a = flow.model.inverse(goal,   ID[r])          # T_r⁻¹ a — obrócenie celu
    b = flow.model.inverse(forbid, ID[s])
    ca = a / a.norm(); cb = b / b.norm()
    c = float(ca @ cb)
    v = ca if c <= tau else (ca - c * cb)
    v = v / v.norm()
    return v, check(flow, v)                       # walidacja (§4)

def check(flow, v):
    return dict(ok_goal   = cos(flow.apply(v, ID_r), goal)  >= cel_min,
                ok_forbid = cos(flow.apply(v, ID_s), forbid) <= tau,
                on_manifold = omd_like(flow, v),             # czy to nie szum
                appl              = flow.applicability(v, ID_r).item())
```

**Kluczowy element, którego nie wolno pominąć:** rozwiązanie musi zostać
**zweryfikowane przed wysłaniem do indeksu**. `v*` leży na sferze, ale może
leżeć poza regionem, w którym cosine cokolwiek znaczy (poza manifoldem).
Kontrola: `applicability(v*, ·)` i porównanie `OMD` — jeśli niskie, wynik
odrzucać z komunikatem „brak pojęcia spełniającego wymagania", a nie
„najbliższy losowy skutek". To jest dokładnie ta uczciwość, której brakuje
metodom generującym wektor zawsze.

## 4. Zastosowania

1. **Naming i design.** „Nazwa jak `Bezpieczeństwo`, bez skojarzenia z
   `Bankowość`, w rejestrze technicznym" → trzy ograniczenia, jedno rozwiązanie.
2. **Dobór przykładów few-shot.** Przykłady muszą być bliskie zadania i
   **dalekie od etykiety docelowej** (żeby nie podpowiadać odpowiedzi) — to
   dosłownie to samo QP.
3. **Testy i zadania.** Rozproszenie (dystraktory): „podobne do poprawnej
   odpowiedzi, ale niepoprawne" = cel `syn`, zakaz `=gold`.
4. **Analogie z przeciwwskazaniami** (własny graf, np. chemiczny/medyczny):
   „analogiczna cząsteczka do X, bez relacji `metabolizuje_jak` Y". Wymaga
   osobnego operatora na tamtej ontologii, ale matematyka identyczna.
5. **Bezpieczeństwo treści:** „odpowiedz na pytanie Q, ale bez treści z
   kategorii Z" — ograniczenie jako filtr na etapie wektora, nie post-filtr.

## 5. Metryki

| metryka | definicja | baseline |
|---|---|---|
| **spełnienie ograniczeń** | % rozwiązań z `ok_forbid = True` | post-filtr cosine: spełnia rzadko przy zachowaniu celu |
| cel (wartość funkcji celu) | `cos(T_r v*, a)` | baseline: `a − b` (odejmowanie wektorów) |
| preferencja ludzka | 50 zadań naming/design, A/B, 3 oceniających | „przepisz prompt + filtr" (realistyczny konkurent, kosztuje wywołanie LLM) |
| **trafienia w indeksie** | ile zapytań o `v*` zwraca ≥1 użyteczny dokument | pokaże, jak często rozwiązanie jest poza danymi |
| odrzucenia poprawne | wśród „brak rozwiązania": ile faktycznie brak | miara uczciwości systemu |

## 6. Fazy

| faza | czas | artefakt |
|---|---|---|
| 11a | 2 dni | solver 1-cel/1-zakaz + **wizualizacja 2D** (sanity: czy `v*` leży tam, gdzie rachunek mówi) |
| 11b | 2 dni | wariant wieloograniczeniowy + walidacja `applicability`/`OMD` |
| 11c | 2 dni | integracja z indeksem (rozwiązanie jako zapytanie) |
| 11d | 4 dni | test A/B preferencji + ablacja post-filtra |

## 7. Ryzyka

| ryzyko | mitygacja |
|---|---|
| rozwiązanie poza manifoldem → retriever zwraca szum | walidacja przed zapytaniem + jawne „brak rozwiązania"; raportować odsetek odrzuceń |
| ograniczenia sprzeczne (brak rozwiązania) | detekcja: `cos(a', b') ≤ τ` — sprawdzalne analitycznie, komunikat zamiast wyniku |
| użytkownik nie umie sformułować ograniczeń | asystent LLM tłumaczy zdanie na ograniczenia; ale **rozwiązanie i tak jest numeryczne i sprawdzalne** |
| `T⁻¹` niedokładne przy włączonym kontekście `ε` | liczyć bez kontekstu (`use_context=False`) — wtedy odwrotność jest dokładna (`CCE ≈ 1e-4`) |

## 8. Test go/no-go

20 wymyślonych par „cel + zakaz" na istniejącym indeksie. Sprawdzić, czy `v*`
spełnia oba warunki oraz czy zwraca coś użytecznego. Jeśli > 50% rozwiązań jest
akceptowalnych i post-filtr baseline osiąga istotnie mniej — pomysł stoi.
Jeden dzień.

## 9. Punkty odniesienia

- Klasyczne „concept blending"/TRIZ — formalizacja intuicyjnych technik
  twórczych; tu z **gwarancją** pozostania w przestrzeni i z jawnością
  ograniczeń.
- Odejmowanie wektorów (`king − man + woman`) — potoczne „analogie wektorowe",
  które nie mają ani uzasadnienia, ani ograniczeń negatywnych.
- W repo: `core/operator/givens.py` (`inverse`, `apply`, `get_relation_matrices`),
  `core/operator/compiled.py` (`CompiledGivensOperator`),
  `core/metrics/offline.py` (`off_manifold_degradation`).
