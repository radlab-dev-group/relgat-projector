"""
Generate the bilingual (pl/en) fixtures of the consistency diagnostics.

Three files are written, all UTF-8, all read by
``relgat_projector_apps.diagnostics.consistency_report``:

* ``documents_pl_en.jsonl`` -- documents whose claims are either consistent or
  drift into self-contradiction (greenwashing, "exception from the exception",
  retracted study results).  ``contradictions`` lists the claim-index pairs
  that genuinely contradict: that is the only gold in this whole fixture, and
  it is gold by construction, not by annotation.
* ``order_tasks_pl_en.jsonl`` -- for every seed word and every *unordered*
  pair of relations, the two chain readings ("r then s" and "s then r")
  written out as text in both languages.
* ``lexicon_pl_en.json`` -- seed pairs of the relations the diagnostics use
  (synonymy, antonymy, fuzzy synonymy, derivation, hyponymy) in both
  languages.  They define the embedder-free baseline transport, so no trained
  model is needed to run 12a/12b at all.

The texts are written by hand, not machine translated: the point of the
English side is to be a *parallel* input, so that the same document can be
scored in either language and the difference can be attributed to the space.

Usage::

    python -m relgat_projector_apps.diagnostics.make_consistency_fixture \\
        --out-dir data/consistency
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

# --------------------------------------------------------------------- documents
# label: "coherent" | "self_negating" | "greenwashing" | "exception_to_exception"
# A positive label means the document *does* drift into its own negation.
DOCUMENTS: List[Dict] = [
    {
        "id": "co-01-klimat",
        "label": "coherent",
        "topic": {"pl": "klimat", "en": "climate"},
        "claims": [
            {"pl": "Nasza firma osiągnie neutralność klimatyczną do 2035 roku.",
             "en": "Our company will reach climate neutrality by 2035."},
            {"pl": "Wszystkie nowe zakłady zasilamy w całości energią ze źródeł odnawialnych.",
             "en": "All new plants run entirely on renewable energy."},
            {"pl": "Co roku publikujemy audytowany raport emisji zakresów jeden, dwa i trzy.",
             "en": "Every year we publish an audited report of scope one, two and three emissions."},
            {"pl": "Ograniczanie śladu węglowego jest priorytetem zarządu.",
             "en": "Cutting the carbon footprint is a board priority."},
        ],
        "contradictions": [],
    },
    {
        "id": "gw-01-klimat",
        "label": "greenwashing",
        "topic": {"pl": "klimat", "en": "climate"},
        "claims": [
            {"pl": "Nasza firma zobowiązuje się do neutralności klimatycznej, o ile pozwolą na to warunki rynkowe.",
             "en": "Our company commits to climate neutrality, provided market conditions allow it."},
            {"pl": "W miarę możliwości zwiększamy udział energii odnawialnej.",
             "en": "We increase the share of renewable energy as far as possible."},
            {"pl": "Raporty emisji publikujemy zgodnie z mającymi zastosowanie przepisami.",
             "en": "We publish emissions reports in line with applicable regulations."},
            {"pl": "W praktyce zdecydowaną większość energii nadal kupujemy z gazu ziemnego.",
             "en": "In practice we still buy most of our energy from natural gas."},
        ],
        "contradictions": [[0, 3], [1, 3]],
    },
    {
        "id": "co-02-prywatnosc",
        "label": "coherent",
        "topic": {"pl": "prywatność", "en": "privacy"},
        "claims": [
            {"pl": "Nie przechowujemy treści wiadomości naszych użytkowników.",
             "en": "We do not store the message content of our users."},
            {"pl": "Logi serwera są usuwane automatycznie po siedmiu dniach.",
             "en": "Server logs are deleted automatically after seven days."},
            {"pl": "Treści wiadomości szyfrujemy tak, aby nikt po naszej stronie nie mógł ich odczytać.",
             "en": "We encrypt message content so that nobody on our side can read it."},
            {"pl": "Dane osobowe przetwarzamy wyłącznie w celu obsługi konta.",
             "en": "We process personal data only to operate the account."},
        ],
        "contradictions": [],
    },
    {
        "id": "sn-02-prywatnosc",
        "label": "self_negating",
        "topic": {"pl": "prywatność", "en": "privacy"},
        "claims": [
            {"pl": "Nie przechowujemy treści wiadomości naszych użytkowników.",
             "en": "We do not store the message content of our users."},
            {"pl": "Treści wiadomości szyfrujemy tak, aby nikt po naszej stronie nie mógł ich odczytać.",
             "en": "We encrypt message content so that nobody on our side can read it."},
            {"pl": "Nasze systemy przechowują jednak pełną treść korespondencji w celu zapewnienia bezpieczeństwa.",
             "en": "Our systems nevertheless store the full content of correspondence for security reasons."},
            {"pl": "Nasz personel ma dostęp do treści wiadomości, gdy wymaga tego obsługa zgłoszenia.",
             "en": "Our staff may access message content when ticket handling requires it."},
        ],
        "contradictions": [[0, 2], [0, 3]],
    },
    {
        "id": "co-03-jakosc",
        "label": "coherent",
        "topic": {"pl": "jakość produktu", "en": "product quality"},
        "claims": [
            {"pl": "Nasze urządzenie przechodzi testy wytrzymałościowe w każdej serii produkcyjnej.",
             "en": "Our device passes endurance testing in every production batch."},
            {"pl": "Gwarancja obejmuje wymianę sprzętu przez trzy lata.",
             "en": "The warranty covers device replacement for three years."},
            {"pl": "Reklamacje rozpatrujemy w ciągu czternastu dni.",
             "en": "We settle complaints within fourteen days."},
            {"pl": "Awaryjność naszych urządzeń jest niższa niż jeden procent.",
             "en": "The failure rate of our devices is below one percent."},
        ],
        "contradictions": [],
    },
    {
        "id": "sn-03-jakosc",
        "label": "self_negating",
        "topic": {"pl": "jakość produktu", "en": "product quality"},
        "claims": [
            {"pl": "Nasze urządzenie przechodzi testy wytrzymałościowe w każdej serii produkcyjnej.",
             "en": "Our device passes endurance testing in every production batch."},
            {"pl": "Awaryjność naszych urządzeń jest niższa niż jeden procent.",
             "en": "The failure rate of our devices is below one percent."},
            {"pl": "Warunki gwarancji nie obejmują usterek wynikających z normalnej eksploatacji.",
             "en": "The warranty terms exclude defects resulting from normal use."},
            {"pl": "Statystyki awaryjności dotyczą wyłącznie warunków laboratoryjnych.",
             "en": "The failure statistics cover laboratory conditions only."},
            {"pl": "Testy wytrzymałościowe przeprowadzamy na losowo wybranej części serii.",
             "en": "We run endurance tests on a randomly selected part of the batch."},
        ],
        "contradictions": [[0, 4], [1, 3]],
    },
    {
        "id": "ex-01-regulamin",
        "label": "exception_to_exception",
        "topic": {"pl": "regulamin", "en": "regulations"},
        "claims": [
            {"pl": "Wstęp do budynku jest dozwolony dla wszystkich interesantów.",
             "en": "Entry to the building is permitted for all visitors."},
            {"pl": "Zakaz wstępu dotyczy osób bez identyfikatora.",
             "en": "Entry is prohibited for persons without a badge."},
            {"pl": "Od zakazu, o którym mowa w ustępie drugim, istnieje wyjątek dla osób z przepustką czasową.",
             "en": "An exception to the prohibition in paragraph two exists for holders of a temporary pass."},
            {"pl": "Wyjątku, o którym mowa w ustępie trzecim, nie stosuje się w dni posiedzeń rady.",
             "en": "The exception in paragraph three does not apply on the days of the council sessions."},
            {"pl": "W dni posiedzeń rady wstęp mają wyłącznie osoby zaproszone, z wyjątkiem przypadków, o których mowa w ustępie trzecim.",
             "en": "On council session days only invited persons may enter, except in the cases referred to in paragraph three."},
        ],
        "contradictions": [[0, 1], [3, 4]],
    },
    {
        "id": "co-04-badanie",
        "label": "coherent",
        "topic": {"pl": "badanie", "en": "study"},
        "claims": [
            {"pl": "Grupa badawcza poprawiła wyniki testu o dwanaście punktów.",
             "en": "The study group improved its test scores by twelve points."},
            {"pl": "Różnica jest istotna statystycznie na poziomie jeden procent.",
             "en": "The difference is statistically significant at the one percent level."},
            {"pl": "Efekt utrzymał się na teście kontrolnym po trzech miesiącach.",
             "en": "The effect persisted in the follow-up test after three months."},
            {"pl": "Wyniki potwierdzono na niezależnej próbie stu osób.",
             "en": "The results were replicated on an independent sample of one hundred persons."},
        ],
        "contradictions": [],
    },
    {
        "id": "sn-04-badanie",
        "label": "self_negating",
        "topic": {"pl": "badanie", "en": "study"},
        "claims": [
            {"pl": "Grupa badawcza poprawiła wyniki testu o dwanaście punktów.",
             "en": "The study group improved its test scores by twelve points."},
            {"pl": "Różnica jest istotna statystycznie na poziomie jeden procent.",
             "en": "The difference is statistically significant at the one percent level."},
            {"pl": "Żadna różnica między grupami nie była istotna statystycznie.",
             "en": "No difference between the groups was statistically significant."},
            {"pl": "Wyniki testu grupy badawczej pozostały bez zmian.",
             "en": "The test scores of the study group remained unchanged."},
        ],
        "contradictions": [[0, 3], [1, 2]],
    },
    {
        "id": "gw-05-recykling",
        "label": "greenwashing",
        "topic": {"pl": "recykling", "en": "recycling"},
        "claims": [
            {"pl": "Nasze butelki powstają w stu procentach z plastiku z recyklingu.",
             "en": "Our bottles are made of one hundred percent recycled plastic."},
            {"pl": "Dbamy o gospodarkę obiegu zamkniętego.",
             "en": "We care about the circular economy."},
            {"pl": "Butelki zawierają trzydzieści procent plastiku z recyklingu, pozostałą część stanowi surowiec pierwotny.",
             "en": "The bottles contain thirty percent recycled plastic, the rest is virgin material."},
            {"pl": "Nasze opakowania nadają się w pełni do recyklingu.",
             "en": "Our packaging is fully recyclable."},
        ],
        "contradictions": [[0, 2]],
    },
    {
        "id": "co-06-zdalna",
        "label": "coherent",
        "topic": {"pl": "praca zdalna", "en": "remote work"},
        "claims": [
            {"pl": "Pracownicy mogą pracować zdalnie przez trzy dni w tygodniu.",
             "en": "Employees may work remotely three days a week."},
            {"pl": "Firma dopłaca do pracy zdalnej ryczałtem miesięcznym.",
             "en": "The company pays a flat monthly allowance for remote work."},
            {"pl": "Spotkania zespołu planujemy w dni stacjonarne.",
             "en": "We schedule team meetings on the on-site days."},
            {"pl": "Sprzęt do pracy zdalnej zapewnia pracodawca.",
             "en": "The employer provides the equipment for remote work."},
        ],
        "contradictions": [],
    },
    {
        "id": "sn-06-zdalna",
        "label": "self_negating",
        "topic": {"pl": "praca zdalna", "en": "remote work"},
        "claims": [
            {"pl": "Pracownicy mogą pracować zdalnie przez trzy dni w tygodniu.",
             "en": "Employees may work remotely three days a week."},
            {"pl": "Praca zdalna nie jest formą pracy uznaną w regulaminie.",
             "en": "Remote work is not a form of work recognised by the regulations."},
            {"pl": "Każdorazowa zgoda na pracę zdalną wymaga pisemnej akceptacji przełożonego.",
             "en": "Every approval of remote work requires the written consent of the supervisor."},
            {"pl": "Firma dopłaca do pracy zdalnej ryczałtem miesięcznym.",
             "en": "The company pays a flat monthly allowance for remote work."},
        ],
        "contradictions": [[0, 1], [1, 3]],
    },
    {
        "id": "co-07-tramwaj",
        "label": "coherent",
        "topic": {"pl": "inwestycja", "en": "investment"},
        "claims": [
            {"pl": "Nowa linia tramwajowa skróci dojazd do centrum o dziesięć minut.",
             "en": "The new tram line will shorten the trip to the city centre by ten minutes."},
            {"pl": "Trasa będzie gotowa do końca przyszłego roku.",
             "en": "The route will be completed by the end of next year."},
            {"pl": "Połowę kosztu inwestycji pokryją fundusze europejskie.",
             "en": "European funds will cover half of the investment cost."},
            {"pl": "W czasie budowy utrzymamy komunikację zastępczą.",
             "en": "During construction we will keep a replacement bus service running."},
        ],
        "contradictions": [],
    },
]

#: labels whose documents are supposed to look self-contradictory
POSITIVE_LABELS = ("self_negating", "greenwashing", "exception_to_exception")

# ---------------------------------------------------------------------- lexicon
#: relation -> [[base, target], ...] per language.  ``T_rel(base) ~= target``.
LEXICON: Dict[str, Dict[str, List[Sequence[str]]]] = {
    "synonimia": {
        "pl": [["duży", "wielki"], ["mały", "niewielki"], ["ładny", "piękny"],
               ["smutny", "ponury"], ["wesoły", "radosny"], ["bogaty", "zamożny"],
               ["łatwy", "prosty"], ["trudny", "skomplikowany"], ["ważny", "istotny"],
               ["miły", "uprzejmy"], ["mądry", "bystry"], ["cichy", "niemy"],
               ["mówić", "rzec"], ["zaczynać", "inicjować"], ["kończyć", "finalizować"],
               ["pomagać", "wspierać"], ["widzieć", "dostrzegać"]],
        "en": [["big", "large"], ["small", "little"], ["pretty", "beautiful"],
               ["sad", "unhappy"], ["happy", "cheerful"], ["rich", "wealthy"],
               ["easy", "simple"], ["difficult", "complex"], ["important", "significant"],
               ["polite", "courteous"], ["clever", "smart"], ["quiet", "silent"],
               ["say", "state"], ["begin", "commence"], ["finish", "complete"],
               ["help", "assist"], ["see", "perceive"]],
    },
    "antonimia właściwa": {
        "pl": [["duży", "mały"], ["ciepły", "zimny"], ["szybki", "wolny"],
               ["jasny", "ciemny"], ["drogi", "tani"], ["mądry", "głupi"],
               ["czysty", "brudny"], ["ciężki", "lekki"], ["otwarty", "zamknięty"],
               ["stary", "nowy"], ["wysoki", "niski"], ["głęboki", "płytki"],
               ["łatwy", "trudny"], ["prawda", "kłamstwo"], ["wojna", "pokój"],
               ["dzień", "noc"], ["zysk", "strata"], ["początek", "koniec"],
               ["miły", "nieprzyjemny"], ["zdrowy", "chory"]],
        "en": [["big", "small"], ["warm", "cold"], ["fast", "slow"],
               ["bright", "dark"], ["expensive", "cheap"], ["wise", "foolish"],
               ["clean", "dirty"], ["heavy", "light"], ["open", "closed"],
               ["old", "new"], ["tall", "short"], ["deep", "shallow"],
               ["easy", "hard"], ["truth", "lie"], ["war", "peace"],
               ["day", "night"], ["profit", "loss"], ["beginning", "end"],
               ["pleasant", "unpleasant"], ["free", "bound"]],
    },
    "fuzzynimia": {
        "pl": [["duży", "spory"], ["mały", "nikły"], ["zimny", "chłodny"],
               ["szybki", "żwawy"], ["ważny", "niebagatelny"], ["brzydki", "nieładny"],
               ["gorący", "ciepły"], ["mądry", "rozsądny"]],
        "en": [["big", "fairly large"], ["small", "scant"], ["cold", "chilly"],
               ["fast", "brisk"], ["important", "nontrivial"], ["ugly", "plain"],
               ["hot", "warm"], ["wise", "sensible"]],
    },
    "derywacyjność": {
        "pl": [["szybki", "szybkość"], ["ciepły", "ciepło"], ["piękny", "piękno"],
               ["duży", "wielkość"], ["dom", "domowy"], ["czytać", "czytanie"],
               ["budować", "budowa"], ["kolor", "kolorowy"]],
        "en": [["fast", "fastness"], ["warm", "warmth"], ["beautiful", "beauty"],
               ["big", "bigness"], ["home", "homely"], ["read", "reading"],
               ["build", "building"], ["colour", "colourful"]],
    },
    "antonimia komplementarna": {
        "pl": [["mąż", "żona"], ["rodzic", "dziecko"], ["przód", "tył"],
               ["góra", "dół"], ["start", "meta"], ["pytanie", "odpowiedź"],
               ["życie", "śmierć"], ["dzień", "noc"]],
        "en": [["husband", "wife"], ["parent", "child"], ["front", "back"],
               ["top", "bottom"], ["start", "finish"], ["question", "answer"],
               ["life", "death"], ["day", "night"]],
    },
    "podtyp nieokreślony": {
        "pl": [["kot", "zwierzę"], ["jabłko", "owoc"], ["samochód", "pojazd"],
               ["krzesło", "mebel"], ["róża", "kwiat"], ["skrzypce", "instrument"],
               ["kaszel", "objaw"], ["kredyt", "zobowiązanie"]],
        "en": [["cat", "animal"], ["apple", "fruit"], ["car", "vehicle"],
               ["chair", "furniture"], ["rose", "flower"], ["violin", "instrument"],
               ["cough", "symptom"], ["loan", "liability"]],
    },
}

#: seed words of the order tasks: short, and present in both spaces
SEED_WORDS: List[Dict[str, str]] = [
    {"pl": "duży", "en": "big"},
    {"pl": "ciepły", "en": "warm"},
    {"pl": "szybki", "en": "fast"},
    {"pl": "jasny", "en": "bright"},
    {"pl": "drogi", "en": "expensive"},
    {"pl": "czysty", "en": "clean"},
    {"pl": "ciężki", "en": "heavy"},
    {"pl": "otwarty", "en": "open"},
    {"pl": "stary", "en": "old"},
    {"pl": "łatwy", "en": "easy"},
    {"pl": "mądry", "en": "wise"},
    {"pl": "głęboki", "en": "deep"},
    {"pl": "ważny", "en": "important"},
    {"pl": "wolny", "en": "slow"},
]

#: how a relation is spelled inside a chain description
RELATION_TEMPLATES: Dict[str, Dict[str, str]] = {
    "synonimia": {"pl": "będący synonimem ({inner})",
                  "en": "being a synonym of ({inner})"},
    "antonimia właściwa": {"pl": "będący przeciwieństwem ({inner})",
                           "en": "being the opposite of ({inner})"},
    "antonimia komplementarna": {"pl": "będący dopełnieniem ({inner})",
                                 "en": "being the complement of ({inner})"},
    "fuzzynimia": {"pl": "znaczący mniej więcej co ({inner})",
                   "en": "meaning roughly ({inner})"},
    "derywacyjność": {"pl": "wyprowadzony od ({inner})",
                      "en": "derived from ({inner})"},
    "podtyp nieokreślony": {"pl": "będący rodzajem ({inner})",
                            "en": "being a kind of ({inner})"},
}

#: unordered relation pairs of the order tasks
ORDER_PAIRS: List[Sequence[str]] = [
    ["synonimia", "antonimia właściwa"],
    ["synonimia", "fuzzynimia"],
    ["antonimia właściwa", "derywacyjność"],
    ["synonimia", "podtyp nieokreślony"],
    ["antonimia właściwa", "antonimia komplementarna"],
    ["fuzzynimia", "derywacyjność"],
]


def compose(chain: Sequence[str], word: str, lang: str) -> str:
    """Write a relation chain as nested text, innermost relation last."""
    out = word
    for rel in reversed(chain):
        out = RELATION_TEMPLATES[rel][lang].format(inner=out)
    return out


def order_tasks() -> List[Dict]:
    tasks = []
    for pair in ORDER_PAIRS:
        first, second = pair
        for word in SEED_WORDS:
            tasks.append({
                "id": f"{first}__{second}__{word['pl']}",
                "relations": [first, second],
                "word": dict(word),
                "chain_a": [first, second],
                "chain_b": [second, first],
                "desc_a": {lang: compose([first, second], word[lang], lang)
                           for lang in ("pl", "en")},
                "desc_b": {lang: compose([second, first], word[lang], lang)
                           for lang in ("pl", "en")},
                "prompt": {
                    "pl": f"Zastosuj do «{word['pl']}» relację {first}, a potem {second}.",
                    "en": f"Apply the relation {first} to «{word['pl']}», then {second}.",
                },
            })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out-dir", default="data/consistency")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    documents = os.path.join(args.out_dir, "documents_pl_en.jsonl")
    with open(documents, "w", encoding="utf-8") as f:
        for doc in DOCUMENTS:
            assert doc["label"] in POSITIVE_LABELS + ("coherent",), doc["id"]
            n = len(doc["claims"])
            for a, b in doc["contradictions"]:
                assert 0 <= a < n and 0 <= b < n and a != b, (doc["id"], a, b)
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    tasks = order_tasks()
    task_path = os.path.join(args.out_dir, "order_tasks_pl_en.jsonl")
    with open(task_path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    lexicon_path = os.path.join(args.out_dir, "lexicon_pl_en.json")
    with open(lexicon_path, "w", encoding="utf-8") as f:
        json.dump({
            "note": "T_rel(base) ~= target; bilingual seed pairs of the relations "
                    "the consistency diagnostics use",
            "positive_labels": list(POSITIVE_LABELS),
            "relation_templates": RELATION_TEMPLATES,
            "pairs": LEXICON,
        }, f, ensure_ascii=False, indent=2)
        f.write("\n")

    n_claims = sum(len(d["claims"]) for d in DOCUMENTS)
    n_gold = sum(len(d["contradictions"]) for d in DOCUMENTS)
    print(f"[fixture] {documents}: {len(DOCUMENTS)} documents, {n_claims} claims, {n_gold} gold contradictions")
    print(f"[fixture] {task_path}: {len(tasks)} order tasks")
    print(f"[fixture] {lexicon_path}: {sum(len(v['pl']) for v in LEXICON.values())} pl / "
          f"{sum(len(v['en']) for v in LEXICON.values())} en seed pairs")


if __name__ == "__main__":
    main()
