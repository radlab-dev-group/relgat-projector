"""
Generate the bilingual (pl/en) fixture of the relational-clustering diagnostic.

Idea 01 claims that grouping by *relation type* beats grouping by *distance*,
because cosine puts "the battery is great" and "the battery is terrible" in the
same cloud: they share the topic, the register and the sentence skeleton.  A
benchmark of that claim needs gold labels, and the only honest gold here is
label by construction:

* every item is one **stance** (``+1`` / ``-1``) on one **topic**,
* two wordings of the same stance are the ``syn`` pair of that cluster,
* the same topic in the opposite stance is the ``ant`` pair -- the trap: those
  two are *near* in a cosine space, and a metric clusterer will merge them,
* two different topics are ``neutral``.

So the gold clustering is ``topic x stance``: a method that merges the two
stances of one topic scores ARI < 1 **and** contradiction purity < 1, which is
exactly the failure idea 01 is about.  Two files are written:

* ``stances_pl_en.jsonl`` -- one item per line: ``id``, ``topic``, ``stance``,
  ``text`` in both languages.
* ``pairs_pl_en.jsonl`` -- the gold pair classes (``syn`` / ``ant`` /
  ``neutral``) derived from the items, so a pair-level metric (the go/no-go AUC
  of section 8 of the idea document) can be computed without any re-annotation.

The texts are written by hand, not machine translated: the English side has to
be a *parallel* input, so the same item can be scored in either language and
the difference attributed to the space rather than to the translation.

Usage::

    python -m relgat_projector_apps.diagnostics.make_clustering_fixture \\
        --out-dir data/clustering
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Dict, List, Sequence

# ---------------------------------------------------------------------- topics
# (topic pl/en, positive wording x2, negative wording x2)
#
# The two stances of one topic are written to share their nouns and their
# register on purpose: that is the property the whole idea rests on, so the
# fixture must not make the trap accidental.  ``mixed`` items carry a stance
# clause ("... but ...") and are the case a pure polarity model misses.
TOPICS: List[Dict] = [
    {
        "topic": {"pl": "bateria telefonu", "en": "phone battery"},
        "positive": [
            {"pl": "Bateria telefonu trzyma dwa dni bez ładowarki.",
             "en": "The phone battery lasts two days without a charger."},
            {"pl": "Bateria tego telefonu jest naprawdę wytrzymała.",
             "en": "This phone's battery is genuinely long-lasting."},
        ],
        "negative": [
            {"pl": "Bateria telefonu nie trzyma nawet połowy dnia.",
             "en": "The phone battery does not last half a day."},
            {"pl": "Bateria tego telefonu rozładowuje się w kilka godzin.",
             "en": "This phone's battery drains within a few hours."},
        ],
    },
    {
        "topic": {"pl": "obsługa klienta", "en": "customer service"},
        "positive": [
            {"pl": "Obsługa klienta odpowiedziała w kwadrans.",
             "en": "The customer service replied within fifteen minutes."},
            {"pl": "Biuro obsługi klienta jest bardzo pomocne.",
             "en": "The customer service desk is very helpful."},
        ],
        "negative": [
            {"pl": "Obsługa klienta nie odpowiada od tygodnia.",
             "en": "The customer service has not replied for a week."},
            {"pl": "Biuro obsługi klienta jest bezużyteczne.",
             "en": "The customer service desk is useless."},
        ],
    },
    {
        "topic": {"pl": "praca zdalna", "en": "remote work"},
        "positive": [
            {"pl": "Praca zdalna zwiększyła moją produktywność.",
             "en": "Remote work made me more productive."},
            {"pl": "Pracuję lepiej z domu i chcę przy tym zostać.",
             "en": "I work better from home and want to stay that way."},
        ],
        "negative": [
            {"pl": "Praca zdalna obniżyła moją produktywność.",
             "en": "Remote work made me less productive."},
            {"pl": "Z domu pracuję gorzej i chcę wrócić do biura.",
             "en": "I work worse from home and want back to the office."},
        ],
    },
    {
        "topic": {"pl": "energia jądrowa", "en": "nuclear power"},
        "positive": [
            {"pl": "Energia jądrowa to najtańsze czyste źródło prądu.",
             "en": "Nuclear power is the cheapest clean source of electricity."},
            {"pl": "Powinniśmy stawiać na energetykę jądrową.",
             "en": "We should bet on nuclear energy."},
        ],
        "negative": [
            {"pl": "Energia jądrowa jest zbyt droga i zbyt niebezpieczna.",
             "en": "Nuclear power is too expensive and too dangerous."},
            {"pl": "Nie powinniśmy budować nowych elektrowni jądrowych.",
             "en": "We should not build new nuclear plants."},
        ],
    },
    {
        "topic": {"pl": "tydzień czterodniowy", "en": "four-day week"},
        "positive": [
            {"pl": "Czterodniowy tydzień pracy sprawdził się u nas.",
             "en": "The four-day work week worked for us."},
            {"pl": "Krótszy tydzień pracy utrzymał tę samą sprzedaż.",
             "en": "The shorter week kept the same sales."},
        ],
        "negative": [
            {"pl": "Czterodniowy tydzień pracy nas pogrążył.",
             "en": "The four-day work week sank us."},
            {"pl": "Krótszy tydzień pracy zbił naszą sprzedaż.",
             "en": "The shorter week cut our sales."},
        ],
    },
    {
        "topic": {"pl": "aplikacja mobilna", "en": "mobile app"},
        "positive": [
            {"pl": "Aplikacja mobilna działa płynnie i jest przejrzysta.",
             "en": "The mobile app is smooth and easy to read."},
            {"pl": "Korzystanie z aplikacji mobilnej to przyjemność.",
             "en": "Using the mobile app is a pleasure."},
        ],
        "negative": [
            {"pl": "Aplikacja mobilna wiesza się i jest nieczytelna.",
             "en": "The mobile app freezes and is unreadable."},
            {"pl": "Korzystanie z aplikacji mobilnej to udręka.",
             "en": "Using the mobile app is a torment."},
        ],
    },
    {
        "topic": {"pl": "dostawa paczek", "en": "parcel delivery"},
        "positive": [
            {"pl": "Paczka dotarła dzień przed terminem.",
             "en": "The parcel arrived a day early."},
            {"pl": "Dostawa była szybka i bezproblemowa.",
             "en": "The delivery was fast and trouble-free."},
        ],
        "negative": [
            {"pl": "Paczka dotarła tydzień po terminie.",
             "en": "The parcel arrived a week late."},
            {"pl": "Dostawa była opóźniona i pełna problemów.",
             "en": "The delivery was delayed and full of problems."},
        ],
    },
    {
        "topic": {"pl": "cena abonamentu", "en": "subscription price"},
        "positive": [
            {"pl": "Abonament jest tani w stosunku do tego, co daje.",
             "en": "The subscription is cheap for what it offers."},
            {"pl": "Za tę cenę abonament to okazja.",
             "en": "At this price the subscription is a bargain."},
        ],
        "negative": [
            {"pl": "Abonament jest przereklamowany i za drogi.",
             "en": "The subscription is overrated and overpriced."},
            {"pl": "Za taką cenę abonament to przepłacanie.",
             "en": "At that price the subscription is overpaying."},
        ],
    },
    # ``mixed``: a stance clause inside one sentence -- polarity lexicons and
    # sentiment classifiers miss these, and cosine loves them.
    {
        "topic": {"pl": "bateria telefonu", "en": "phone battery"},
        "id_prefix": "mixed-bat",
        "mixed": True,
        "positive": [
            {"pl": "Bateria telefonu jest pojemna, choć ładowarka jest fatalna.",
             "en": "The phone battery is capacious, although the charger is awful."},
            {"pl": "Bateria trzyma długo, ale wolno się ładuje.",
             "en": "The battery lasts long, but it charges slowly."},
        ],
        "negative": [],
    },
    {
        "topic": {"pl": "obsługa klienta", "en": "customer service"},
        "id_prefix": "mixed-cs",
        "mixed": True,
        "positive": [
            {"pl": "Obsługa klienta jest miła, tylko nie rozwiązuje problemu.",
             "en": "The customer service is friendly, it only never solves the problem."},
            {"pl": "Doradca był uprzejmy, lecz odesłał mnie do regulaminu.",
             "en": "The adviser was polite but sent me back to the terms."},
        ],
        "negative": [],
    },
]

#: gold relation of a pair, read off the labels
PAIR_SYNONYMOUS = "syn"
PAIR_ANTONYMOUS = "ant"
PAIR_NEUTRAL = "neutral"
PAIR_MIXED = "mixed"


def items() -> List[Dict]:
    """One record per stance-item, gold cluster = ``<topic>|<stance>``."""
    rows: List[Dict] = []
    for t_pos, spec in enumerate(TOPICS):
        prefix = spec.get("id_prefix") or f"t{t_pos:02d}"
        mixed = bool(spec.get("mixed"))
        for stance_key in ("positive", "negative"):
            for w, text in enumerate(spec[stance_key]):
                stance = 1 if stance_key == "positive" else -1
                topic = spec["topic"]
                # a "yes, but" item is neither of the two stances: left in the
                # positive cluster it would punish a correct method for being
                # right, so it gets a gold cluster of its own
                tag = "mixed" if mixed else ("pos" if stance > 0 else "neg")
                rows.append({
                    "id": f"{prefix}-{stance_key[0]}{w}",
                    "topic": dict(topic),
                    "stance": stance,
                    "group": f"{topic['pl']}|{tag}",
                    "mixed": mixed,
                    "text": {"pl": text["pl"], "en": text["en"]},
                })
    return rows


def gold_pairs(rows: List[Dict]) -> List[Dict]:
    """Pair classes derived from the gold labels, no annotation involved.

    ``syn``   -- same topic, same stance, different wording (a cluster's own two
                 wordings: they *are* the same position),
    ``ant``   -- same topic, opposite stance: the contradiction a metric
                 clusterer hides inside one cluster,
    ``mixed`` -- same topic, one side hedged ("pojemna, choć ładowarka fatalna"):
                 neither attracted nor contradictory, and the class a polarity
                 model gets wrong in both directions,
    ``neutral`` -- different topics.
    """
    out: List[Dict] = []
    for a, b in itertools.combinations(range(len(rows)), 2):
        ra, rb = rows[a], rows[b]
        same_topic = ra["topic"]["pl"] == rb["topic"]["pl"]
        hedged = ra["mixed"] or rb["mixed"]
        if not same_topic:
            label = PAIR_NEUTRAL
        elif hedged:
            label = PAIR_MIXED
        elif ra["stance"] == rb["stance"]:
            label = PAIR_SYNONYMOUS
        else:
            label = PAIR_ANTONYMOUS
        out.append({"a": a, "b": b, "label": label,
                    "id_a": ra["id"], "id_b": rb["id"],
                    "topic_a": ra["topic"], "topic_b": rb["topic"]})
    return out


def main(argv = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate the bilingual fixture of the relational-clustering "
                    "diagnostic (ideas/01)")
    parser.add_argument("--out-dir", default="data/clustering")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = items()
    seen = set()
    for r in rows:
        assert r["id"] not in seen, f"duplicate id {r['id']}"
        seen.add(r["id"])
        assert r["stance"] in (-1, 1), r["id"]
        assert set(r["text"]) == {"pl", "en"}, r["id"]

    path_items = os.path.join(args.out_dir, "stances_pl_en.jsonl")
    with open(path_items, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pairs = gold_pairs(rows)
    path_pairs = os.path.join(args.out_dir, "pairs_pl_en.jsonl")
    with open(path_pairs, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_ant = sum(1 for p in pairs if p["label"] == PAIR_ANTONYMOUS)
    n_syn = sum(1 for p in pairs if p["label"] == PAIR_SYNONYMOUS)
    n_neu = sum(1 for p in pairs if p["label"] == PAIR_NEUTRAL)
    n_mix = sum(1 for p in pairs if p["label"] == PAIR_MIXED)
    n_groups = len({r["group"] for r in rows})
    print(f"[fixture] {path_items}: {len(rows)} items, {n_groups} gold clusters, "
          f"{len({r['topic']['pl'] for r in rows})} topics")
    print(f"[fixture] {path_pairs}: {len(pairs)} pairs "
          f"({n_syn} syn / {n_ant} ant / {n_mix} mixed / {n_neu} neutral)")


if __name__ == "__main__":
    main()
