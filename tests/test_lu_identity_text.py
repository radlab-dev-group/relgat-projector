"""Tests for the LU identity-text recipe used to embed graph nodes.

The recipe lives in the plwordnet repository (``lu_identity_text``) because it
is the same text the Milvus pipeline embeds; this repo consumes it through
``build_same_embedder_dataset``.  The tests import it from a checkout next to
this one and skip when the checkout is not there, so the suite never depends
on a sibling repo being installed.

What is pinned here is the property the node embeddings depend on: the lemma
must be part of the text.  A graph node embedded from a bare gloss loses the
identity of the unit, and 54 % of the nodes then share a vector with another
node (measured on ``data/same-embedder``, the lemma-only build).
"""

import os
import sys
import types

import pytest

PLWORDNET_REPO = os.path.abspath(os.environ.get(
    "PLWORDNET_REPO",
    os.path.join(os.path.dirname(__file__), "..", "..", "plwordnet")))

# ``plwordnet`` is installed in this environment as well, but the test is about
# the *checkout*, not the installed copy: the checkout is what
# ``build_same_embedder_dataset`` imports from.  So it goes first on the path
# (before site-packages), and the test is skipped when the checkout is absent.
if not os.path.isdir(PLWORDNET_REPO):
    pytest.skip("the plwordnet checkout is not next to this repo "
                f"({PLWORDNET_REPO}); set PLWORDNET_REPO to run it",
                allow_module_level=True)
sys.path.insert(0, PLWORDNET_REPO)

from plwordnet_ml.embedder.generator.base_embeddings.lexical_unit import (  # noqa: E402
    LU_POS_NAMES,
    lu_identity_text,
)


def _lu(lemma="absolutny", pos=4, domain="og.", definition=None, examples=()):
    comment = types.SimpleNamespace(
        base_domain=domain,
        definition=definition,
        usage_examples=[types.SimpleNamespace(text=t) for t in examples],
    )
    return types.SimpleNamespace(lemma=lemma, pos=pos, comment=comment)


def test_identity_text_carries_the_identity_of_the_unit():
    lu = _lu(definition="oparte na zasadach absolutyzmu.",
             examples=["Różnica między monarchią absolutną a despotyczną."])
    text = lu_identity_text(lu)
    assert text.startswith("absolutny (przymiotnik)")
    assert "[og.]" in text
    assert "oparte na zasadach absolutyzmu." in text
    assert "Różnica między monarchią absolutną" in text


def test_identity_text_survives_a_missing_comment():
    """Proper names have no definition at all; the lemma still anchors them."""
    lu = types.SimpleNamespace(lemma="Naseby", pos=6, comment=None)
    assert lu_identity_text(lu) == "Naseby (obcy)"
    empty = _lu(lemma="hejszanzaur", pos=2, definition=None, domain=None)
    assert lu_identity_text(empty) == "hejszanzaur (rzeczownik)"


def test_pos_is_never_invented():
    """An unknown code is rendered as a marker, not as a guessed Polish word."""
    assert "pos=99" in lu_identity_text(_lu(pos=99))
    for code, name in LU_POS_NAMES.items():
        assert name in lu_identity_text(_lu(pos=code))


def test_examples_are_capped_and_the_text_is_bounded():
    lu = _lu(definition="definicja", examples=[f"przykład {i}" for i in range(9)])
    assert "przykład 1" in lu_identity_text(lu, max_examples=2)
    assert "przykład 2" not in lu_identity_text(lu, max_examples=2)
    long_lu = _lu(definition="x" * 4000)
    assert len(lu_identity_text(long_lu, max_chars=128)) <= 128


def test_identity_text_separates_units_that_share_a_definition():
    """The whole point: the same gloss on two lemmas is two different texts."""
    a = _lu(lemma="kimonowy", definition="z kimonu")
    b = _lu(lemma="daglezjowy", definition="z kimonu")
    assert lu_identity_text(a) != lu_identity_text(b)
    # and the lemma-only recipe this replaces could not tell them apart
    assert a.lemma != b.lemma


def test_builder_imports_the_same_function():
    """The builder must not carry its own copy of the recipe."""
    from relgat_projector_apps.diagnostics import build_same_embedder_dataset as b

    assert b._import_from_plwordnet(PLWORDNET_REPO, "lu_identity_text") \
        is lu_identity_text
