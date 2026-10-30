"""The training *runner* end to end: dataset files -> report.json -> artifact.

The runner is where a run becomes an artifact, and it is the part the unit
tests of the trainer never touched: dial calibration, the choice between the
last-epoch and the best checkpoint, and the report that documents both.
"""

import json
import pickle
import random

import torch

from relgat_projector_apps.trainers import wiazka_trainer as app
from relgat_projector.inference import RelationFlow


def _write_dataset(tmp_path, n_nodes=64, n_rels=3, dim=16):
    rng = random.Random(7)
    triplets = []
    for i in range(n_nodes):
        for _ in range(3):
            j = rng.randrange(n_nodes)
            if j != i:
                triplets.append([i, j, f"r{rng.randrange(n_rels)}"])
    nodes = tmp_path / "emb.pickle"
    with open(nodes, "wb") as f:
        pickle.dump({i: torch.randn(dim).numpy() for i in range(n_nodes)}, f)
    rels = tmp_path / "rel2idx.json"
    rels.write_text(json.dumps({f"r{i}": i for i in range(n_rels)}),
                    encoding="utf-8")
    edges = tmp_path / "triplets.json"
    edges.write_text(json.dumps(triplets), encoding="utf-8")
    return str(nodes), str(rels), str(edges)


def test_runner_produces_a_complete_report(tmp_path, monkeypatch):
    nodes, rels, edges = _write_dataset(tmp_path)
    out = tmp_path / "run"
    monkeypatch.setattr(
        "sys.argv",
        ["wiazka", f"--nodes-embeddings-path={nodes}", f"--relations-mapping={rels}",
         f"--relations-triplets={edges}", "--op-dim=32", "--num-generators=8",
         "--gat-num-layers=1", "--gat-heads=1", "--epochs=3", "--batch-size=32",
         "--train-ratio=0.8", "--device=cpu", "--probe-size=64",
         "--log-every-n-steps=10000", "--qr-relations=r0", f"--save-dir={out}"],
    )
    app.main()

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    # every checkpoint variant and both dial variants are reported, so an
    # overfitted dial or a lucky checkpoint is visible without rerunning
    assert set(report["metrics_by_checkpoint"]) >= {"last"}
    assert set(report["metrics_by_checkpoint"]["last"]) == {
        "trained_dial", "calibrated_dial"}
    assert report["shipped_checkpoint"] in report["metrics_by_checkpoint"]
    assert report["checkpoint_scores"][report["shipped_checkpoint"]] == max(
        report["checkpoint_scores"].values())
    assert report["final_metrics"]["eval/mrr"] > 0.0
    assert report["final_stage"] in {"F1_nodes", "F2_relations",
                                     "F3_composition", "F4_shadows"}
    # per-epoch loss terms + eval metrics: the only place a dead term shows up
    assert len(report["history"]) == 3
    assert "L_edge" in report["history"][0] and "eval/mrr" in report["history"][0]
    # the dial is fitted per relation, on train edges, inside its valid range
    assert len(report["dial_calibration"]) == 3
    assert all(0.0 <= lam <= 1.0 for lam in report["dial_calibration"].values())
    assert {"relation", "mrr_model", "relation_acc", "transport_lambda"} <= set(
        report["per_relation"][0])
    assert "r0" in report["qr_recall"]

    # the shipped checkpoint loads and answers the served query
    conn = RelationFlow.load(str(out), device="cpu")
    q = conn.production_query(torch.randn(16), "r0")
    assert q.shape[-1] == 16 and torch.isfinite(q).all()
