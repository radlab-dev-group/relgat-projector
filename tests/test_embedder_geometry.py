"""Tests for the embedder-independence layer.

The method has to run, and mean the same thing, on a base space it was not
tuned on.  These tests pin the three mechanisms that make that true --
a measured :class:`EmbedderProfile`, an ``op_dim`` derived from the spectrum
instead of a constant, and cosine knobs expressed as a ratio of this space's
cosine spread -- plus the honest (held-out) calibration split.
"""

import random

import pytest
import torch

from relgat_projector.core.geometry.embedder import (
    EmbedderProfile,
    auto_op_dim,
)
from relgat_projector.core.model.wiazka import WiazkaModel
from relgat_projector.core.operator import build_operator
from relgat_projector.inference import RelationFlow
from relgat_projector.trainer.wiazka_trainer import (
    REFERENCE_COS_STD_CENTRED,
    REFERENCE_COS_STD_RAW,
    WiazkaConfig,
    WiazkaTrainer,
)


# ------------------------------------------------------------------ profile
def test_profile_measures_anisotropy_and_norms():
    torch.manual_seed(0)
    x = torch.randn(400, 24)
    shared = torch.randn(24)
    x = x + 0.6 * shared                      # a shared direction = anisotropy
    x = x * 2.5                                # and not unit-norm
    pr = EmbedderProfile.from_embeddings(x, cos_pairs=4000, svd_rows=400,
                                         dup_rows=200)
    assert pr.dim == 24 and pr.num_vectors == 400
    assert not pr.unit_norm and abs(pr.norm_mean - 2.5 * 1.0) > 0.1
    assert pr.anisotropy > 0.05, "a shared direction must show up as anisotropy"
    centred = abs(pr.pair_cos_mean_centred)
    assert centred < pr.anisotropy, "centring must reduce the mean pair cosine"


def test_profile_reports_duplicates():
    """Exact duplicate rows inflate the raw baseline; the profile says so."""
    torch.manual_seed(1)
    x = torch.randn(100, 16)
    x = torch.cat([x, x[:20]], dim=0)          # 20 exact duplicates
    pr = EmbedderProfile.from_embeddings(x, cos_pairs=2000, svd_rows=120,
                                         dup_rows=120)
    assert pr.dup_1nn_fraction > 0.0
    # the duplicated rows score cos 1.0, so the mean 1-NN cosine sits above
    # the random-pair level of a space with no duplicates
    random = EmbedderProfile.from_embeddings(
        torch.randn(120, 16), cos_pairs=2000, svd_rows=120, dup_rows=120)
    assert pr.dup_1nn_cos > random.dup_1nn_cos


def test_auto_op_dim_tracks_the_variance_target():
    torch.manual_seed(2)
    # a spectrum with a clear elbow: 4 very strong directions, then a 100x
    # drop.  The coarse profile grid (32/64/...) cannot separate targets
    # below d=32, so this is the cap path; the separation is asserted with
    # an explicit cap, which is exactly the knob ``--op-dim-cap`` exposes.
    s = torch.cat([torch.full((4,), 100.0), torch.full((56,), 1.0)])
    x = torch.randn(600, 60) * s
    pr = EmbedderProfile.from_embeddings(x, cos_pairs=4000, svd_rows=600,
                                         dup_rows=200)
    # the top 4 directions carry ~98.6 % of the variance
    assert pr.explained_variance[str(32)] >= 0.98
    # the answer respects the cap, not just the spectrum
    capped = auto_op_dim(pr, target=0.99, cap=16)
    assert 2 <= capped <= 16
    assert capped < auto_op_dim(pr, target=0.99)


def test_auto_op_dim_stays_below_the_space_dimension():
    torch.manual_seed(3)
    pr = EmbedderProfile.from_embeddings(torch.randn(200, 8), cos_pairs=2000,
                                         svd_rows=200, dup_rows=100)
    for target in (0.5, 0.9, 0.95, 0.99, 1.0):
        assert 2 <= auto_op_dim(pr, target=target) < 8


def test_op_dim_zero_runs_on_a_low_dimensional_embedder():
    """The historical constant (256) could not even be *constructed* here."""
    node2emb = {i: torch.randn(24) for i in range(40)}
    rel2idx = {"r0": 0, "r1": 1}
    edges = [(i, (i + 5) % 40, f"r{i % 2}") for i in range(40)]
    cfg = WiazkaConfig(op_dim=0, op_dim_variance=0.9, num_generators=4,
                       gat_num_layers=1, gat_heads=1, batch_size=16,
                       epochs=1, device="cpu")
    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg)
    assert 2 <= trainer.cfg.op_dim < 24
    assert trainer.model.refine().shape[1] == trainer.cfg.op_dim


# ------------------------------------------------------------- geometry auto
def _tiny_trainer(**kw):
    node2emb = {i: torch.randn(20) for i in range(36)}
    rel2idx = {"r0": 0, "r1": 1}
    edges = [(i, (i + 7) % 36, f"r{i % 2}") for i in range(36)]
    cfg = WiazkaConfig(op_dim=8, num_generators=4, gat_num_layers=1,
                       gat_heads=1, batch_size=16, epochs=1, device="cpu", **kw)
    return WiazkaTrainer(node2emb, rel2idx, edges, cfg)


def test_geometry_off_keeps_the_configured_knobs():
    trainer = _tiny_trainer(geometry_auto=False, contrast_temperature=0.07,
                            anchor_margin=0.05, knn_slack=0.02)
    assert trainer.geometry["contrast_temperature"] == 0.07
    assert trainer.geometry["anchor_margin"] == 0.05
    assert trainer.geometry["knn_slack"] == 0.02


def test_geometry_auto_scales_knobs_by_the_cosine_spread():
    trainer = _tiny_trainer(geometry_auto=True, contrast_temperature=0.07,
                            anchor_margin=0.05, knn_slack=0.02)
    used = trainer.geometry["cos_std_used"]
    reference = trainer.geometry["cos_std_reference"]
    # _tiny_trainer runs base_center=False, so the raw reference is used
    scale = used / reference
    assert used > 0
    assert abs(trainer.geometry["contrast_temperature"] - 0.07 * scale) < 1e-9
    assert abs(trainer.geometry["anchor_margin"] - 0.05 * scale) < 1e-9
    assert abs(trainer.geometry["knn_slack"] - 0.02 * scale) < 1e-9
    # the resolved temperature is what the losses actually use
    assert trainer.contrast_loss.temperature == trainer.geometry[
        "contrast_temperature"]


def test_reference_scale_reproduces_the_shipped_defaults():
    """At the reference cosine spread the scaling must be the identity.

    The reference is the Slowosiec profile of run F4 -- the run the shipped
    knobs were tuned on -- and the raw/centred spread is selected the same
    way the chart is: an as-stored chart (``base_center=False``, which is how
    F4 ran) constrains the raw spread, a centred chart the centred one.
    """
    profile = EmbedderProfile.from_embeddings(
        torch.randn(300, 20), cos_pairs=4000, svd_rows=300, dup_rows=100)
    object.__setattr__(profile, "pair_cos_std", REFERENCE_COS_STD_RAW)
    object.__setattr__(profile, "pair_cos_std_centred", REFERENCE_COS_STD_CENTRED)
    cfg = WiazkaConfig(geometry_auto=True)
    g = WiazkaTrainer._resolve_geometry(profile, cfg)
    assert abs(g["contrast_temperature"] - 0.07) < 1e-6
    assert abs(g["anchor_margin"] - 0.05) < 1e-6
    assert abs(g["knn_slack"] - 0.02) < 1e-6

    # the centred chart takes the centred reference
    cfg = WiazkaConfig(geometry_auto=True, base_center=True)
    g = WiazkaTrainer._resolve_geometry(profile, cfg)
    assert abs(g["contrast_temperature"] - 0.07) < 1e-6

    # and the reference that F4 actually used is what the *raw* chart
    # reproduces: 0.07 / 0.1477 is the ratio that run trained with
    object.__setattr__(profile, "pair_cos_std", 0.14771294593811035)
    object.__setattr__(profile, "pair_cos_std_centred", 0.1400369554758072)
    cfg = WiazkaConfig(geometry_auto=True)
    g = WiazkaTrainer._resolve_geometry(profile, cfg)
    assert abs(g["contrast_temperature"] - 0.07) < 1e-6


# ------------------------------------------------------------- centring
def _model(center_mean=None, base_dim=12, op_dim=6):
    torch.manual_seed(0)
    x = torch.randn(24, base_dim)
    ei = torch.tensor([[i for i in range(20)], [(i + 3) % 24 for i in range(20)]])
    et = torch.tensor([i % 2 for i in range(20)])
    op = build_operator("givens", dim=op_dim, num_rel=2, num_generators=3)
    return WiazkaModel(x, ei, et, 2, op, center_mean=center_mean), x


def test_chart_is_built_on_the_centre_and_export_returns_to_base_space():
    mu = torch.randn(12)
    model, x = _model(center_mean=mu.clone())
    assert model.centered
    chart = model.base_to_op(x)
    expected = (x - mu) @ model.base_to_op.weight.t()
    assert torch.allclose(chart, expected, atol=1e-5)
    # the origin of the chart exports to the centre: the retriever space is
    # the *stored* vectors, not the centred ones
    origin = model._export_linear(torch.zeros_like(chart[:1]))
    assert torch.allclose(origin, mu.unsqueeze(0), atol=1e-5)


def test_uncentred_model_is_unchanged():
    model, x = _model()
    assert not model.centered
    assert getattr(model.base_to_op, "bias", None) is None
    assert "base_to_op.bias" not in model.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_centred_chart_builds_when_the_centre_lives_on_the_training_device():
    """The trainer passes the centre computed on the device copy of the
    embeddings while the module is still being built on the CPU; the
    centering-bias math must not assume both operands share a device."""
    mu = torch.randn(12).cuda()
    model, x = _model(center_mean=mu.clone())
    assert model.centered
    assert torch.allclose(model.base_to_op.bias.cpu(),
                          -(model.base_to_op.weight @ mu.cpu()), atol=1e-5)
    model = model.to("cuda")
    assert model.base_to_op.bias.device.type == "cuda"
    assert torch.allclose(model.center_mean, mu, atol=1e-6)
    with torch.no_grad():
        chart = model.base_to_op(x.cuda())
    assert chart.device.type == "cuda"


def test_centred_checkpoint_round_trips(tmp_path):
    mu = torch.randn(12)
    model, x = _model(center_mean=mu.clone())
    model.save_state_dict if False else None
    from relgat_projector.handlers.wiazka_checkpoint import save_checkpoint

    save_checkpoint(str(tmp_path / "ckpt"), model, {"r0": 0, "r1": 1})
    reloaded = RelationFlow.load(str(tmp_path / "ckpt"), device="cpu").model
    assert reloaded.centered
    assert torch.allclose(reloaded.center_mean, mu, atol=1e-6)
    assert torch.allclose(reloaded.base_to_op(x), model.base_to_op(x), atol=1e-6)


# ------------------------------------------------------------- calibration
def _calib_trainer(calib_ratio=0.5, n=60, n_rel=3):
    node2emb = {i: torch.randn(12) for i in range(n)}
    rel2idx = {f"r{k}": k for k in range(n_rel)}
    edges = [(i, (i + 5) % n, f"r{i % n_rel}") for i in range(n)]
    cfg = WiazkaConfig(op_dim=8, num_generators=4, gat_num_layers=1,
                       gat_heads=1, batch_size=16, epochs=1, device="cpu",
                       train_ratio=0.8, calib_ratio=calib_ratio)
    return WiazkaTrainer(node2emb, rel2idx, edges, cfg)


def test_calibration_slice_is_disjoint_from_the_scored_probe():
    trainer = _calib_trainer(calib_ratio=0.5)
    assert trainer.calib_edges and trainer.eval_edges
    assert set(trainer.calib_edges).isdisjoint(trainer.eval_edges)
    # nothing in the supervised set is a calibration edge either
    assert set(trainer.calib_edges).isdisjoint(trainer.edges)
    assert trainer._calibration_edges() == trainer.calib_edges


def test_without_calib_ratio_the_fits_keep_using_train_edges():
    trainer = _calib_trainer(calib_ratio=0.0)
    assert trainer.calib_edges == []
    assert trainer._calibration_edges() is trainer.edges


def test_dial_is_fitted_on_calibration_edges(tmp_path):
    """A dial fitted on calibration edges must not consult the probe."""
    trainer = _calib_trainer(calib_ratio=0.5)
    seen = []

    original = trainer._targets_by_src_rel

    class _Spy(dict):
        def get(self, key, default=()):
            seen.append(key)
            return dict.get(self, key, default)

    trainer._targets_by_src_rel = _Spy(original)
    fit = trainer.calibrate_transport_dial(n_per_rel=4)
    assert fit, "every relation with calibration edges must get a lambda"
    # the probe pool must never be asked about: only calibration + train edges
    probe_keys = {(s, r) for s, _d, r in trainer.eval_edges}
    assert probe_keys.isdisjoint(set(seen) & probe_keys)


def test_rare_relation_dial_falls_back_to_supervised_edges(capsys):
    """< min_calib_per_rel calibration edges must not decide a relation."""
    trainer = _calib_trainer(calib_ratio=0.5, n=60, n_rel=3)
    # make r2 almost absent from the calibration slice
    trainer.calib_edges = [e for e in trainer.calib_edges if e[2] != 2][:1]
    trainer.calibrate_transport_dial(n_per_rel=4, min_calib_per_rel=24)
    out = capsys.readouterr().out
    assert "below 24" in out


def test_selection_score_uses_the_dial_invariant_term():
    base = {"eval/mrr": 0.2, "eval/mrr_macro": 0.2, "eval/prod_mrr": 0.2,
            "eval/rel_acc_macro_ensemble": 0.4}
    without = WiazkaTrainer.selection_score(base)
    better = dict(base, **{"eval/prod_mrr_traffic_bestlam": 0.6})
    worse = dict(base, **{"eval/prod_mrr_traffic_bestlam": 0.05})
    assert WiazkaTrainer.selection_score(better) > without
    assert WiazkaTrainer.selection_score(worse) < without


def test_edge_probe_unique_by_deterministic():
    from relgat_projector.core.metrics.ranking import edge_probe
    edges = [(i, (i + 3) % 50, i % 2) for i in range(50)]
    a = edge_probe(edges, 20, seed=5, unique_by="src_rel")
    b = edge_probe(edges, 20, seed=5, unique_by="src_rel")
    assert a == b
    assert len({(s, r) for s, _d, r in a}) == len(a)


def test_profile_and_geometry_travel_with_the_checkpoint(tmp_path):
    node2emb = {i: torch.randn(16) for i in range(32)}
    rel2idx = {"r0": 0, "r1": 1}
    edges = [(i, (i + 3) % 32, f"r{i % 2}") for i in range(32)]
    cfg = WiazkaConfig(op_dim=0, op_dim_variance=0.9, num_generators=4,
                       gat_num_layers=1, gat_heads=1, batch_size=16,
                       epochs=1, device="cpu")
    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg)
    resolved = trainer.cfg.op_dim
    trainer.save(str(tmp_path / "ckpt"))
    import json
    config = json.load(open(tmp_path / "ckpt" / "config.json"))["config"]
    assert config["op_dim"] == resolved
    assert 0 < resolved < 16
    assert config["base_profile"]["dim"] == 16
    assert config["geometry"]["contrast_temperature"] > 0
    assert config["num_calibration_edges"] == 0


def test_calibrated_traffic_mrr_is_nan_without_a_calib_slice():
    trainer = _calib_trainer(calib_ratio=0.0)
    x = trainer._node_emb_dev[:8]
    rel = torch.tensor([0, 1] * 4)
    exported = trainer.model._export_linear(
        trainer.model.apply(trainer.model.base_to_op(x), rel))
    dst = torch.randint(0, 36, (8,))
    out = trainer._calibrated_traffic_mrr(x, exported, rel, dst, [], [])
    assert out != out  # NaN: the historical config has no honest slice


def test_selection_prefers_calibrated_traffic_over_a_dead_dial():
    """The epoch score must track the *shipped* dial, not the frozen one."""
    base = {"eval/mrr": 0.2, "eval/mrr_macro": 0.2, "eval/prod_mrr": 0.2,
            "eval/rel_acc_macro_ensemble": 0.4,
            "eval/prod_mrr_traffic_bestlam": 0.30}
    # the calibrated dial reaches further than any global lambda
    good = dict(base, **{"eval/prod_mrr_traffic_calib": 0.55})
    assert WiazkaTrainer.selection_score(good) > WiazkaTrainer.selection_score(base)
    # a NaN calib term (no slice) must not poison the score
    import math
    nan = dict(base, **{"eval/prod_mrr_traffic_calib": float("nan")})
    assert math.isfinite(WiazkaTrainer.selection_score(nan))
