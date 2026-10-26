# WIĄZKA — what the training quality numbers mean, and what was wrong

Auditing a training run is easy in one direction: the loss goes down.  This
document is about the other direction — the cases where the loss went down, the
metrics looked reasonable, and the model was not learning what the metric was
supposed to mean.  Each entry names the *measurement* that exposes the defect,
because that is the part worth keeping.

All numbers below are held-out (10 % of the edges, removed from the GAT
topology as well as from supervision), on the Słowosieć export: 243 468 nodes,
359 818 edges, 56 relations, `op_dim=256`, batch 2048, cosine LR.

## 1. Four protocols, not one

A single MRR number cannot tell a good model from a broken one here, because
the model can win at one level and lose at another:

| protocol | query | index | what it can hide |
|----------|-------|-------|------------------|
| `eval/mrr` | `T_r(refined_a)` | refined | a refinement that is better than the index it replaced |
| `eval/id_mrr` | `refined_a` | refined | — (this *is* the refinement's own value) |
| `eval/raw_mrr` | `base_a` | base | — (plain cosine; the thing to beat) |
| `eval/prod_mrr` | `(1-λ_r)·a + λ_r·export(T_r(chart(a)))` | base | what the vector database would actually return |

Every one of them masks two classes of columns: the other true targets of the
same `(source, relation)`, and the row of the query itself (§2).  The probe is
relation-stratified (`stratified_edge_probe`) and unique per `(source, relation)`.  This matters more than it looks: on a graph where
`synonimia` holds 178 k of 360 k edges, a uniform probe scores one relation.
The same model reads `raw_mrr` 0.3288 on the raw held-out set and 0.0914 on the
stratified probe — every comparison in this document uses the stratified one.

Reference points (40 epochs unless noted):

| model | `mrr` | `mrr_macro` | `prod_mrr` | `prod` lift over raw | relation naming (macro) |
|-------|-------|-------------|-----------|----------------------|-------------------------|
| raw cosine | 0.0914 | 0.0724 | 0.0914 | — | 0.018 (1/56) |
| before this pass (`V_dial`) | 0.2096 | 0.2170 | 0.1528 | +0.061 | 0.286 |
| after (`W1`) | 0.1967 | 0.2028 | **0.1722** | **+0.081** | **0.466** |

The graph-side MRR is slightly lower than the reference run and the production
number, the one the product uses, is clearly higher; §6 ablates the difference.

## 2. Defects found by measurement

**The connection was numerically dead.**  `θ(r, x) = θ_r + ε·P(relu(Wx))` had
`log_eps` initialised at −6, so ε ≈ 0.0025 ≈ 1 % of a rotation angle.  Adam
moves the log of a scalar by ≈ lr per step, so a whole run explores e^1.5 ≈ 4.5×
around its initialisation: a finished 35-epoch run ended at `log_eps = −5.43`.
Detection: log `eval/eps` every epoch.  Fix: initialise at −2 (ε ≈ 0.135).  The
run above ends at ε = 0.220 — the connection trains, and §6 shows it pays.

**`C_rs` never learned anything** (0.49 ± 0.02, i.e. the initial value).  The
witness sampler found 5 947 of the 203 208 commutation diamonds in the graph,
and an unsupervised holonomy-gap gradient fought the supervision on top of that.
Detection: print the matrix — a matrix whose entries are all 0.49 is not
learning.  Fix: exhaustive diamond compilation, balanced witnessed/unwitnessed
pairs, and `.detach()` on `C_rs` inside the unsupervised term so only the
supervision moves it (asserted by a test).

**The λ dial never moved.**  One scalar per relation, a near-flat loss, and a
shared learning rate left every relation at λ ≈ 0.81 after 35 epochs.  Detection:
the dial table.  Fix: a dedicated optimizer group (`dial_lr_scale`), plus
`calibrate_transport_dial()`, which fits λ directly against the metric the
product reports, on train edges.  It now spans 0.00–1.00 (mean 0.60) and, per
relation, does exactly what it should: `aspektowość czysta` (where the transport
hurts) goes to λ = 0, `kolokacyjność` (where it helps) to λ = 0.8.

**Two calibrations were hardcoded, and both were wrong in opposite directions.**

* The relation ensemble mixed head and transport with a fixed equal weight.  That
  was right for a head trained with plain cross-entropy (transport 0.344 macro,
  head 0.288, ensemble 0.384) and wrong once the head was class-balanced (head
  0.466, ensemble 0.445).
* The frequency correction always subtracted `log p_train(r)`, which is the
  Bayes rule only for unweighted training; a class-weighted head fits
  `p(r|x)·w(r)`, so it needs `+β·log count` — the opposite sign.  Applied to the
  balanced head it scored **0.058** against 0.465 uncorrected.

Both are one scalar each, now fitted on train edges
(`fit_relation_ensemble_weight`, `fit_relation_prior_strength`) and stored in the
checkpoint so a loaded model behaves like its report.  Verification on the real
graph: the train-fitted ensemble weight scores 0.4664 held-out where the best
possible held-out choice scores 0.4665 and the old fixed weight scored 0.4458 —
the fit is within 0.0001 of an oracle tuned on the test set, and the fitted prior
strength is 0.0 exactly, as the weighted-likelihood argument predicts.

**A stage gate was unreachable, and it cost a third of every run.**  F3 was
gated on `PCE(2)/PCE(1) ≤ 1.6`; two independent angular errors add up to a ratio
of ≈ 2.0 by construction, so the gate never opened (best measured 2.07) and every
run left F3 through the `max_steps` valve, starting the export stage (`L_pull`)
on the decaying tail of the LR schedule.  Detection: log whether a stage passed
its gate or timed out — the log line now says so.  Fix: gate F3 on what
composition is for (`eval/path_lift ≥ 0`); F4 now starts at epoch ~10 instead of
~20, and `prod_mrr` climbs 0.092 → 0.172 within the run.

**`L_pull` was not the objective it claimed to be.**  It scored a cosine margin
against 8 *random* negatives; with 243 k nodes those are almost all trivially
dissimilar, the term nearly vanished, and `eval/mrr` 0.19 sat next to
`eval/prod_mrr` 0.11.  Fix: full softmax of the exported query over the whole
base index, with false-negative masking, fed from the same input the deployed
rewriter uses.

**The report described a model that was not shipped.**  Training kept
`<save-dir>/best`; the report and the run checkpoint held the last epoch.  Now
both are scored with the training criterion (`shipped_checkpoint`,
`metrics_by_checkpoint`) and the better one is what the run saves.

**Batched `torch.linalg.solve` is broken on multi-threaded oneMKL**
(`Pivots given to lu_solve…`, `SLASWP` parameter error) at `op_dim=256` with more
than one relation in the batch: every CPU inference of a trained model crashed.
The operator detects the failure once and switches to `lstsq` (identical to
1e-6; rotations orthogonal to 3e-6).

**The dial was calibrated on a biased sample.**  It took the first ~96 edges of
each relation in file order, which is 96 adjacent source nodes of one frequency
band.  Fix: seeded uniform sample (`spread_sample`) — note that a fixed stride is
*not* a fix: a stride commensurate with the interleaving period aliases, which a
test now catches.

**The ranking protocol counted the query as a candidate answer.**  For a probe
edge `a -r-> b` the model ranked `b` against **every** node, including `a`
itself.  `a` is never the answer (0 self-loops in 359 818 Słowosieć edges) and a
retriever never returns the query document, but while it stayed in the index it was
the one candidate certain to be wrong *and*, of all of them, the one the rewritten
query resembles most: the served query is `(1-λ_r)·x_a + λ_r·export(...)`, so at
small `λ_r` it still points at `x_a`, whose cosine with itself is 1.  Two
consequences, one cosmetic and one not:

* every `mrr`-type number was understated by exactly the probes where the source
  sat above the gold (largest for the relations the raw embedding already solves --
  a synonym sits one position from its own row);
* the λ fit on train edges was paid to *move away from `x_a`* rather than to find
  the answer, i.e. it was biased towards rewriting more than the served metric
  wants.  `fuzzynimia`/`synonimia` are exactly where the fit over rewrote, and
  QR-Recall (which has always excluded the query, `exclude_self=True`) is where it
  showed up: `fuzzynimia` 0.316 → 0.299 while `prod_mrr` rose.

Detection: the losses already masked the subject (`L_con`, `L_pull` set
`mask[i, src_i] = True`) and QR-Recall already excluded it -- the metric and the
dial fit were the odd ones out, which is also why the two disagreed.  Fix:
`WiazkaTrainer._rank_mask` returns the two column classes (`other_targets`,
`never_valid`) and is the single source of truth for `evaluate()`,
`calibrate_transport_dial()` and `per_relation_report()`; `filtered_ranking` masks
them and never masks the gold even if a caller lists it.

Measured on the same weights (one checkpoint, one probe, two protocols):

| | `raw_mrr` | `mrr` | `mrr_macro` | `id_mrr` | `prod_mrr` | `hits@1` |
|---|---|---|---|---|---|---|
| query left in the index | 0.0914 | 0.2010 | 0.1911 | 0.0660 | 0.1802 | 0.0986 |
| query excluded | 0.1399 | **0.2269** | **0.2113** | 0.1061 | **0.2176** | 0.1357 |

The absolute numbers rise, the *lift* over the baseline shrinks (`mrr - raw`
0.110 -> 0.087): part of what had been read as the transport's advantage was the
model being paid to escape its own embedding.

And the part that was not cosmetic -- refitting the dial under the corrected
protocol (fit on train edges, scored on held-out edges, query excluded in both):

| relation | `lam` before | `lam` after | held-out production MRR before | after |
|----------|--------------|-------------|-------------------------------|-------|
| `synonimia` | 0.20 | 0.10 | 0.4387 | **0.6414** |
| `fuzzynimia` | 0.40 | 0.30 | 0.2845 | **0.3703** |
| `antonimia właściwa`, `derywacyjność` | 0.7 | 0.7 | unchanged | unchanged |

`synonimia` is 49.7 % of the edges, so this is the largest single quality effect
found in this pass -- and in the *stratified* probe it is worth only +0.004
`prod_mrr`, because the stratified average gives `synonimia` 1/56 of the vote.
That blind spot is why `evaluate()` now also reports
`eval/prod_mrr_traffic` / `eval/raw_mrr_traffic` / `eval/mrr_prod_lift_traffic`:
the same production query scored on a probe drawn as traffic is distributed,
where the lift over cosine goes **+0.007 -> +0.040** on the same weights.

The product-facing version of the same check is QR-Recall (held-out, 512 queries
per relation, other golds filtered against the graph -- which is why `before`
reads higher than the pre-fix reports):

| relation | `mrr` without rewriting | with the shipped `lam` | with `lam` refitted under the served protocol |
|----------|-------------------------|------------------------|-----------------------------------------------|
| `synonimia` | 0.686 | 0.659 | **0.693** |
| `fuzzynimia` | 0.316 | 0.301 | **0.313** |
| `antonimia właściwa` | 0.010 | 0.049 | 0.050 |
| `derywacyjność` | 0.040 | 0.077 | 0.078 |

Before the fix the rewriter made retrieval *worse* than sending the embedding
unchanged on the two relations the base space already encodes, and better on the
two it does not.  After it, no relation is degraded and none of the gain is lost.

**The dial scan argmax was a coin flip on ~100 edges.**  One relation, 96 probe
edges, 11 grid points: neighbouring points differ by a handful of probes, so the
fitted `λ_r` could be decided by a single edge.  Fix: the scan reads 256 edges per
relation and picks the argmax of a centred 3-point moving average
(`smooth_curve`) -- spikes disappear and, because a centred average preserves
monotone order, a curve that only decreases still peaks at `λ = 0`, so "do not
rewrite this relation" stays reachable.

**QR-Recall filtered against the sample, not the graph.**  The other correct
answers were taken from the pairs that were sampled, so a subject with 40 correct
synonyms that appears once in the probe got no filtering and scored its other 39
correct answers as misses -- the pessimisation was worst on precisely the
high-multiplicity relations.  `qr_recall(other_map=...)` now receives the graph's
answer table.

**`load_checkpoint` returned an uncalibrated head.**  The two calibration
scalars (`relation_prior_strength`, `relation_ensemble_weight`) are attributes, not
parameters, so they are not in the state dict -- and only `RelationFlow.load`
restored them.  Anyone reaching for the low-level loader got the constructor
default, i.e. the Bayes correction of an *unweighted* head, applied to a class-balanced
one: relation naming 0.05 macro instead of 0.47, measured by loading a finished
checkpoint both ways.  Fix: `wiazka_checkpoint.apply_calibrated_scalars`, called by
`load_checkpoint` itself (one definition of what the run fitted), with
`load_weights_from` copying the scalars too.

## 2a. The 2026-10-26 pass: measuring the space, not assuming it

Two more measurement defects were found the same way as the rest: the method
made statements about the base space it was given, and nothing checked them.

* **`op_dim` was a constant, not a measurement.**  The operator dimension 256
  was picked for this one embedder and raised `ValueError` on any base space
  with at most 256 dimensions — the method could not even start on such an
  embedder.  Now `--op-dim 0` (the default) derives it from the measured
  explained-variance curve of the base embeddings (`--op-dim-variance 0.95`,
  bounded by `--op-dim-cap`), and the Słowosieć resolves to 256 (EV@256 =
  0.9946, effective rank ≈ 100) while a 64-d embedder resolves to something
  that fits.
* **Cosine knobs sat on this embedder's cosine scale.**  `anchor_margin 0.05`,
  `knn_slack 0.02` and `contrast_temperature 0.07` are all statements about a
  cosine difference.  In a space whose random pairs sit at cos +0.13 ± 0.147
  (measured) a slack of 0.02 means something different than in an isotropic
  space.  `--geometry-auto` rescales all three by
  `std(this space) / REFERENCE_COS_STD`, reproducing the shipped defaults
  exactly on the reference scale and travelling as a ratio elsewhere.
* **Anisotropy was invisible.**  The base vectors are not unit-norm
  (mean ‖x‖ 0.82) and carry a shared direction (random-pair cosine +0.1323).
  `EmbedderProfile` now records both at startup, prints them, and stores them
  in the checkpoint and the report, so a run says which space its numbers
  were measured in.  `--base-center` moves the operator chart to the data
  centroid (the export adds the mean back, so the retriever index is
  untouched); measured, that removes the anisotropy (0.1323 → 0.0006) and is
  worth +0.004 identity MRR.
* **The served scalars were fitted on the edges they were scored against.**
  The dial, the relation prior and the ensemble weight were fitted on train
  edges — the edges the decoder memorised — and scored on the same edges'
  statistics.  `--calib-ratio 0.5` splits the held-out set: the scalars are
  fitted on one half, every reported metric on the other, and a relation with
  fewer than 24 calibration edges falls back to supervised edges for its
  dial (an honest fit needs data; a random fit on two edges is not honest).
* **The "best" checkpoint was chosen by a metric the product does not ship.**
  The selection score contained `prod_mrr` measured with the *gradient-trained*
  dial — a dial that never leaves `lam ≈ 0.8` — while the artifact ships the
  *calibrated* dial.  On F2 the traffic metric rose to 0.4696 around epoch 15
  and then decayed to 0.4467 (below the raw 0.4499) while the loss kept
  improving, so the criterion ranked a decaying run as its own best.  The fix
  measures what ships: `eval/prod_mrr_traffic_calib` fits a per-relation dial
  on the calibration slice (never the probe) and scores the disjoint traffic
  probe with it; `eval/prod_mrr_traffic_bestlam` (best *global* lambda) is the
  fallback when no slice exists.  A global lambda alone cannot substitute: on
  F2 none of them beat the raw query on traffic, while the calibrated
  per-relation dial scored 0.4902 against 0.4499 raw — the useful answer is
  "rewrite `antonimia` fully and `synonimia` not at all", and no constant can
  say that.  The selection score is now
  `0.30·mrr + 0.18·mrr_macro + 0.10·prod_mrr + 0.24·(traffic_calib | traffic_bestlam | prod_mrr)
  + 0.18·rel_acc_macro_ensemble`.

Dead ends measured on F2's weights, so the next run does not repeat them:

* a **learned per-relation translation** on top of the exported transport
  (56 × 1152 parameters, 12 epochs on 12 k train edges): held-out probe
  0.2307 → 0.2206, traffic 0.4902 → 0.4858.  The rotation family is not
  missing a shift.
* feeding the **refined** node vector into the production path instead of the
  unrefined chart: 0.2165 against 0.2307 served.  The message passing does
  not help the query; the served input was already the right one.
* opening the **export gate** past its trained value (tanh 0.487): 0.2307 →
  0.1875 fully open.  The gate sits where it should.

## 3. What is deliberately not a bug

* `L_cycle`, `L_shadow_cycle`, `L_isom` are **exactly 0** — the inverse is exact
  and the operator is conformal by construction.  A non-zero value would be the bug.
* `L_curv ≈ 0` at initialisation (rotation angles start small, so holonomy is
  genuinely small).
* `chart_mrr ≈ raw_mrr`: the frozen orthonormal PCA chart loses nothing; the gap
  between `id_mrr` (0.070) and `chart_mrr` (0.091) is the refinement trading
  index quality for transport quality, which is the trade `--knn-weight` argues
  against.
* Graph-candidate masking from held-out edges would *look* like a large accuracy
  gain and is leakage (the gold is in the mask by construction); it was measured
  and rejected.

## 4. Known limits (measured, not assumed)

* **The dial is finished as a knob.**  Fitting the scan on a finer grid is worse
  than the 0.1 grid (traffic-micro held-out production MRR 0.4848 against
  0.4870 -- the extra resolution is spent on noise), and choosing `lam_r` with
  hindsight on held-out edges gives 0.4898 against the fitted 0.4870.  The fit is
  within 0.003 of the best per-relation dial that exists, so nothing more is
  reachable there; quality has to come from the transport itself.

* Relations the base embeddings already encode are the ones the transport loses
  on: `synonimia` raw 0.590 against operator 0.349.  The dial recovers most of it
  in the production path (0.585) and `żeńskość` goes *above* raw (0.211 → 0.324),
  but the honest statement is: for those relations the model is a retrieval
  rewriter, not a better ranker.
* The export is a *linear* rewrite into the base space, so it cannot express a
  relation whose signal is not linearly recoverable there: `antonimia właściwa`
  reaches 0.143 in operator space and 0.009 after export.  The gated export
  residual is trained (gate 0.38, weights 75× its initialisation), so this is a
  capacity limit of the base space, not a dead parameter.
* Relation naming overfits: macro accuracy 0.88 on train pairs against 0.47 held
  out, and the held-out errors are not random.  They concentrate on *adjacent*
  relation types — `aspektowość czysta` ↔ `wtórna`, `ekspresywność|augmentatywność`
  ↔ `deminutywność`, `zawieranie pacjensa|obiektu` ↔ `zawieranie
  wytworu|rezultatu` — and on relations the probe can barely cover: the ten worst
  scored relations have between 1 and 25 held-out edges, the five best (the
  inter-paradigmatic `synonimia` families, 24-25 edges each) score 0.76-1.00.
  Part of the 0.47 ceiling is therefore the granularity of the relation set, not
  the classifier: an ontology-level decision (merge the aspectivity family, or
  score naming hierarchically) would move this number more than any trainer
  change.
* Composition is not in the selection score: `PCE(2)/PCE(1)` sits at ≈ 2.02 and
  single-hop rewriting is the shipped path.  A closed block-diagonal algebra
  would compose exactly, and is a research change, not a bug fix.

## 4a. F4: the first run trained and selected by the honest criterion

F4 (`wiazka-models/F4_20261026_024039`) is F2's configuration plus
`--calib-ratio 0.5` and the new dial-invariant selection score.  Scored
against F1 and F2 with one protocol (probe 1024, `calib-ratio 0.5` applied
to *all three* so every run is probed on the same 18 k held-out slice and the
dial/prior/alpha refit on the same calibration slice):

| run  | mrr    | macro  | prod   | raw    | rel    | prod_tr | raw_tr | score  |
|------|--------|--------|--------|--------|--------|---------|--------|--------|
| F1   | **0.2421** | **0.2182** | **0.2345** | 0.1588 | 0.4841 | 0.4677 | 0.4439 | 0.3349 |
| F2   | 0.2398 | 0.2153 | 0.2306 | 0.1588 | **0.4969** | **0.4687** | 0.4439 | 0.3326 |
| F4   | 0.2399 | 0.2153 | 0.2343 | 0.1588 | 0.4963 | 0.4677 | 0.4439 | **0.3361** |

* F4 wins the composite score and, unlike F1/F2, its **shipped calibration is
  already optimal** (shipped = refit to the last digit): the old artifacts
  left free calibration gain on the table (F2 prod 0.2306 → 0.2342 after
  refit), F4 ships nothing it could not ship.
* F1 is still the best on the *graph* protocols (mrr 0.2421, macro 0.2182,
  path lift 0.3218); the three runs are within noise on traffic
  (0.4677–0.4687 against raw 0.4439).  The honest reading is that the
  training recipe has converged and the remaining variance is selection,
  not architecture.
* The traffic *lift* here (+0.024) is smaller than the 0.4902 / 0.4499
  (+0.040) quoted for F2, but that is the probe, not the model:
  `calib-ratio 0.5` takes half the held-out set away, and the traffic lift is
  sensitive to which half is probed.  Every comparison in this table uses the
  same half.
* F4's dial does the right thing per relation: `synonimia` λ=0.00 (raw 0.751,
  do not touch), `antonimia właściwa` λ=1.00 (0.000 → 0.078),
  `nosiciel_stanu|cechy` λ=0.50 (0.086 → 0.272), `derywacyjność` λ=0.60
  (0.054 → 0.157).  QR-Recall follows: antonimia 0.009 → 0.059, derywacyjność
  0.031 → 0.084, synonimia 0.680 → 0.680 (not hurt), fuzzynimia 0.312 → 0.294.

## 5. Reproducing

```bash
# one run, all defaults that matter fixed
training_scripts/run-wiazka-trainer.sh          # edit the dataset paths first
# the ablations this document is based on
--contrast-temperature 0.05      # sharper contrastive ranking
--knn-weight 1.0                 # keep chart neighbours neighbours
--num-generators 32              # wider shared Lie basis (inference cost ~unchanged)
--no-connection                  # ablate the F4 connection
```

`report.json` carries the full per-epoch `history` (every loss term and metric),
`final_metrics` / `final_metrics_trained_dial`, `metrics_by_checkpoint`,
`shipped_checkpoint`, the fitted `dial_calibration`, the per-relation table and
the learned `commutation_C`.

### The protocol must be identical on both sides of every comparison

Two changes here altered the *scale* of every ranking number: masking the query
row (§2) lifted `raw_mrr` 0.0914 -> 0.1399, and before that the relation-stratified
probe replaced a uniform one (the same weights read `raw_mrr` 0.3288 on the raw
held-out set).  **A number from a report written before such a change is not a
comparable number**, and the difference is far larger than the effect any
hyperparameter has.  To compare two runs, load both checkpoints and score them
with one build of the code and one probe:

```bash
python3 score_checkpoints.py 2048 wiazka-models/RUN_A/best wiazka-models/RUN_B/best
```

which prints, per checkpoint, the metrics with the dial as shipped and with the
dial refitted under the current protocol, the per-relation table, and QR-Recall.
It is also how a checkpoint trained before a protocol fix is read honestly
without retraining it.  The equivalent for the fitted scalars is
`python3 score_checkpoints.py --only-calibration <ckpt>`, which refits
`lam_r`, the prior strength and the ensemble weight on train edges and rewrites
the checkpoint -- the calibration is a property of the report, not of training.

Reading a run health-wise, in one pass: `eval/eps` (connection alive),
`eval/rel_prior_strength` (`0` for a balanced head), `eval/rel_ensemble_alpha`,
`eval/mrr` against `eval/id_mrr` and `eval/raw_mrr` (transport vs refinement vs
cosine), `eval/prod_mrr_lift` and `eval/mrr_prod_lift_traffic` (the product), and
the terms that are zero by construction (`L_cycle`, `L_shadow_cycle`, `L_isom`)
so that a term that is zero *by dying* stands out.
