# Methodology

IS460 Group 03 · *Predicting Airline Passenger Satisfaction via Deep Tabular Ensembles: Evaluating
Synthetic Fidelity and Explainable Drivers*. Reference numbers follow `docs/references.md`.

## 1. Question

Kaggle Playground S6E10 [1] publishes 699,635 synthetic passengers produced by a generative model
trained on a real airline survey of 129,880 passengers [33]. Most work on this data optimises
leaderboard AUC on the synthetic copy. We ask a different question:

> **Can a synthetic copy of a customer survey be trusted for the decisions an airline makes?**

An airline uses a survey for three decisions, one per research question:

| RQ | Decision | Test |
|---|---|---|
| RQ1 | Which model do we deploy? | Score a ladder of models in four train→test settings; check whether the synthetic data picks the same winner and the same tricks as the real data. |
| RQ2 | How well will it work on real customers, and what did the generator change? | Distortion, detectability, label noise and utility of the synthetic data against the real survey. |
| RQ3 | Which services should we fix first, and for whom? | Service rankings per traveller segment from synthetic- and real-trained models; do they agree? |

## 2. Data

| | Synthetic (S) | Real (R) |
|---|---|---|
| Source | S6E10 `train.csv` [1] (`test.csv`, 299,844 unlabelled rows, used only for count encodings and a train/test sanity check) | arseniyshutko "Binary Aviation Satisfaction 129k" [33] |
| Rows | 699,635 | 129,880 |
| Satisfied | 44.4 % | 43.4 % |
| Missing arrival delay | 292 rows | 393 rows |

Both files share 21 features: age, flight distance, departure and arrival delay (minutes),
13 service ratings (0 = not applicable, 1–5), gender, customer type, type of travel and class.
The target is `satisfaction` (1 = satisfied). Categoricals are coded against fixed vocabularies
(`airsat.config.CATEGORY_LEVELS`) so a model trained on one dataset can score the other, and
unknown levels raise an error rather than being silently mapped.

**Memorisation check.** Only 17 of the 699,635 synthetic rows (0.002 %) copy a real row on all 21
features (4 of 299,844 in the test file), so the generator did not reproduce the survey row by row.
The real survey is, however, the generator's own training data. A synthetic-trained model scored
on real rows (S→R below) is therefore evaluated on the distribution the generator imitated, which
makes S→R an optimistic estimate of performance on *new* real passengers.

## 3. Evaluation protocol

**Folds.** One `StratifiedKFold(5, shuffle=True, random_state=42)` split per dataset, saved to
`artifacts/folds/{synthetic,real}.npy` and reused by every model and analysis (`airsat.folds`).

**Four settings from two cross-validation runs** (`airsat.runner`, `airsat.cv.run_cv`):

| Setting | Train | Scored on | How |
|---|---|---|---|
| S→S | synthetic folds | held-out synthetic fold | out-of-fold (OOF) predictions; the Kaggle setting |
| S→R | synthetic folds | all real rows | mean of the 5 fold models' predictions; "train synthetic, test real" (TSTR) [40] |
| R→R | real folds | held-out real fold | OOF; the reference an airline would use if it had the real data |
| R→S | real folds | all synthetic rows | mean of the 5 fold models' predictions |

Every run writes its OOF and applied predictions to a prediction store
(`artifacts/oof/<train>/<model>__<features>.npz` + `.json`), and every later analysis (stacking,
ranking agreement, label noise) reads from that store, so all results use the same predictions.

**Feature fitting inside folds.** Any statistic learnt from labels (target encodings) is fitted on
the training fold only; the training rows themselves receive out-of-fold encodings from five inner
folds (§4.2). Real labels are never used as features when real rows are the test set.

**Early stopping.** Boosted trees and the pytabkit networks stop on the outer validation fold, the
usual Kaggle practice. This makes S→S and R→R slightly optimistic (typically < 0.0002 AUC). The
transfer settings S→R and R→S are not affected, because the other dataset is never used for
stopping. The sklearn MLP stops on its own 10 % split of the training fold; logistic regression
and TabPFN do not stop early.

**Metric and uncertainty.** ROC AUC [2]. We report the pooled OOF AUC, per-fold AUCs, percentile
bootstrap 95 % intervals (1,000 row resamples) [41], and paired bootstrap intervals for differences
between two models scored on the same rows (`airsat.metrics`). With 140k rows per synthetic fold,
S→S differences below about 0.0005 are within noise.

## 4. RQ1: the model ladder

### 4.1 Rungs

| Key | Model | Inputs | Main settings |
|---|---|---|---|
| `logreg` | Logistic regression (course baseline) | median-imputed, standardised numerics; one-hot ratings and categoricals | C = 1 |
| `mlp` | scikit-learn MLP (course method) | as `logreg` | 128-64 ReLU, Adam, batch 2,048, ≤ 50 epochs, early stopping on an internal 10 % split |
| `lgbm` | LightGBM [21] | raw, native categoricals | 63 leaves, lr 0.05, ≤ 4,000 trees, 200-round early stopping |
| `xgb` | XGBoost [20], CUDA | raw, native categoricals | depth 6, lr 0.05, 256 histogram bins, ≤ 5,000 trees |
| `xgb_bins` | XGBoost | raw | 1,024 bins: keeps near-identical flight distances apart ("distance as route ID", [39]) |
| `xgb_ev` | XGBoost | raw + exact-value encodings (§4.2) | as `xgb_bins` |
| `catboost_cat` | CatBoost [22], GPU | every column as a categorical (exact values become levels, [39]) | depth 6, lr 0.08, ≤ 5,000 iterations |
| `realmlp` | RealMLP-TD [26] (pytabkit) | raw; arrival-delay NaN filled with departure delay | 16 epochs, batch 1,024, no label smoothing, best epoch by validation AUC |
| `tabm` | TabM-D [27] (pytabkit) | as `realmlp` | batch 1,024, ≤ 40 epochs, patience 8 |
| `tabpfn` | TabPFN v2 [28] | raw | in-context learning on a stratified 10,000-row context per fold, 4 estimators |

The last three rungs are the "deep tabular" models of the proposal; the course methods (logistic
regression, MLP, trees, ensembles) are part of the same ladder. Hyper-parameters are library
defaults or the settings reported in the S6E10 forum [39]; they are not tuned per setting, so no
setting gets an unfair tuning advantage.

### 4.2 Exact-value encoding

The forum's biggest lever on S6E10 [39] treats the exact values of flight distance, age and the
two delays as identifiers. For each such column, `airsat.encoders.ExactValueEncoder` adds
`te_<col>`, the share of satisfied passengers with that exact value, smoothed towards the prior
with 20 pseudo-counts [37], and `cnt_<col>`, the value's frequency (training fold plus the
unlabelled test file). Training rows get out-of-fold `te_` values from five inner folds; validation
and other-dataset rows use statistics from the whole training fold. If these columns encode
generator artefacts rather than passenger behaviour, `xgb_ev` should gain on S→S but not on S→R.

### 4.3 Ensembles

`airsat.stacking` combines the stored predictions of all rungs in each training direction with
(i) a rank mean, (ii) L2 logistic regression on logits [29] with C chosen by inner 3-fold CV, and
(iii) greedy ensemble selection with replacement (hill climbing) [30]. In-dataset scores are
nested: the combiner for outer fold *k* is fitted only on the other folds' OOF predictions. The
transfer score fits the combiner on all OOF rows of one dataset and applies it to the fold-averaged
predictions on the other. Comparing the weights learnt on S and on R shows whether the synthetic
data would pick the same ensemble.

### 4.4 Model-selection fidelity

Kendall τ [42] and Spearman ρ between the model rankings of every pair of settings
(`airsat.fidelity.ranking_agreement`), plus whether the top model is the same. A synthetic copy is
trustworthy for model selection if the S→S ranking matches the R→R ranking.

## 5. RQ2: fidelity of the synthetic data

All functions are in `airsat.fidelity` and `airsat.fidelity_utility`; the entry point is
`experiments/rq2_fidelity.py`.

1. **Marginal distortion.** Two-sample Kolmogorov–Smirnov statistic for numerics, total variation
   distance for ratings and categoricals, with where the gap is largest; delay quantiles and
   delay-band shares; the largest differences between the two Spearman correlation matrices.
2. **Detectability (adversarial validation)** [43]. LightGBM separates 100,000 synthetic from
   100,000 real rows; 3-fold OOF AUC (0.5 = indistinguishable), gain importance, and single-feature
   AUCs. The synthetic train vs test files give the ≈ 0.5 baseline.
3. **Target dependence.** Univariate AUC of an out-of-fold exact-value target encoding of each
   feature, on equal-size samples of S and R (larger samples would favour the bigger dataset).
4. **Label noise** (in the spirit of confident learning [44]). Share of rows where a real-trained
   LightGBM is confident (p > t or p < 1 − t, t ∈ {0.9, 0.95, 0.99}) and disagrees with the label,
   on synthetic rows vs on held-out real rows. The ratio estimates how many labels the generator
   flipped beyond the survey's own noise.
5. **Utility.** On real fold 0 as a fixed test set: a real-data learning curve (1k → 104k training
   rows, 3 seeds up to 20k) against TSTR, a LightGBM trained on all 699,635 synthetic rows. The
   curve gives the number of real passengers the synthetic set is worth, with an interval from the
   TSTR bootstrap bounds.
6. **Model-selection fidelity** (§4.4).

## 6. RQ3: drivers of satisfaction

All functions are in `airsat.drivers`; the entry point is `experiments/rq3_drivers.py`. One
LightGBM per dataset is fitted on folds 1–4 (early stopping on a 10 % split of those rows), and
every analysis uses the untouched fold 0.

1. **Grouped permutation importance.** AUC drop when all ratings of a service group are permuted
   together [41]: Digital (Wi-Fi, online booking, online boarding), Comfort (seat, leg room,
   cleanliness, entertainment, food and drink), Ground (check-in, baggage, gate), Crew (on-board
   service), Schedule (time convenience), Delays, Profile. Compared with the sum of single-feature
   drops, this shows how correlated ratings hide each other.
2. **Segment TreeSHAP** [38], [45]. Mean |SHAP| of the 13 services in six segments (cabin class ×
   trip purpose), ≤ 20,000 rows per segment, with the share of 200 row-bootstrap resamples in which
   each service ranks first. Three views: synthetic model on synthetic passengers, real model on
   real passengers, synthetic model on real passengers.
3. **Agreement.** Per segment: Spearman ρ of the two service rankings, top-3 overlap, and whether
   the "fix-first" service (rank 1) is the same.
4. **What-if.** Mean change in predicted P(satisfied) when one service is raised by one point
   (capped at 5) for passengers who rated it 1–3, per segment, for both models on both
   populations.

These quantities describe how a fitted model's predictions respond to the ratings. They are
associational: a survey cannot tell us what would happen if the airline actually improved a service.

## 7. Compute

Development runs use samples (`--sample 20000`) on a laptop (RTX 3080 10 GB, 17 GB RAM). Full-size
runs are SLURM jobs on the SCIS Origami GPU cluster (RTX 3090 24 GB or A40/L40S 48 GB, 8 CPU
cores, 48 GB RAM), driven by `cluster/remote.py` (see `cluster/README.md`). Seeds are fixed (42)
everywhere; GPU training of XGBoost, CatBoost and the networks is not bit-for-bit deterministic, so
reruns can differ in the fifth decimal.

## 8. Threats to validity

- **S→R is optimistic** (§2): the real rows trained the generator.
- **One real survey.** R→R is itself an estimate from one dataset of unknown provenance; it is the
  best available reference, not ground truth about all airline passengers.
- **No tuning per setting.** A model tuned on S could rank differently; we hold settings fixed to
  isolate the effect of the data.
- **Early stopping on the validation fold** (§3) slightly favours the in-dataset settings.
- **Associational drivers** (§6): rankings are model explanations, not causal effects.
