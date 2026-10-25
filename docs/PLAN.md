# Project Plan: Synthetic Fidelity and Explainable Drivers of Airline Passenger Satisfaction

IS460 Machine Learning & Applications · Group 03
(Chai Yichen · Chetan s/o Haresh Kishnani · Lee Jia Le · Joash Lau Rong Wei · Cody Loo Yee · Tristan Ang Boon Yew)

This file is the single source of truth for continuing the project from any session or by any
group member. Update the **Status** column and the **Decision log** whenever work lands.

---

## 1. The question we are answering (not "the best score")

Kaggle Playground S6E10 gives 699,635 *synthetic* passengers generated from a real survey of
129,880 passengers. Everyone else optimises leaderboard AUC on the synthetic copy. We ask the
question an airline would actually care about:

> **Can a synthetic copy of a customer survey be trusted for the decisions an airline makes?**
> (1) *Which model should we deploy?* (2) *How well will it work on real customers?*
> (3) *Which services should we fix first, for whom?*

This keeps the three research questions from the proposal deck, sharpened:

| RQ | Proposal wording | What we actually test |
|---|---|---|
| **RQ1 Accuracy / transfer** | How much do exact-value encoding, tabular neural nets and foundation models add over boosted trees? | Every rung of the model ladder is scored in **three settings**: synthetic→synthetic (Kaggle setting), real→real, synthetic→real. Which "winning" techniques help real passengers, and which only exploit generator artefacts? |
| **RQ2 Fidelity** | Does a model trained on synthetic passengers recognise real ones? What did the generator distort? | Distribution distortion per feature, detectability (adversarial validation), label-noise estimate, synthetic-data utility (TSTR vs TRTR), "how many real rows are 700k synthetic rows worth", and whether model *rankings* agree across settings. |
| **RQ3 Insight / drivers** | Which services move satisfaction most per segment, once overlapping ratings are accounted for? | Grouped permutation importance + TreeSHAP per segment (cabin class × trip purpose), computed on synthetic- and real-trained models; agreement of the "fix-first" list; what-if (+1 rating) effects, stated as associational. |

Course fit: Prof. Wang's project types #1 (repeat & validate published work) and #3 (combine works).
Course methods (logistic regression, MLP, trees, ensembles) are first-class rungs of the ladder.

## 2. Evaluation protocol (applies to everything)

- **Data**: synthetic = `data/raw/train.csv` (S6E10, n = 699,635); real = `data/original_official/data.csv`
  (arseniyshutko "Binary Aviation Satisfaction 129k", n = 129,880; identical to teejmahal20 minus `Inflight service`).
- **Folds**: `StratifiedKFold(5, shuffle=True, random_state=42)` on each dataset, saved once to
  `artifacts/folds/{synthetic,real}.npy` and reused by every model (`airsat.folds.get_folds`).
- **Settings** (one CV run yields all of them):
  - `syn_cv`: out-of-fold AUC on synthetic (S→S).
  - `syn_to_real`: the 5 synthetic fold-models predict all real rows, averaged (TSTR).
  - `real_cv`: out-of-fold AUC on real (R→R, TRTR).
  - `real_to_syn`: the 5 real fold-models predict all synthetic rows (TRTS).
- **Leakage rules**: target/count encodings are fitted inside each training fold (nested inner folds for
  the training rows). Nothing computed from real labels may be used as a feature when real data is the test set.
- **Metric**: ROC AUC; report fold mean ± std and a bootstrap 95% CI. Differences between models use a
  paired bootstrap on the same rows. Differences below ~0.0005 on synthetic are treated as noise.
- **Prediction store**: every run writes `artifacts/oof/<setting>/<model>__<featureset>.npz` + `.json`
  metadata (`airsat.oof`). Stacking and analysis read only from the store.

## 3. Repository layout

```
src/airsat/            library (importable, unit-tested)
  config.py            paths, seeds, column groups, service groups, segments
  data.py              load_synthetic(), load_real(), to_model_frame()
  folds.py             fixed stratified folds, saved/loaded
  encoders.py          fold-safe exact-value target + count encoding
  cv.py                run_cv(): one model spec → OOF + applied predictions for all settings
  oof.py               prediction store (save/load/list)
  metrics.py           AUC, fold stats, bootstrap CI, paired bootstrap delta
  models/              model specs with a common fit/predict interface
  features.py          feature pipelines (exact-value encodings) used by ladder rungs
  locking.py           one-heavy-job-at-a-time lock for the 17 GB laptop
  ladder.py            RQ1 ladder table + paired deltas per technique; ladder_plots.py figures
  stacking.py          rank mean, logistic stacking, hill climbing, nested-CV evaluation (WP4)
  fidelity.py          distortion, detectability, dependence, label noise, ranking   (WP2)
  fidelity_utility.py  real learning curve vs TSTR; fidelity_plots.py figures
  drivers.py           grouped permutation importance, segment SHAP, what-if, agreement (WP3)
  drivers_plots.py     RQ3 figures; plotstyle.py figure style (slide-legible)
experiments/           runnable entry points: rq1_ladder, rq1_stacking, rq1_report, rq2_fidelity,
                       rq3_drivers, eda, prelim/ (proposal-stage scripts)
cluster/               remote.py + README: run full-size jobs on the SCIS GPU cluster (SLURM)
tests/                 pytest unit tests for src/airsat
docs/                  PLAN (this), METHODOLOGY, RESULTS, references, research notes, figures/
artifacts/, data/, reports/   generated or downloaded: git-ignored
```

## 4. Work packages

| WP | Owner | Scope | Deliverables | Status |
|---|---|---|---|---|
| WP0 | Joash + Claude | Infrastructure: package, data loaders, folds, encoders, CV runner, prediction store, metrics, tests, cluster tooling | `src/airsat/*`, `tests/*`, `pyproject.toml`, `cluster/*` | ☑ |
| WP1 | Stream A + B | RQ1 model ladder in 3 settings: LogReg, MLP (sklearn), LightGBM, XGBoost (GPU), CatBoost all-categorical, +bins, +exact-value encoding, RealMLP, TabM, TabPFN (subsampled context, optional) | `src/airsat/models/*`, `experiments/rq1_ladder.py`, `experiments/rq1_report.py` | ☑ |
| WP2 | All six | RQ2 fidelity analyses | `src/airsat/fidelity*.py`, `experiments/rq2_fidelity.py`, tables + figures | ☑ |
| WP3 | All six | RQ3 drivers by segment, synthetic vs real | `src/airsat/drivers*.py`, `experiments/rq3_drivers.py`, tables + figures | ☑ |
| WP4 | Stream B | Stacking vs hill climbing vs averaging, nested CV, in each setting | `src/airsat/stacking.py`, `experiments/rq1_stacking.py` | ☑ |
| WP5 | All six | Write-up: METHODOLOGY.md, RESULTS.md, README, final figures | `docs/METHODOLOGY.md`, `docs/RESULTS.md`, `docs/figures/` | ☑ |

### WP1 model ladder (each row × 3 settings)

| Rung | Model | Feature set | Why |
|---|---|---|---|
| 1 | Logistic regression | one-hot ratings + scaled numerics | course baseline; published 0.94859 on S→S |
| 2 | MLP (scikit-learn) | same as 1 | course method |
| 3 | LightGBM | raw | standard GBDT |
| 4 | XGBoost (CUDA) | raw, 1,024 bins | "Flight Distance as route ID" artefact |
| 5 | XGBoost | raw + exact-value target/count encoding | the forum's biggest lever (+0.0014 on S→S) |
| 6 | CatBoost | all columns categorical | forum: +0.0022 over native handling |
| 7 | RealMLP (pytabkit) | raw | strong tabular NN |
| 8 | TabM (pytabkit) | raw | parameter-efficient ensemble NN |
| 9 | TabPFN (optional) | raw, ≤ 50k context | tabular foundation model |
| 10 | Stack (WP4) | OOF logits of 1–9 | ensembling |

### WP2 fidelity analyses

1. Per-feature distortion: KS (numeric), total variation distance (categorical/ratings), synthetic vs real.
2. Detectability: adversarial AUC overall, per-feature single-feature AUC, top features of the adversarial model.
3. Dependence: target mutual information per feature, synthetic vs real; correlation-matrix difference.
4. Label noise: share of synthetic rows where a real-trained model is confident (p > 0.95 or < 0.05) and disagrees with the label, vs the same on held-out real rows.
5. Utility: TSTR vs TRTR AUC; learning curve on real (n = 1k…104k) to find how many real rows equal the full synthetic set.
6. Model-selection fidelity: Kendall τ between the WP1 rankings in S→S, R→R, S→R.

### WP3 driver analyses

1. Service groups (grouped permutation importance, AUC drop): Digital (Wi-Fi, online booking, online boarding), Comfort (seat, leg room, cleanliness, entertainment, food & drink), Ground (check-in, baggage, gate location), Crew (on-board service), Schedule (time convenience), Delays, Profile.
2. Single vs grouped importance: shows how overlapping ratings hide each other.
3. Per-segment (Class × Type of Travel, 6 segments) TreeSHAP ranking of the 13 services, synthetic vs real.
4. Agreement: Spearman ρ and top-3 overlap of rankings; same "fix-first" service per segment?
5. What-if: mean change in predicted P(satisfied) when one service is raised by +1 (max 5) for passengers rating it ≤ 3; synthetic vs real models. Associational, not causal.

## 5. Machine constraints

Laptop: RTX 3080 10 GB; 12 CPU cores; **17 GB RAM, often < 2 GB free**. Develop on samples
(`--sample 20000`), one full-size job at a time (`airsat.locking`).
Full-size runs go to the **SCIS Origami GPU cluster** (account `is460`, QoS `is460qos`: one running
and two queued jobs per user, 1-day limit; RTX 3090 24 GB, A40/L40S 48 GB). See `cluster/README.md`.
Any free GPU is fine for every rung; TabPFN with a 10k context fits in 24 GB.

## 6. How to run

```
pip install -e ".[dev]"            # library + pytest
pip install -e ".[deep]"           # RealMLP / TabM / TabPFN (optional)
kaggle competitions download -c playground-series-s6e10 -p data/raw          # then unzip
kaggle datasets download -d arseniyshutko/binary-aviation-satisfaction-129k -p data/original_official --unzip
pytest --cov=airsat
# local dev runs (outputs under .../dev/)
python experiments/rq1_ladder.py --models all --sample 20000
python experiments/rq1_report.py --dev 20000
python experiments/rq2_fidelity.py --sample 20000
python experiments/rq3_drivers.py --sample 20000
# full runs on the cluster (see cluster/README.md for setup)
python cluster/remote.py push
python cluster/remote.py submit --name full-a --time 23:00:00 -- "python experiments/rq1_ladder.py --models logreg mlp lgbm xgb xgb_bins xgb_ev catboost_cat; python experiments/rq3_drivers.py; python experiments/rq2_fidelity.py --skip ranking"
python cluster/remote.py submit --name full-b --time 23:00:00 -- "python experiments/rq1_ladder.py --models realmlp tabm tabpfn; python experiments/rq1_stacking.py; python experiments/rq1_report.py; python experiments/rq2_fidelity.py --steps ranking"
python cluster/remote.py fetch
```

## 7. Commit policy

Commit only `src/`, `experiments/`, `tests/`, `docs/`, `cluster/`, `pyproject.toml`, `README.md`, `.gitignore`.
Never commit `data/`, `artifacts/`, `reports/`, `slides/`, course PDFs/spreadsheets, or credentials
(`~/.kaggle/access_token` stays outside the repo). Conventional commit messages (`feat:`, `docs:`, `test:`).

## 8. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-07 | Topic: S6E10 airline satisfaction | Data on hand, measurable results; see `docs/idea-comparison.md` |
| 2026-10-07 | Real data used as features/priors, never as extra training rows | Appending real rows hurt S→S AUC by 0.00022 (our test) |
| 2026-10-25 | Research framing: decision fidelity of synthetic data, not leaderboard score | Group decision; distinctive vs other projects and vs prior studies |
| 2026-10-25 | Every model scored in S→S, R→R, S→R | Separates real skill from generator artefacts |
| 2026-10-25 | Kaggle competition is a side track, not part of the course deliverables | Group decision |
| 2026-10-25 | Also score R→S (real-trained model on synthetic rows) | Completes the 2×2 design; shows how much of the synthetic signal real behaviour explains |
| 2026-10-25 | Full-size runs on the SCIS GPU cluster, dev runs on 20k samples locally | Laptop RAM (< 2 GB free) cannot hold the full ladder; cluster gives 24–48 GB GPUs |
| 2026-10-25 | Early stopping on the outer validation fold, documented as a small optimistic bias for S→S/R→R | Standard practice; transfer settings unaffected (METHODOLOGY §3) |
| 2026-10-25 | Treat S→R as optimistic: the real survey trained the generator | Only 17/699,635 exact copies, but same distribution (METHODOLOGY §2) |
| 2026-10-25 | TabPFN v2 weights with a 10k stratified context | v2.5+ weights are licence-gated; 10k context fits a 24 GB GPU |

## 9. Headline results (full runs, 25 Oct 2026) — details in `docs/RESULTS.md`

- Best model on real passengers: TabM (R→R 0.9954, S→R 0.9941); best on synthetic CV: CatBoost
  all-categorical (S→S 0.9603). Synthetic CV ranks models poorly vs real (Kendall τ 0.42).
- The S6E10 forum tricks (fine bins, exact-value encoding, all-categorical CatBoost) gain on S→S but
  not on real passengers; RealMLP/TabM do the opposite.
- 699,635 synthetic rows ≈ 21,700 real rows; ~4 % of synthetic labels are confidently wrong (25× real).
- Delays collapsed (91 % zero vs 56 %); adversarial AUC 0.832.
- Fix-first by SHAP: online boarding (synthetic) vs Wi-Fi (real) in all 6 segments; what-if agrees 6/6.

### Next steps for the group
- Final report and slides from `docs/RESULTS.md` and `docs/figures/` (cite [40]–[45] in `docs/references.md`).
- Optional robustness: repeat the ladder with 3 seeds; TabPFN with a larger context on a 48 GB GPU.

## 10. Prior results (from the proposal, 7 Oct 2026)

| Experiment | Result |
|---|---|
| Logistic regression, one-hot ratings (S→S) | 0.94859 |
| LightGBM raw / XGBoost raw (S→S) | 0.95887 / 0.95872 |
| LightGBM + real rows appended (S→S) | 0.95866 |
| XGBoost 1,024 bins / + exact-value encoding (S→S) | 0.95920 / 0.96009 |
| Adversarial AUC: synthetic train vs test / synthetic vs real | 0.502 / 0.831 |
| Train synthetic → test real / train real → test synthetic | 0.9929 / 0.9546 |
