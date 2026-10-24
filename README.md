# Cleared for Satisfaction

**Predicting airline passenger satisfaction via deep tabular ensembles: evaluating synthetic fidelity and explainable drivers.**
IS460 Machine Learning & Applications, SMU · Group 03:
Chai Yichen · Chetan s/o Haresh Kishnani · Lee Jia Le · Joash Lau Rong Wei · Cody Loo Yee · Tristan Ang Boon Yew

## The question

Kaggle Playground S6E10 releases 699,635 *synthetic* passengers generated from a real survey of 129,880
passengers. Instead of chasing leaderboard AUC on the synthetic copy, we ask whether that copy can be trusted
for the decisions an airline actually makes:

| | Question | How we test it |
|---|---|---|
| **RQ1** | Which modelling techniques help *real* passengers, and which only exploit the synthetic generator? | A ladder from logistic regression and MLP to boosted trees, tabular neural nets and ensembles, each scored synthetic→synthetic, real→real, synthetic→real and real→synthetic |
| **RQ2** | What did the generator distort, and how much is the synthetic data worth? | Per-feature distortion, adversarial detectability, label noise, utility learning curve, model-ranking agreement |
| **RQ3** | Which services drive satisfaction per traveller segment, and does the synthetic copy give the same "fix-first" answer? | Grouped permutation importance, segment-level TreeSHAP, ranking agreement, what-if analysis |

Full plan and status: [`docs/PLAN.md`](docs/PLAN.md) · methods: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) ·
findings: [`docs/RESULTS.md`](docs/RESULTS.md) · references: [`docs/references.md`](docs/references.md).

## Setup

Python 3.11. An NVIDIA GPU is used for XGBoost, CatBoost and the neural nets (CPU works, slower).

```bash
pip install -e ".[dev]"     # library + tests
pip install -e ".[deep]"    # optional: RealMLP / TabM (pytabkit), TabPFN
```

Data (needs a Kaggle API token and joining the competition):

```bash
kaggle competitions download -c playground-series-s6e10 -p data/raw && unzip data/raw/*.zip -d data/raw
kaggle datasets download -d arseniyshutko/binary-aviation-satisfaction-129k -p data/original_official --unzip
```

## Reproduce

```bash
pytest --cov=airsat                                    # unit tests
python experiments/rq1_ladder.py --models all          # RQ1: every model in every setting
python experiments/rq1_stacking.py                     # RQ1: ensembles (nested CV)
python experiments/rq1_report.py                       # RQ1: table + figure
python experiments/rq2_fidelity.py                     # RQ2
python experiments/rq3_drivers.py                      # RQ3
python experiments/eda.py                              # exploratory figures
```

Add `--sample 20000` to `rq1_ladder`, `rq2_fidelity` or `rq3_drivers` (and `--dev 20000` to `rq1_report`)
for a fast development run; outputs go to `.../dev/` folders. Full-size experiments take a lock
(`artifacts/locks/heavy.lock`) so only one runs at a time on a memory-constrained machine. We ran the
full-size experiments on the SCIS GPU cluster with `cluster/remote.py` (see [`cluster/README.md`](cluster/README.md)).

## Layout

```
src/airsat/     library: data, folds, leak-safe encoders, CV runner, prediction store, models,
                stacking, fidelity, drivers, metrics
experiments/    one entry point per research question (+ prelim/ from the proposal)
tests/          pytest unit tests (toy data; no downloads needed)
cluster/        submit full-size runs to the SCIS SLURM GPU cluster and fetch the results
docs/           plan, methodology, results, references, research notes
```

Generated folders (`data/`, `artifacts/`, `reports/`) are git-ignored.

## Data licences

Synthetic data: Kaggle Playground S6E10 (CC BY 4.0). Real survey: Kaggle dataset
`arseniyshutko/binary-aviation-satisfaction-129k` (CC0). Raw data is not redistributed in this repository.
