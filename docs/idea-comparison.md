# Which project idea scores better? — S6E10 Airline Satisfaction vs Xiangqi-R1

*Prepared 2026-10-07 for the IS460 group. Scores below are our own **estimates** against the
Prof's rubric (Week-3 proposal + Week-7 final-report criteria, each item scored 1–10, equal
weights by default). They are judgement calls, not facts — the evidence behind each is listed.*

## TL;DR

**Recommend the Kaggle S6E10 airline-satisfaction project**, with three deliberate
"difficulty boosters" (below). It wins on the criteria that depend on *actually producing
results* — Dataset, Results, Figures/Tables, Milestones — which make up most of the rubric,
and its one weak criterion (Difficulty) can be argued up. Xiangqi-R1 has the higher ceiling on
Difficulty and Literature Review, but its results depend on artefacts that are **not released**
and on RL-training an LLM on a shared cluster in ~5 weeks.

## Evidence gathered today

| Check | S6E10 Airline | Xiangqi-R1 |
|---|---|---|
| Data available? | ✅ Downloaded: 699,635 train / 299,844 test rows + 129,880-row original survey ([competition](https://www.kaggle.com/competitions/playground-series-s6e10/data), [original](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)) | ⚠️ Official repo releases code + one `test_data.json` (10.6 MB) only; the 5M board-move training set is not in the repo ([GitHub API listing of `data/`](https://github.com/MINE-USTC/Xiangqi-R1/tree/main/data)) |
| Model weights? | n/a (we train our own, minutes per model) | ⚠️ No official weights on the repo or Hugging Face; only unofficial community re-implementations (e.g. `hoduyquocbao/xiangqi-r1-0.5b`) |
| Objective external benchmark? | ✅ Live Kaggle leaderboard (ROC AUC), private LB revealed after 31 Oct 2026 | ❌ None — we would define our own metric (faithfulness) |
| Preliminary results by proposal? | ✅ Already have EDA figures + 5-fold CV baselines (see `reports/`) | ⚠️ Only a small "illegal-move rate" probe is feasible quickly |
| Compute fit | ✅ Runs on one RTX 3080 laptop/desktop GPU; SCIS cluster optional | ⚠️ GRPO on a 7B LLM needs multi-GPU; SCIS cluster notebooks cap at 6 h, batch jobs at 24 h and queue when busy (SCIS GPU Cluster slides, AY25-26 T2) |
| Timeline fit | ✅ Competition closes 31 Oct 2026 — results final before the Week-11 report | ⚠️ RL instability is the plan's own stated top risk |

## Rubric scoring (estimated, 1–10)

| Rubric item | S6E10 | Xiangqi | Why |
|---|:-:|:-:|---|
| Title | 9 | 9 | Both can state subject + method explicitly |
| Introduction / problem statement | 9 | 8 | S6E10 is crisp and measurable; "faithfulness of LLM explanations" needs more setup |
| Motivation | 8 | 8 | Airline satisfaction has strong industry stats; Xiangqi leans on LLM-hallucination evidence |
| Literature review | 8 | **9** | Xiangqi has a richer story (AlphaZero → LLM reasoning → Xiangqi-R1) |
| Datasets | **10** | 6 | Public, downloaded, documented vs. training data not released |
| Tools & resources | 9 | 8 | Both fine; Xiangqi depends on cluster availability |
| Results & discussion | **10** | 5 | Real CV + leaderboard numbers vs. risky RL outcome |
| Milestones | 9 | 7 | Hard external deadline vs. GPU-queue-dependent plan |
| References | 9 | 9 | Both can be IEEE-complete |
| Figures & tables | **10** | 7 | Tabular ML produces many clean, captioned figures |
| Difficulty | 7 | **10** | S6E10's weak spot — mitigated below |
| **Total (/110)** | **≈98** | **≈86** | |

Final-report-only items: Methodology (S6E10 ≈9, Xiangqi ≈8 if RL works, lower if it falls back),
Conclusion & future work (both ≈8–9).

## How we lift S6E10's "Difficulty" score

1. **A real research question, not only a leaderboard:** *Does a model trained on synthetic
   data still capture real passengers' satisfaction drivers?* We measured a large synthetic-vs-real
   shift already (only 9.2% of synthetic passengers are delayed vs 43.5% in the real survey) and
   test train-on-synthetic → evaluate-on-real transfer.
2. **Grandmaster-level modelling:** GBDT trio (LightGBM/XGBoost/CatBoost) + tabular deep
   learning (RealMLP/TabM) + tabular foundation model (TabPFN) + hill-climbing / stacking
   ensembles over ~1M rows on GPU — the recipe used by recent Playground winners.
3. **Explainability → decisions:** SHAP attributions converted into a ranked "fix this service
   first" list per passenger segment (stretch goal: an LLM that writes the brief for airline
   managers, which ties back to the course's DM + ML + DL + LLM theme).

## Risks to raise with Prof. Wang

- Both ideas are **off the official topic list**. Confirm off-list topics are allowed.
- Kaggle Playground data is synthetic; frame results as a study of methods + synthetic-data
  fidelity, and always report results on the *real* survey data too.
