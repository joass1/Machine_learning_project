# Kaggle Intel: PS S6E10 (Predicting Airline Satisfaction) and what wins Playground binary-AUC competitions

Compiled 2026-10-07 (competition day 7 of 31) for the SMU IS460 project.

**How to read this file**
- Every factual claim has a source URL beside it. "Unverified" means I could not confirm it from a primary source.
- Kaggle pages render with JavaScript, so most Kaggle pages (overview, rules, leaderboards, discussions, writeups) were read through the `r.jina.ai` reader proxy on 2026-10-07. The URLs below are the original Kaggle URLs.
- **Primary** sources are the competition pages, discussion posts and solution writeups themselves. **Secondary** sources are GitHub repos by other competitors and the BlamerX writeup digests (https://github.com/BlamerX/Kaggle-Competition-Season-6/tree/main/writeup_digests). The BlamerX digests are AI-written summaries. Their own audit log records fabricated fields that had to be corrected (https://github.com/BlamerX/Kaggle-Competition-Season-6/blob/main/writeup_digests/_AUDIT_LOG.md), so I checked the key first-place claims for S6E9, S6E8, S6E5 and S6E3 against the original writeups.
- S6E10 is still running. Leaderboard numbers and community findings will change, so re-check before relying on them.

---

## 0. Most important findings (summary)

1. **Rules.** Metric is ROC AUC on a predicted probability for `satisfaction`. You may make 10 submissions per day and choose 2 final submissions. Teams can have up to 3 people. The deadline is **31 Oct 2026, 23:59 UTC**. The public leaderboard uses about 20% of test and the private leaderboard the other 80%. External data is allowed. Prizes are Kaggle merchandise only, with no points or medals. Sources: [overview](https://www.kaggle.com/competitions/playground-series-s6e10/overview), [rules](https://www.kaggle.com/competitions/playground-series-s6e10/rules), [leaderboard](https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard).
2. **The official source dataset is not the teejmahal20 one.** The data page names `arseniyshutko/binary-aviation-satisfaction-129k`. That dataset has 21 features and **no `Inflight service` column**, which matches the competition data. teejmahal20 has 22 features. Sources: [data page](https://www.kaggle.com/competitions/playground-series-s6e10/data), [arseniyshutko](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k), [teejmahal20](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction).
3. **The leaderboard is extremely compressed.** On 2026-10-07 the public #1 scored 0.96183 and the top 50 were all at or above 0.96175 ([leaderboard](https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard)). Stacking the shared public out-of-fold (OOF) predictions already gives about 0.96163–0.96176 on the public LB ([746148](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746148), [746363](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746363)). The difference between two submissions on the public LB has noise of about ±0.0002 ([jamesidriss STATUS](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md)). **Choose models by CV, not by the public LB.**
4. **What the community has measured so far, with the CV AUC gain of each step:**
   - Fold-safe target encoding (TE) of the **exact `Flight Distance` value**: +0.00127 to +0.00135. It behaves like a route ID, with 3,474 distinct values.
   - A "teacher" model trained on the original data, with its prediction used as a feature: +0.00059.
   - 10 folds instead of 5: +0.0001 to +0.0002.
   - CatBoost with all columns treated as categorical: +0.0022 over native categorical handling.
   - CatBoost with 39 crosses of each rating with Class, Type of Travel and Customer Type: +0.00034.
   - Logistic regression on logits as the stacker.

   Sources: [745908](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745908), [745892](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745892), [thomasjpfan analysis](https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md), [ferreret](https://github.com/ferreret/kaggle-playground-s6e10).
5. **Use the original dataset as knowledge, not as extra rows.** Appending the original rows *hurts*: −0.00035 to −0.0011, and the harm grows with the sample weight. Statistics computed from the original data and a teacher model trained on it *help*. Sources: [745098](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745098), [shahed-swe PLAN](https://github.com/shahed-swe/playground-series-s6e10/blob/HEAD/PLAN.md), [jamesidriss STATUS](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md).
6. **Tabular foundation models (TFMs) are this month's new source of signal.**
   - TabPFN-3.5 reaches 0.961145 OOF on the raw 21 features with no feature engineering. Tuned XGBoost reaches 0.959852.
   - Adding TFMs to the public OOF stack moved the public LB from 0.96167 to 0.96176.
   - Extra GBDT and RealMLP members now add only about +0.00001 to +0.00003 each.

   Sources: [746363](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746363), [746148](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746148).
7. **Pattern across recent Playground winners.**
   - Typical recipe: one fixed set of folds, a wide and genuinely different pool of models (GBDTs plus RealMLP and TabM, now plus TFMs), heavy feature engineering on exact values (TE, counts, digits, statistics from the original data), and a **linear meta-learner** to combine them (logistic regression on logits, or ridge).
   - Hill climbing and nonlinear stackers usually lost to logistic regression or ridge in 2026.
   - Exception: S6E8 was won by one RealMLP model.
   - Sources: details and URLs in Part B.

---

## Part A: S6E10 specifics

### A1. Competition facts

| Item | Value | Source |
|---|---|---|
| Title / slug | "Predicting Airline Satisfaction", Playground Series Season 6 Episode 10 | https://www.kaggle.com/competitions/playground-series-s6e10/overview |
| Host / citation | Kaggle; "Yao Yan, Walter Reade, Elizabeth Park. Predicting Airline Satisfaction … 2026" | https://www.kaggle.com/competitions/playground-series-s6e10/overview |
| Metric | Area under the ROC curve between the predicted probability and the observed target | https://www.kaggle.com/competitions/playground-series-s6e10/overview |
| Target | `satisfaction`; you predict its probability | https://www.kaggle.com/competitions/playground-series-s6e10/data |
| Target encoding in the file | Boolean True/False (True = satisfied). Positive rate 44.36% (310,339 / 699,635) | https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md , https://github.com/tuannm3812/kaggle-s6e10-predicting-airline-satisfaction/blob/HEAD/docs/2_eda_insights.md , https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction/blob/HEAD/reports/eda.md |
| ID column | `id`. Test ids run on from train: 699,635–999,478 | https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md |
| Submission format | Header `id,satisfaction`, then one probability per test id (for example `699635,0.2`) | https://www.kaggle.com/competitions/playground-series-s6e10/overview |
| Files | `train.csv`, `test.csv`, `sample_submission.csv`; 102.54 MB in total; 47 columns across the 3 files (= 2 + 22 + 23) | https://www.kaggle.com/competitions/playground-series-s6e10/data |
| Shapes | train 699,635 × 23 (id + 21 features + target); test 299,844 × 22 | https://github.com/tuannm3812/kaggle-s6e10-predicting-airline-satisfaction/blob/HEAD/docs/2_eda_insights.md , https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md |
| Start date | October 1, 2026 | https://www.kaggle.com/competitions/playground-series-s6e10/overview |
| Entry and team-merger deadlines | Same as the final submission deadline | https://www.kaggle.com/competitions/playground-series-s6e10/overview |
| **Final submission deadline** | **October 31, 2026, 11:59 PM UTC** ("All deadlines are at 11:59 PM UTC … unless otherwise noted") | https://www.kaggle.com/competitions/playground-series-s6e10/overview |
| Daily submission limit | **10 per day** | https://www.kaggle.com/competitions/playground-series-s6e10/rules |
| Final submissions | **Up to 2** selected for judging | https://www.kaggle.com/competitions/playground-series-s6e10/rules |
| Team size | Up to 3; mergers allowed, subject to the combined submission-count cap | https://www.kaggle.com/competitions/playground-series-s6e10/rules |
| Public / private split | Public LB is "approximately 20% of the test data"; final standings use the other 80% | https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard |
| Prizes | 1st–3rd: choice of Kaggle merchandise, awarded only once per person across the series. "Does not award Points or Medals" | https://www.kaggle.com/competitions/playground-series-s6e10/overview |
| External data | Allowed if it is publicly available, equally accessible and of minimal cost ("Reasonableness Standard"). AutoML tools allowed | https://www.kaggle.com/competitions/playground-series-s6e10/rules |
| Code sharing | No private sharing between teams. Public sharing only on Kaggle forums or notebooks | https://www.kaggle.com/competitions/playground-series-s6e10/rules |
| Data licence | CC BY 4.0 | https://www.kaggle.com/competitions/playground-series-s6e10/data |
| Participation (snapshot) | 2,241 entrants, 1,006 teams, 7,531 submissions (about day 6) | https://www.kaggle.com/competitions/playground-series-s6e10/overview |

### A2. Data dictionary

The competition columns keep the original names, **including spaces and a slash**. Quote them exactly in code.

| # | Column | Type | Values / notes | Source |
|---|---|---|---|---|
| 0 | `id` | int | Row id. Carries no signal: digit, modulo and bin probes all give AUC ≈ 0.500 | https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md |
| 1 | `Gender` | categorical (2) | Female / Male. Weakest feature (univariate AUC 0.5056) | https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction/blob/HEAD/reports/eda.md |
| 2 | `Customer Type` | categorical (2) | `Loyal Customer` (82.5%, 49.5% satisfied) / `disloyal Customer` (17.5%, 20.1% satisfied) | https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md |
| 3 | `Age` | int | 75 unique values. Effect is not monotonic: about 60% satisfied at 41–60, 14% at ≤18 and >60 | https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md |
| 4 | `Type of Travel` | categorical (2) | `Business travel` (58.4% satisfied) / `Personal Travel` (9.7% satisfied) | https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction/blob/HEAD/reports/eda.md |
| 5 | `Class` | categorical (3) | `Business` (72.5%) / `Eco` (16.8%) / `Eco Plus` (24.2%) | https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction/blob/HEAD/reports/eda.md |
| 6 | `Flight Distance` | int | 3,474 unique values. Behaves like a route ID; satisfaction rises with distance | https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md |
| 7–19 | 13 service ratings: `Inflight wifi service`, `Departure/Arrival time convenient`, `Ease of Online booking`, `Gate location`, `Food and drink`, `Online boarding`, `Seat comfort`, `Inflight entertainment`, `On-board service`, `Leg room service`, `Baggage handling`, `Checkin service`, `Cleanliness` | int 0–5 | **0 means "not applicable", not "worst"**. `Baggage handling` has no zeros. `Online boarding` is the strongest single feature (AUC 0.8404) | https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction/blob/HEAD/reports/eda.md , https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md |
| 20 | `Departure Delay in Minutes` | int | About 90% zeros, right-skewed (max ≈ 490) | https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md |
| 21 | `Arrival Delay in Minutes` | float | **Only column with missing values: 292 train / 130 test.** Correlation with departure delay r = 0.84 | https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction/blob/HEAD/reports/eda.md |
| 22 | `satisfaction` (train only) | bool target | True = satisfied, 44.36% positive | https://github.com/tuannm3812/kaggle-s6e10-predicting-airline-satisfaction/blob/HEAD/docs/2_eda_insights.md |

**Discrepancies to be aware of**
- The task brief cites teejmahal20 as the original dataset. The competition data page cites **arseniyshutko/binary-aviation-satisfaction-129k** ([data page](https://www.kaggle.com/competitions/playground-series-s6e10/data)).
  - arseniyshutko: 129,880 rows; 21 features + target; 13 service ratings with no `Inflight service`; CC0 licence; target True/False ([dataset card](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k)).
  - teejmahal20: includes `Inflight service`; labels are "satisfied" / "neutral or dissatisfied" ([dataset card](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)).
  - One competitor confirmed the schema fingerprint against arseniyshutko: 3 Class levels, 5 Baggage handling levels, 75 Age levels, and 100% of synthetic values present in the original ([jamesidriss STATUS](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md)).
  - The same author lists the teejmahal20 train/test files as a verbatim mirror of the same survey data ([STATUS](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md)). This is unverified by me.
- Some secondary write-ups miscount the columns:
  - shahed-swe says "22 features" ([README](https://github.com/shahed-swe/playground-series-s6e10/blob/HEAD/README.md)).
  - thomasjpfan says "14 ordinal ratings", but his own table lists 13 ([analysis](https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md)).
  - One discussion says "55.6% satisfied" ([745555](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745555)). That contradicts three independent EDAs that give 44.36% positive.

### A3. Known quirks and measured insights (each with its source)

**Data quality and train/test shift**
- **Missing values.** Only `Arrival Delay in Minutes` has any (292 train / 130 test). Missing rows are 36.3% satisfied, against 44.4% overall ([tuannm3812 EDA](https://github.com/tuannm3812/kaggle-s6e10-predicting-airline-satisfaction/blob/HEAD/docs/2_eda_insights.md)). Trees handle NaN natively. Filling the missing values had no effect beyond noise ([745098](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745098)).
- **No train/test shift.**
  - Adversarial validation AUC (a classifier trying to tell train from test) is 0.5004 ([tuannm3812](https://github.com/tuannm3812/kaggle-s6e10-predicting-airline-satisfaction/blob/HEAD/docs/2_eda_insights.md)), 0.4987 ([ferreret](https://github.com/ferreret/kaggle-playground-s6e10)) and 0.50027 ([jamesidriss](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md)).
  - One post claimed a "Cleanliness covariate shift" (KS p = 0.0376) ([744809](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/744809)). That is a single false positive out of 21 tests, as a commenter on that thread and jamesidriss both point out ([744809 comments](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/744809), [STATUS](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md)).
- **No leakage shortcuts.**
  - There are no duplicate feature rows within train or test, and no test row matches a train row exactly.
  - The original dataset overlaps train by 17 rows and test by 4 ([745932](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745932)).
  - The id–target AUC is 0.50007 ([shahed-swe PLAN](https://github.com/shahed-swe/playground-series-s6e10/blob/HEAD/PLAN.md)).

**Feature behaviour**
- **A rating of 0 is a "not applicable" sentinel.**
  - Wifi = 0 → 88.7% satisfied, against 39.1% at 1. Online boarding = 0 → 62.0%, against 10.5% at 1 ([744809](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/744809), [thomasjpfan](https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md)).
  - Linear models lose about 0.027 AUC unless the ratings are one-hot encoded: logistic regression scores 0.92156 with numeric ratings and 0.94859 with one-hot ratings ([ferreret](https://github.com/ferreret/kaggle-playground-s6e10)).
  - For gradient-boosted trees (GBDTs), treating ratings as categories made no difference beyond noise ([745098](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745098), [shahed-swe](https://github.com/shahed-swe/playground-series-s6e10/blob/HEAD/PLAN.md)).
- **`Flight Distance` behaves like a route ID.** This is the generator artifact (a trace left by the synthetic-data generator) that most top approaches exploit.
  - It has 3,474 distinct values. Each exact value has its own satisfaction rate, about 19× more variation than binomial noise would explain ([thomasjpfan](https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md)).
  - Only 4 of 699,635 train rows have a distance that does not appear in the original data ([ferreret](https://github.com/ferreret/kaggle-playground-s6e10)).
  - Fold-safe TE of the exact value: +0.00127 CV and +0.00125 LB ([ferreret](https://github.com/ferreret/kaggle-playground-s6e10)), and +0.00135 ([745908](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745908)).
  - LightGBM needs `max_bin=4095`. The default 255 bins merge the exact distances ([thomasjpfan](https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md)).
  - "Route profile" features (per-distance means and spreads of the ratings) add about +0.00014 ([745908](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745908), [745555](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745555)).
  - Digit features of the flight distance: +0.00036 ([745098](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745098)).
- **CatBoost does best when every column is treated as categorical.**
  - The "allcat" setup (numerics also passed as categories) beats native categorical handling by +0.0022. The tuned version reaches 0.96075 OOF, with depth 7 and `max_ctr_complexity=4` ([thomasjpfan](https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md)).
  - Adding 39 crosses (each of the 13 ratings × `Class` / `Type of Travel` / `Customer Type`) gives +0.000337 OOF and improves 5 of 5 folds, using GPU CatBoost (depth 6, lr 0.05) ([745892](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745892)).
- **Hand-crafted rating aggregates do not help GBDTs.**
  - Examples: zero counts, rating mean/min, the travel × class cross.
  - Measured effect: −0.0001 to −0.00023 ([thomasjpfan](https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md), [shahed-swe](https://github.com/shahed-swe/playground-series-s6e10/blob/HEAD/PLAN.md), [aditbytes](https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction)).

**The original dataset: does adding it help?**
- **Not as extra training rows.**
  - −0.00035 on all 5 folds ([shahed-swe](https://github.com/shahed-swe/playground-series-s6e10/blob/HEAD/PLAN.md)) and −0.00038 ([745098](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745098); that author notes a 100-round cap as a caveat).
  - −0.000585 at sample weight 0.3 and −0.001103 at weight 1.0, so the harm grows with the weight ([jamesidriss](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md)).
  - Likely reason: the generator shifted the continuous columns. The univariate AUC of `Flight Distance` is 0.696 in the competition data versus 0.649 in the original, and `Age` is 0.624 versus 0.584 ([745932](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745932)).
- **Yes, as knowledge.**
  - A teacher model trained only on the original data, used as a logit feature: +0.00059 ([745908](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745908)). TE plus teacher plus frequency encoding together gave +0.00147 ([shahed-swe](https://github.com/shahed-swe/playground-series-s6e10/blob/HEAD/PLAN.md)).
  - Smoothed P(y | exact value) per column computed on the original data: +0.001015. Once those statistics are present, the teacher adds only +0.000035 ([jamesidriss](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md)).
  - One competitor's final model used `orig_proba` / `orig_logit` features from an XGBoost trained on the original data ([alvis1111](https://github.com/alvis1111/s6e10-airline-solution)).

**Label noise sets a ceiling**
- A model trained on the original data scores 0.995 on the original but only about 0.954 on the synthetic train ([ferreret](https://github.com/ferreret/kaggle-playground-s6e10)). A teacher reached 0.99515 on the original and 0.95463 on the competition train ([shahed-swe](https://github.com/shahed-swe/playground-series-s6e10/blob/HEAD/PLAN.md)).
- About 3.7% of rows are confidently wrong (|pred − y| > 0.9) ([thomasjpfan](https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md)).
- About 49% of rows share a key made of the 4 categoricals plus all 13 ratings. Target-encoding that key gives no gain (−0.00005). This is consistent with labels being drawn independently per row ([jamesidriss](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md)).

**Training levers**
- **10 folds instead of 5:** +0.00009 to +0.00018 per model ([745908](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745908)); +0.00017 ([shahed-swe](https://github.com/shahed-swe/playground-series-s6e10/blob/HEAD/PLAN.md)); about +0.00011 ([jamesidriss](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md)).
- **Single-source levers. Treat these as unverified until we reproduce them.**
  - LightGBM `extra_trees=True`: +0.00023 on a 285-feature view. It **hurts** on the raw 21 features (−0.0056 when re-checked on a second, independent fold split).
  - RealMLP `n_ens` 8 → 32: +0.0002.
  - Both claims are from [jamesidriss STATUS](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md).

**Tabular foundation models (TFMs)**
- Chris Deotte's table ran each model with full context (about 560k rows per fold, no subsampling) on an A100 ([746363](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746363)). OOF AUC:
  - TabPFN-3.5: 0.961145
  - Kumo-Tabular large: 0.960036
  - tuned XGBoost (raw features): 0.959852
  - tuned RealMLP: 0.959311
  - TabFM: 0.958871
  - TabICL2: 0.958088
  - Mitra2: 0.957075
- TFMs lifted the public-OOF stack from LB 0.96167 to 0.96176, and 5 of the top 20 of 135 members by importance are TFMs ([746363](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746363)).
- Kaggle-GPU advice from the same thread: few `n_estimators` (about 2) and 25k–100k rows of context. TabFM AUC rises with context: 0.95752 at 25k, 0.95888 at full context.
- The OOF and test predictions are public: https://www.kaggle.com/datasets/cdeotte/s6e10-tfm-oof-and-test-predictions

**Shared public predictions and stacking**
- The community split is `StratifiedKFold(5, shuffle=True, random_state=42)`, which makes everyone's OOF predictions stackable together ([746148](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746148), [shahed-swe](https://github.com/shahed-swe/playground-series-s6e10/blob/HEAD/README.md)).
- Stacking all 119 public members with logistic regression on logits: nested CV 0.96200, public LB 0.96163 ([746148](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746148)).
  - In that stack, 10-fold members add about +0.0001 and TabPFN-3.5 / TabICL members about +0.0001. Other GBDT and RealMLP families add only +0.00001 to +0.00003.
  - Four new 10-fold members (XGBoost, RealMLP, two TabM), each 0.9600–0.9611 alone, added nothing: 0.962013 → 0.962002.
- Chris Deotte's 124-member public OOF aggregation is published at https://www.kaggle.com/datasets/cdeotte/s6e10-124-public-oof-and-test-predictions. One team stacked 137 members (12 own + Deotte's 124 public + 11 TFM, after removing near-duplicates): CV 0.96210, public LB 0.96174 ([shahed-swe](https://github.com/shahed-swe/playground-series-s6e10/blob/HEAD/README.md)).
- Plain logistic regression beat every fancier stacker sachith7 tried ([745908](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745908)).
- jamesidriss tested a public "local reliability gating" post-processing method (credited with LB 0.96152) under nested CV and measured a gain of 0.000000 ([STATUS](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md)).

**CV versus the public LB**
- LightGBM CV 0.95882 vs LB 0.95848; the next model gained +0.00127 CV and +0.00125 LB ([ferreret](https://github.com/ferreret/kaggle-playground-s6e10)). CV was about 0.0006 optimistic in another baseline ([shahed-swe](https://github.com/shahed-swe/playground-series-s6e10/blob/HEAD/PLAN.md)). CV and LB agreed within 0.0006 in a third ([aditbytes](https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction)).
- Paired public-LB noise is about ±0.0002, so only OOF gains of about +0.0003 or more show up on the board ([jamesidriss](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md)).

### A4. Leaderboard snapshot and baselines

| Snapshot | Value | Source |
|---|---|---|
| 2026-10-07, public #1 | 0.96183 (isthereanycode) | https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard |
| 2026-10-07, public #2–#4 | 0.96181 | same |
| 2026-10-07, #39 (Chris Deotte, 1 entry) | 0.96176 | same |
| 2026-10-07, #49 | 0.96175; about 1,008 teams | same |
| 2026-10-04 | leader 0.96158, rank 100 at 0.96105 | https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md |
| 2026-10-05 | top 100 at about 0.96150 | https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction |
| LightGBM, raw 21 features | CV about 0.9588–0.9590 / LB about 0.9584–0.9585 | https://github.com/ferreret/kaggle-playground-s6e10 , https://github.com/aditbytes/kaggle-s6e10-airline-satisfaction |
| Logistic regression with one-hot ratings | CV 0.94859 | https://github.com/ferreret/kaggle-playground-s6e10 |
| Strong single GBDT with exact-value TE, counts and teacher | about 0.9603–0.9612 OOF | https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md , https://github.com/alvis1111/s6e10-airline-solution |

**Interpretation.** The gap from a raw baseline (about 0.9585) to the top (about 0.9618) is about 0.003. Roughly half comes from exact-value encodings. Most of the rest comes from model diversity (TFMs, CatBoost allcat, NNs) and linear stacking. The top 50 sit within 0.00008 of each other on the public LB, which is inside the ±0.0002 public noise. The private-LB order will be set by CV-sound choices and luck in which two submissions are selected (see S6E8 and S6E9 below).

---

## Part B: What wins Playground tabular competitions (Season 5 and 6, binary AUC)

### B1. Winning solutions by episode (binary classification, ROC AUC)

"Final blend size" is the number of members in the submitted ensemble.

| Episode | Winner (private) | Models | Feature engineering | CV | Ensembling | Pseudo-labels / GPU | Final blend size | Source |
|---|---|---|---|---|---|---|---|---|
| **S6E9** EV purchase (Sep 2026) | Chris Deotte, **0.94602** (3rd public, 1st private) | Mostly XGBoost plus NNs (RealMLP, TabM, MLPs), some linear models, ports of public models, "a few special models" | Not itemised. 2nd/3rd place used exact-value TE, rounded income, original-data statistics, GPT-2 BPE token features | **One shared 5-fold file (`folds.npy`)** for every agent and model | **NVIDIA cuML logistic regression on logits** | **No pseudo-labels.** Built by LLM agents in a relay (GPT-5.6 Sol, Fable 5/5.1, GPT-6-Astra, Opus 5.5) | **60** | https://www.kaggle.com/competitions/playground-series-s6e9/writeups/1st-place-the-ai-relay , https://www.kaggle.com/competitions/playground-series-s6e9/leaderboard |
| **S6E8** smartphone addiction (Aug 2026) | Chris Deotte, public LB 0.97174 (private not published) | **A single stock PyTabKit `RealMLP_TD_Classifier`** (CV 0.97070). A 380-model ensemble was built but not needed | Withheld. CV ladder: raw 0.9660 → basic FE 0.9695 → tuning +6e-4 → advanced FE +6e-4 | Not stated | None. "A single model has not won Kaggle's Playground competition in 18 months" | LLM agent swarm (Codex GPT 5.6 Sol, Fable 5, ChatGPT Pro, NVIDIA Inference Hub LLMs) | **1** (of 380 built) | https://www.kaggle.com/competitions/playground-series-s6e8/writeups/1st-place-distributed-intelligence-nvidia-infe |
| **S6E5** F1 pit stops (May 2026) | Optimistix, **0.95503** | XGB, LGBM, CatBoost, RealMLP, TabM, HGB, RF, YDF, FT-Transformer, MLP-PLR; AutoGluon, LightAutoML, FLAML, PyTabKit | "Heavy FE" notebooks with 200–400 features each. Original data at sample weight 0.5–1 (0.25 hurt). "Driver_Dropped" models because `Driver` shifted most between original and competition data | Not stated | AutoGluon meta-model and logistic regression on logits, then a **50/50 blend of the two** (won by +0.00001). 4 of the 186 columns are level-2 OOFs | Used Claude all month | **186 OOF columns** | https://www.kaggle.com/competitions/playground-series-s6e5/writeups/1st-place-by-the-skin-of-my-teeth |
| **S6E3** customer churn (Mar 2026) | Chris Deotte (private score not published in the writeup) | 850 models built, 150 selected: XGB (including XGBRanker), LightGBM, CatBoost, YDF, cuML RF, plus 25 NN families (RealMLP, TabM, TabICL, FT-Transformer, GNN, DAE, FFM…) | **"Snap" features** (nearest original IBM value, and the difference from it); **decimal digit extraction**; nested 5×5 TE on bigrams and trigrams; original-data priors; count and frequency ratios; arithmetic deviations (`TC − tenure × MC`); quantile bins up to 5,000 | 5-fold `StratifiedKFold`, SEED = 42; nested 5×5 OOF between levels | **4-level stack**, ending in cuML L2 logistic regression. Ridge and logistic regression beat hill climbing on CV | Agents GPT5.4, Gemini3.1, ClaudeOpus4.6; 4× A100; cuDF/cuML; 600k lines of code | **150** | https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm |
| **S6E2** heart disease (Feb 2026) | masayakawamata, **0.95535** | XGB, LGB, CatBoost, RealMLP (`n_ens=8`), RGF, TabICL, AutoGluon across about 10 feature views | Feature views: BIN (qcut/cut/round), DIGIT, ALL_CATS (all columns as strings), FREQ, GP features (gplearn), ORIG (target mean, Weight of Evidence and entropy from the original data), denoising VAE latents | 5-fold `StratifiedKFold(shuffle=True, random_state=42)` | Optuna search over subsets (2,500 trials) → **ridge** on the selected OOFs. **Deliberately submitted a CV in the 0.95578–0.95580 band, not the maximum CV** | Pseudo-labelling **did not work** | About 150 candidate OOFs, about 1/10 consistently selected | https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t (via [BlamerX digest](https://github.com/BlamerX/Kaggle-Competition-Season-6/blob/main/writeup_digests/season6/S6E2.md)) |
| **S5E12** diabetes (Dec 2025) | wind1234it, **0.70504** | "Many diverse" tree and NN models | 12 ratio pairs, squares and cubes of the top numerics, TE; original data concatenated as rows | "Post-cutoff" CV that targets the id shift between train and test (credited to masayakawamata) | Hill climbing to select members → **ridge (alpha = 10) on rank-transformed predictions** of the top 36 | Not stated | **36** | https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl , https://www.kaggle.com/competitions/playground-series-s5e12/leaderboard |
| **S5E11** loan payback (Nov 2025) | Mahog (mahoganybuttstrings), **0.92939** | 100 models across 23 families (XGB best single: CV 0.928175 / private 0.92923) | "Probably the most important part": pairwise combinations + TE and count encoding (CE); **digits** and digit pairs, triples and quadruples + TE/CE; TE computed on the original dataset; rounding | Not stated (his notebook uses 20 splits) | Ridge or hill climbing ("HC slightly better on CV, about the same on private"). Nonlinear stackers were "much worse than linear methods" | Not stated | **100** | https://www.kaggle.com/competitions/playground-series-s5e11/writeups/1st-place-a-lot-of-features-a-lot-of-models-an (via [BlamerX digest](https://github.com/BlamerX/Kaggle-Competition-Season-6/blob/main/writeup_digests/season5/S5E11.md)) |
| **S5E8** bank marketing (Aug 2025) | 1st Optimistix, 0.97801 (**no writeup found**). 2nd Mahog, 0.97796 | 2nd: 59 models from 13 families (TabM best, then XGB, LGBM, RealMLP, CatBoost, DeepTables, TabR, GANDALF, …) | 2nd: TE mean and CE on bigrams, using both competition and **original** targets; products of bigrams; cyclical features | Not stated | 2nd: **CatBoost meta-learner** (CatBoost 0.97796 = LightGBM 0.97796 > hill climbing 0.97789 > ridge 0.97786 on private) | Not stated | 59 | https://www.kaggle.com/competitions/playground-series-s5e8/leaderboard , https://www.kaggle.com/competitions/playground-series-s5e8/writeups/2nd-place-yet-another-ensemble |
| **S5E3** rainfall (Mar 2025) | 1st: no writeup in the index. 2nd Chris Deotte, 0.90604 | XGBoost, TabPFN, RAPIDS SVC, … | **No FE** (the data is effectively 366 real rows) | GroupKFold with one synthetic year per fold | Equal-weight blend | GPU (RAPIDS) | 6 | https://www.kaggle.com/c/playground-series-s5e3/discussion/571176 (via [BlamerX digest](https://github.com/BlamerX/Kaggle-Competition-Season-6/blob/main/writeup_digests/season5/S5E3.md)) |

### B2. Runner-up writeups with methods worth copying

- **S6E9, 2nd place (Team Alicia, 0.94588)** ([writeup](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/2nd-place-solution)):
  - **TabPFN-3.5 with the full labelled context** (about 428k rows per fold, no subsampling) was the most valuable member: dropping it costs −2.40e-5. AUC rose about +1.86e-4 per doubling of context (the writeup's "+18.6u", where 1u = 1e-5), with no sign of saturating.
  - They used a **sealed holdout** and a 50k-row screening subset, logged every candidate before seeing its result, and required gains of at least +1e-5, 2 fold-standard-errors and 4/5 folds.
  - The final combiner was an **AUC-direct blend** (a smooth approximation to AUC over all positive–negative pairs, computed exactly with an FFT): +1.62e-5 over a log-loss stack.
  - Fitting submissions to the public LB raised public to 0.94945 but **dropped private to 0.94313** (about 1646th).
  - Things that did not help in their stack: adding the original rows, pseudo-labels, nonlinear level-3 combiners.
- **S6E9, 3rd place (M & M, 0.94584)** ([writeup](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/3rd-place-solution-feature-engineering-and-logis)):
  - L2-regularised logistic regression on clipped logits of **115 columns**, following Deotte's S6E5 recipe.
  - Shared `StratifiedKFold(5, shuffle=True, random_state=42)`.
  - Exact-value and rounded-value TE.
  - **Soft** pseudo-labels from fold-excluded teachers helped. Hard labels did worse than no pseudo-labels.
  - Appending the original rows did not help, but original-data statistics did.
  - A GLM's prediction used as the offset for LightGBM gave +0.000176 single-model CV.
- **S6E5, 2nd place (Chris Deotte, 0.95502)** ([writeup](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/2nd-place-autonomous-codex-yolo)):
  - An autonomous Codex agent built **218 models across 37 families** in about 48 hours on 4× A100.
  - Submission: cuML logistic regression on logits. Every model above about 0.930 CV was included, with no selection, because selection did worse in his past competitions.
  - Every model, including public notebooks, was rerun on his own 5 folds.
  - Best family was RealMLP (CV 0.954426). Fine-tuned TabICL ended up as a heavily weighted member.
  - TE was always nested-fold-safe.
- **S6E5, 5th place** ([writeup](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/5th-place-solution-a-99-model-logit-stack)):
  - **Average the ranks, not the probabilities**, when blending two ensembles on different scales: the unsubmitted rank average scored 0.95505 private versus 0.95503 ([BlamerX digest](https://github.com/BlamerX/Kaggle-Competition-Season-6/blob/main/writeup_digests/season6/S6E5.md)).
  - A weak but different Deep FFM model (0.918) was worth more than another LightGBM.
- **S6E8, 2nd / 7th / 14th** ([2nd](https://www.kaggle.com/c/playground-series-s6e8/writeups/2nd-place-solution), [7th](https://www.kaggle.com/c/playground-series-s6e8/writeups/7th-place-many-models-one-simple-stack), [14th](https://www.kaggle.com/c/playground-series-s6e8/writeups/14th-place-278-shared-oof-sets-36-own-models)):
  - **Exact-value TE and exact-value categorical copies next to the numeric columns** were the main feature-engineering lever. 14th: LightGBM went from 0.96276 to 0.96605 (+0.00329). Making every column categorical and dropping the numerics *hurt* (−0.00417).
  - All three used **logistic regression as the meta-learner**. Ridge, AutoGluon and tree or NN blenders failed for them.
  - **Shared community OOFs beat extra models of their own.** 14th: +0.00048 nested CV from 207 shared columns.
  - Week 4 of 2nd place's blender work bought only +0.00003. His verdict: "I thought I needed a better blender. In fact, I needed a better model."
  - Source: [BlamerX S6E8 digest](https://github.com/BlamerX/Kaggle-Competition-Season-6/blob/main/writeup_digests/season6/S6E8.md).
- **S6E3, 16th** ([writeup](https://www.kaggle.com/c/playground-series-s6e3/writeups/16th-place-solution-ridge-ensembling)): a ridge meta-learner fed degree-2 products of the OOF columns (34 → 561 inputs) gave +0.00004 to +0.00008 on private ([digest](https://github.com/BlamerX/Kaggle-Competition-Season-6/blob/main/writeup_digests/season6/S6E3.md)).

### B3. Recurring patterns, ranked by how often they appear

Counted over the 9 top solutions in the B1 table: the 1st-place writeup, or the best available writeup where 1st published none (S5E8 and S5E3 use 2nd place).

| Rank | Pattern | Count (of 9) | Where it appears |
|---|---|---|---|
| 1 | **Wide, diverse pool from different model families**: XGB/LGBM/CatBoost plus RealMLP/TabM, often TabICL/TabPFN or AutoGluon. Diversity comes from different families and feature views, not from seeds | 9 | All. S6E8 built 380 models but submitted a single one |
| 2 | **One fixed fold split reused for every model, with OOF and test predictions saved** (usually 5-fold `StratifiedKFold`, shuffle, rs = 42) | 7 stated (S6E9, S6E3, S6E2, S5E12, S5E3; S6E5 via 2nd place's practice; S6E8 via 2nd/7th/14th) | [S6E9-1st](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/1st-place-the-ai-relay), [S6E3-1st](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm) |
| 3 | **Trust CV, not the public LB, for selection.** Sometimes this means deliberately *not* submitting the maximum CV | 7 (S6E9, S6E8 via 2nd/14th, S6E5, S6E2, S5E12, S5E11, S5E3) | [S6E9-2nd](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/2nd-place-solution), [S6E2-1st](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t) |
| 4 | **Original dataset used as features** (target statistics, priors, snap values, nearest-neighbour labels, teacher models), not just appended as rows | 7 (S6E3, S6E2, S5E11, S5E8-2nd, S6E5, plus S6E9 2nd/3rd; S5E12 appended rows) | B1 table |
| 5 | **Fold-safe target encoding of exact values and of column combinations** (pairs and n-grams, often with extra statistics such as std and quantiles) | 6 confirmed (S6E3, S6E2, S5E12, S5E11, S5E8-2nd, S6E5-2nd); S6E8/S6E9 runners-up too. S6E8's winner withheld FE | B1 table |
| 6 | **Linear meta-learner**: logistic regression on logits (cuML or sklearn) or ridge. Usually beats hill climbing and nonlinear stackers | 6 (S6E9, S6E5, S6E3, S6E2, S5E12, S5E11). Exceptions: S5E8-2nd (CatBoost meta), S5E3-2nd (equal weights), S6E8 (single model) | B1 table |
| 7 | **Digit, rounding and binning features; numerics also cast as categorical** | 5 (S6E3, S6E2, S5E11, S6E9 runners-up, S6E8 runners-up) | B1 table, B2 |
| 8 | **Hill climbing** used to select members or as a candidate ensembler | 4 (S5E12 as the selector; S5E11 roughly equal to ridge; S6E3 and S6E2 tried it and it lost to logistic regression/ridge) | B1 table |
| 9 | **LLM coding agents** writing most of the code | 4 of 9 winners (S6E9, S6E8, S6E3; S6E5's 1st used Claude) | B1 table |
| 10 | **Frequency / count encoding** | 4 (S6E3, S6E2, S5E11, S5E8-2nd) | B1 table |
| 11 | **GPU acceleration** (cuDF, cuML logistic regression, A100s) | 4 (S6E9, S6E8, S6E3, S5E3-2nd) | B1 table |
| 12 | **Pseudo-labelling in the final solution** | 0 confirmed among winners. S6E9-1st: "no pseudo-labels". S6E2-1st: pseudo-labelling "did not work". Soft pseudo-labels helped S6E9's 3rd place | [S6E9-3rd](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/3rd-place-solution-feature-engineering-and-logis) |

**Cross-season architecture tally.** BlamerX's rollup of 386 writeups ([architecture_matrix.md](https://github.com/BlamerX/Kaggle-Competition-Season-6/blob/main/writeup_digests/architecture_matrix.md)) gives these winning topologies for Season 6:
- Two-level stack (STACK2): S6E1, S6E2, S6E4, S6E5, S6E6
- Deeper stack (STACKN): S6E3
- Single model: S6E8

For Season 5 binary episodes: S5E11 was a flat blend and S5E12 a two-level stack. The matrix does not yet include S6E9. S6E9 1st place was a 60-model logistic-regression stack (STACK2) per its [writeup](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/1st-place-the-ai-relay).

**Anti-patterns that repeatedly failed**
- Blending or tuning against the public LB. S6E9's 2nd place tilted toward the public labels and lost 0.0024 private ([writeup](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/2nd-place-solution)).
- Nonlinear or deep stackers once the pool is diverse. In S6E5 the level-3+ stacks finished 7th–11th ([digest](https://github.com/BlamerX/Kaggle-Competition-Season-6/blob/main/writeup_digests/season6/S6E5.md)).
- Averaging every OOF without selection or regularisation (S6E2-1st, [digest](https://github.com/BlamerX/Kaggle-Competition-Season-6/blob/main/writeup_digests/season6/S6E2.md)).
- Spending the last week on blenders instead of models (S6E8-2nd, [digest](https://github.com/BlamerX/Kaggle-Competition-Season-6/blob/main/writeup_digests/season6/S6E8.md)).

### B4. Chris Deotte (NVIDIA) published methodology

**NVIDIA "Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data"**
- Authors: Kazuki Onodera, Théo Viel, Gilberto Titericz, Chris Deotte. Published 2025-09-18 ([blog](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)).
- Two core principles: fast experimentation on GPU (cuDF, cuML, GPU XGBoost/LightGBM/CatBoost) and local k-fold validation matched to the data's structure.
- The seven techniques:
  1. **Smarter EDA**: compare train and test distributions; look for trends in the target.
  2. **Diverse baselines, fast**: linear, GBDT and NN models side by side. Example: S5E3's 2nd place used GBDT + NN + SVR with no feature engineering.
  3. **Generate more features**: hundreds to thousands of them. Example: 8 categorical columns → 28 pairwise combinations.
  4. **Hill climbing**: start from the best model and add others with weights while CV improves. Use CuPy to evaluate weights in parallel.
  5. **Stacking**: either on residuals or with OOF predictions as features. Example: a 3-level stack won "Podcast Listening Time".
  6. **Pseudo-labelling**: soft labels, multiple rounds, k separate sets of pseudo-labels for k-fold to avoid leakage.
  7. **Extra training**: average many seeds (about 100 XGBoost seeds in the Fertilizer competition) and retrain on 100% of the data.

**KGMON Playbook 2026** ([GitHub repo](https://github.com/cdeotte/KGMON-Playbook-2026))
- Notebooks: EDA → Feature Engineering (`%load_ext cudf.pandas`) → Baselines → Stacking (on OOF predictions *or* on residuals) → Hill Climbing → Pseudo Label / knowledge distillation → Conclusion (`%load_ext cuml.accel`).
- Its conclusion: "the key … is speed and experimentation."
- His S6E3 winning writeup lists the steps as EDA, Build Baselines, GPU Feature Engineering, GPU Hill Climbing, Stacking, Pseudo Labeling, GPU Extra Training ([S6E3-1st](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)).

**cuDF feature-engineering blog (S5E2 backpack competition, 1st place with a single model)** ([blog](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas), 2025-04-17)
- Feature recipes:
  - `groupby(COL1)[COL2].agg(STAT)` with mean, std, count, min, max, nunique and skew
  - histogram bins and quantiles within each group
  - all missing-value flags combined into one base-2 column
  - digit extraction
  - all 28 pairwise combinations of the categoricals
  - values looked up from the original dataset
  - ratio features
- Scale: more than 10,000 features generated, 500 kept for XGBoost.

**GenAI-assisted coding blog (S6E3)** ([blog](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/))
- Save OOF and test predictions for every experiment as numpy files.
- Run every experiment on GPU.
- Ask LLMs to find and read research papers for feature-engineering ideas.
- Use knowledge distillation from all the OOF and test predictions.

**Deotte's recipe across his 2026 writeups**
- One shared fold file for everything ([S6E9](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/1st-place-the-ai-relay), [S6E5](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/2nd-place-autonomous-codex-yolo)).
- Nested-fold-safe TE ([S6E5](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/2nd-place-autonomous-codex-yolo)).
- Build "really different" models, not seed or one-feature variants ([S6E5](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/2nd-place-autonomous-codex-yolo)).
- **cuML logistic regression on logits** over everything above a CV floor ([S6E5](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/2nd-place-autonomous-codex-yolo), [S6E9](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/1st-place-the-ai-relay)).
- Judge a model by how much it adds to the ensemble, not by its own score. Rule of thumb credited to Paul Bryan Elefante: look for models scoring above 0.94 that are less than 0.99 correlated with the existing pool ([S6E9](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/1st-place-the-ai-relay)).
- Have a second agent review any big improvement for leaks; write handoff notes (`plan.md`, `progress.md`, `discoveries.md`, a closed-ideas list); trust CV over the public LB ([S6E9](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/1st-place-the-ai-relay)).
- For synthetic data, mine the generator's artifacts: snap features and decimal digits ([S6E3](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)).
- This month he promotes TFMs as a "secret weapon" ([S6E10 discussion 746363](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746363)).

---

## Part C: Implications for our S6E10 plan

Ordered by expected gain per unit of effort. Each item is backed by the sources above.

1. **Validation first.**
   - Adopt `StratifiedKFold(5, shuffle=True, random_state=42)` and save it once to a file. This is the community split, so we can stack public OOFs ([746148](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746148)).
   - Save OOF and test predictions for every model ([Deotte blog](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/)).
   - Score every ensemble decision with nested CV ([S6E8-14th](https://www.kaggle.com/c/playground-series-s6e8/writeups/14th-place-278-shared-oof-sets-36-own-models), [S6E9-2nd](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/2nd-place-solution)).
2. **Core feature engineering.**
   - Exact-value fold-safe TE (`Flight Distance` first, then `Age` and the delays); count/frequency encoding; original-data smoothed target statistics plus an original-only teacher logit; flight-distance digits; route-profile aggregates. Expect about +0.0015 to +0.0019 over raw features ([745908](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745908), [jamesidriss](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md)).
   - Do **not** append the original rows.
3. **Model families.**
   - LightGBM with `max_bin=4095`; XGBoost.
   - CatBoost allcat with the 39 rating × context crosses.
   - RealMLP and TabM.
   - **TFMs: TabPFN-3.5, Kumo-Tabular.** These are the most "different" signal available. Watch the TabPFN-3.5 licence: non-commercial only, competition use allowed ([S6E9-2nd](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/2nd-place-solution)).
   - Train on 10 folds for the final members.
4. **Stack** with L2 logistic regression on clipped logits or ranks, nested CV-selected C ([S6E9-1st](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/1st-place-the-ai-relay), [S6E9-3rd](https://www.kaggle.com/competitions/playground-series-s6e9/writeups/3rd-place-solution-feature-engineering-and-logis)).
   - Optionally add public OOF libraries, provided they are leak-gated and their folds are confirmed ([shahed-swe](https://github.com/shahed-swe/playground-series-s6e10/blob/HEAD/PLAN.md)).
   - For the report: show results both with and without public OOFs, so our own contribution is clear.
5. **Final two slots.** Use one strict nested-CV best and one structurally different hedge, chosen by CV. Ignore public differences below about 0.0002 ([jamesidriss](https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md), [S6E8-7th](https://www.kaggle.com/c/playground-series-s6e8/writeups/7th-place-many-models-one-simple-stack)).
6. **Low priority, or skip.**
   - Hand-crafted rating aggregates.
   - Pseudo-labelling (if tried, soft labels only).
   - Deep or nonlinear stackers.
   - Hill climbing on the public LB.
   - "Region reliability gating".

---

## Top source URLs

1. https://www.kaggle.com/competitions/playground-series-s6e10/overview: metric, submission format, timeline, prizes
2. https://www.kaggle.com/competitions/playground-series-s6e10/rules: 10/day, 2 finals, team of 3, external data
3. https://www.kaggle.com/competitions/playground-series-s6e10/data: target name, official source dataset
4. https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k: original data (21 features)
5. https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746363: Deotte TFM benchmark on S6E10
6. https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746148: public OOF stacking, nested CV 0.96200
7. https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745908: step-by-step ablation (TE, teacher, 10 folds)
8. https://www.kaggle.com/competitions/playground-series-s6e9/writeups/1st-place-the-ai-relay: latest winning recipe (60-model cuML logistic-regression stack)
9. https://www.kaggle.com/competitions/playground-series-s6e9/writeups/2nd-place-solution: TabPFN-3.5 full context, nested gates, public-LB-fitting cost
10. https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/: NVIDIA playbook

Further references:
- https://github.com/BlamerX/Kaggle-Competition-Season-6 (digests and link index, secondary)
- https://github.com/thomasjpfan/kaggle-playground-series-s6e10/blob/HEAD/analysis.md
- https://github.com/jamesidriss/airline_S6E10/blob/HEAD/STATUS.md
- https://github.com/shahed-swe/playground-series-s6e10/blob/HEAD/PLAN.md
- https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm
- https://www.kaggle.com/competitions/playground-series-s6e8/writeups/1st-place-distributed-intelligence-nvidia-infe
- https://www.kaggle.com/competitions/playground-series-s6e5/writeups/1st-place-by-the-skin-of-my-teeth
- https://github.com/cdeotte/KGMON-Playbook-2026
- https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas
