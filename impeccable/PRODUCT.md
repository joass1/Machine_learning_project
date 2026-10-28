# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Slide deck, not an app. Delivery format: whichever file the Canva MCP can import as editable components (PPTX is the working assumption; confirm by test import), because the team will hand-refine the deck in Canva afterwards. Undecided until a test import succeeds.

## Users

Primary audience: the IS460 professor (Prof. Zhaoxia Wang) and graders, scoring the final report presentation against the Week 7/Week 11 criteria sheet. Presenters: Group 03 (Chai Yichen, Chetan s/o Haresh Kishnani, Lee Jia Le, Joash Lau Rong Wei, Cody Loo Yee, Tristan Ang Boon Yew). Graders read for rigour and validity of the synthetic-vs-real analysis, and score each rubric item 1 to 10.

## Product Purpose

"Cleared for Satisfaction": predicting airline passenger satisfaction with deep tabular ensembles, and testing whether a synthetic copy of a survey (Kaggle Playground S6E10, 699,635 synthetic passengers from a real survey of 129,880) can be trusted for airline decisions. The deck presents the final project: RQ1 (which techniques help real passengers vs. only exploit the generator), RQ2 (what the generator distorted, what the synthetic data is worth), RQ3 (which services drive satisfaction per segment, and whether synthetic and real give the same fix-first answer). Success is a high rubric score and a clear, defensible story within the time limit.

## Positioning

Instead of chasing leaderboard AUC on the synthetic data, the project asks whether the synthetic copy is trustworthy for real decisions, scoring every model in four settings (syn to syn, real to real, syn to real, real to syn). Findings live in `docs/RESULTS.md` with figures in `docs/figures/`.

## Operating Context

- Final report presentation: 15 minutes talk plus 5 minutes Q&A (about 20 minutes total).
- Deck length: 15 to 40 slides.
- Slide phases mirror the report phases, which follow the rubric: Title; Introduction / Problem Statement; Motivation (may be combined with Introduction); Literature Review; Datasets; Methodology; Results and Discussion; Conclusion and Future Work; References. Figures and tables need clear captions. Project difficulty should be evident.
- Source rubric: `../Project 26 - The Criteria released in Week 7 and utilized in Week 11--Reference for Project Final Report Evaluation Criteria_IS460.xlsx`. Report structure example: `../Project Proposal and Report Structure Example -IS460-Z. Wang.pdf`.

## Capabilities and Constraints

- Footer rule: a slide carries a source footer only when it quotes data or claims that are not the team's own findings (literature, dataset facts, external statistics), and the footer sits on that same slide. The title slide never has a footer, and slides showing only our own results have none. A footer must serve the reader, not fill space. Datasets need download links on the slide that describes them.
- If AI was used, it is disclosed on a dedicated slide at the end.
- Title must be concise, clear, unique, and make subject area and method explicit.
- Results slides need figures or charts with captions; tables need titles.
- Deck must import into Canva as editable components (real text, shapes, and native tables where possible; avoid flattened images of text) so it can be hand-refined.
- Content truth lives in the repo: `README.md`, `docs/PLAN.md`, `docs/METHODOLOGY.md`, `docs/RESULTS.md`, `docs/references.md`, `docs/research/literature.md`, `docs/figures/*.png`.
- Data: synthetic = Kaggle Playground S6E10 (CC BY 4.0); real = Kaggle `arseniyshutko/binary-aviation-satisfaction-129k` (CC0).

## Evidence on Hand

Ten generated figures in `docs/figures/` (rq1_ladder, rq1_tricks, rq2_delays, rq2_distortion, rq2_target_dependence, rq2_utility, rq3_agreement, rq3_permutation, rq3_shap_heatmap, rq3_whatif), full-size run results in `docs/RESULTS.md`, references in `docs/references.md`. Team-wide claims about AI usage, speaker assignments, and any institution branding are not yet confirmed and must not be invented.

## Product Principles

1. The rubric is the spine: slide order follows the grading criteria, and each slide earns marks for a named criterion.
2. Evidence over decoration: every claim is paired with a figure, table, or cited source on the same slide.
3. One idea per slide, sized for a 15-minute talk (roughly 15 to 25 core slides, appendix as buffer for Q&A).
4. Honest about limits: associational claims are labelled as such; nothing is stated beyond what `docs/RESULTS.md` supports.
5. Built to be edited: structure and text stay native and simple so the team can refine in Canva.

## Accessibility & Inclusion

Projected in a classroom: large legible text, high contrast, colour-blind-safe chart colours (figures already use a shared `plotstyle.py`).
