# Claims to source (verification pass, 28 Oct 2026)
Labels: V = Verified (primary source or repo file opened), R = Reported (secondary), H = Hypothesis (ours, untested). "Own" = the team's finding from the full-size runs of 25 Oct 2026 (docs/RESULTS.md), shown with no footer.

| Slide | Claim | Source | Label |
|---|---|---|---|
| 3, 7 | S6E10 has 699,635 synthetic rows; real survey has 129,880 | Kaggle S6E10 [1]; Kaggle dataset [33]; METHODOLOGY 2 | V (repo); Kaggle pages not re-opened (need JS/API) |
| 3, 7 | Licences CC BY 4.0 (synthetic), CC0 (real) | references.md [1], [33] | V (repo), not re-opened |
| 4 | 5.1 billion passengers expected in 2026 | IATA press release 7 Jun 2026 [3], re-opened 28 Oct | V |
| 4 | Net profit per passenger $4.50 (2026), half of $9.10 (2025) | IATA press release [3], re-opened 28 Oct | V |
| 4 | US airline satisfaction 76 on ACSI scale, one point below the record | ACSI Travel Study 2026 [5]; verified 7 Oct (literature.md M1); re-fetch 28 Oct blocked (HTTP 403) | V (earlier), not re-confirmed |
| 4 | New customer costs 5 to 25 times more than keeping one | Gallo, HBR 2014 [6], re-opened 28 Oct (the article hedges "depending on which study you believe") | V |
| 5 | Zhang: XGBoost test AUC 0.987, SHAP ranks Wi-Fi first | [15] via literature.md | V (abstract-level) |
| 5 | Hyassat et al.: LightGBM AUC 0.994, Wi-Fi and online boarding lead | [18] via literature.md | V (abstract-level) |
| 5 | Jiang et al.: RF accuracy 0.963, AUC 0.961 | [14] via literature.md; AUC may be from hard labels (our observation) | V, with caveat |
| 5 | Lee: stacking 96.3 %, bagging 96.1 % accuracy | [16] via literature.md | V |
| 5 | Satisfaction drives repurchase/recommendation; SERVQUAL | [7], [8] | V |
| 5, 6 | No study we found tests drivers or model choice on a synthetic copy | literature.md section 2.3 positioning | H (absence claim, limited to studies reviewed) |
| 6 | Trees remain state of the art on medium-sized tabular data | Grinsztajn et al. [23] | V |
| 6 | Deep+XGBoost ensemble beats XGBoost; cross-model ensembles advance state of the art | [24], [25] | V |
| 6 | TSTR is the standard fidelity check, introduced by Esteban et al. | [40] (references.md note) | V |
| 9 | Model descriptions (XGBoost, LightGBM, CatBoost, RealMLP, TabM, TabPFN) | [20], [21], [22], [26], [27], [28] | V |
| 8-10 | Evaluation protocol, ladder, compute, reproducibility | METHODOLOGY 3-4, 7; RESULTS 5 | Own |
| 11 | Table 1 (AUC of ten models in four settings) | RESULTS 1.1 | Own |
| 12 | Table 2 (paired deltas), tricks hurt on real | RESULTS 1.2 | Own |
| 13 | Table 4 (tau 0.42), stacking +0.0005 / +0.0002 | RESULTS 1.4, 1.3 | Own |
| 14 | Delays 91.6 % vs 56.2 % zero; AUC 0.832; 17 of 699,635 copies | RESULTS 2.1 | Own |
| 15 | Label noise 3.93 % vs 0.16 % (25x) | RESULTS 2.3 | Own |
| 16 | Copy worth about 21,700 real passengers | RESULTS 2.4 | Own |
| 17-19 | Grouped permutation; SHAP swap; what-if agrees | RESULTS 3.1, 3.2 | Own (associational) |
| 20 | Trust table | RESULTS 4 | Own |
| 21 | Limitations | RESULTS 6; METHODOLOGY 8 | Own |
| 22 | Future work | PLAN 9 next steps; second survey and A/B trials | H (proposals) |
| 24 | AI use | PLAN WP0 "Joash + Claude"; references.md (Higgsfield, proposal deck); this session | V for stated items; team to confirm the rest |

## Discrepancies found (shown, not resolved)
- Bootstrap resamples: RESULTS Table 2 says 200; METHODOLOGY section 3 says 1,000. Slide 9 states both.
- Reference numbering: docs/research/literature.md uses [1]-[56]; docs/references.md uses [1]-[45]. The deck follows references.md (as the brief says); literature.md numbers differ for the same works.
- Brief says synthetic AUC "understates real accuracy (0.995)". The 0.995 is a real-data AUC, so slides say AUC.
- idea-comparison.md (7 Oct) cites 9.2 % vs 43.5 % delayed passengers; RESULTS reports 91.6 % / 56.2 % zero arrival delay (8.4 % / 43.8 % delayed). Different measure (proposal stage vs full run); departure figures 90.8 % / 56.5 % match the proposal's 9.2 % / 43.5 %. Not used on slides.

## Not verified
- Kaggle competition and dataset pages (need JavaScript or API token).
- ACSI page this session (HTTP 403); relies on the 7 Oct verification.
- J.D. Power study and the Noviantoro and Huang full text (flagged [P] in literature.md); neither is used on a slide.
- Reference [43] (Pan et al.) carries a warning mark in references.md; not used.
- Fonts: Host Grotesk could not be set by the Canva tools; Canva's default grotesk is used until changed by hand.

## Claims cut
- Skytrax "24.8 million responses" (not on Skytrax's page, per literature.md).
- J.D. Power scores (page blocked to automated fetch).
- Any hero metric slide (big-number tiles) in favour of figures and tables.
