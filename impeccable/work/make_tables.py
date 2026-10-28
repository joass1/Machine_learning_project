"""Slide tables for the final deck, drawn in the project's figure style (docs/figures).
All numbers are copied from docs/RESULTS.md (Tables 1, 2, 4, 5 and section 4)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib.pyplot as plt
from airsat import plotstyle as ps
ps.apply()
OUT = ROOT / "impeccable" / "assets" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
PX = 1 / (72 * ps.SLIDE_PX_PER_PT)  # inches per slide px

def table(name, header, rows, widths, aligns, bold=(), tint=(), row_px=56, pad=28, fs=12):
    W = sum(widths) + 2 * pad
    H = row_px * (len(rows) + 1) + 2 * pad // 2
    fig = plt.figure(figsize=(W * PX, H * PX), dpi=ps.DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    xs = [pad]
    for w in widths: xs.append(xs[-1] + w)
    def put(ci, y, s, weight="normal", color=ps.INK):
        a = aligns[ci]
        x = xs[ci] + (10 if a == "l" else widths[ci] - 10)
        ax.text(x, y, s, ha="left" if a == "l" else "right", va="center", fontsize=fs, fontweight=weight, color=color)
    y0 = pad // 2
    for ci, h in enumerate(header): put(ci, y0 + row_px / 2, h, "bold", ps.INK_2)
    ax.plot([pad, W - pad], [y0 + row_px] * 2, color=ps.INK, lw=1.4)
    for ri, r in enumerate(rows):
        yc = y0 + row_px * (ri + 1) + row_px / 2
        if ri in tint:
            ax.add_patch(plt.Rectangle((pad, y0 + row_px * (ri + 1)), W - 2 * pad, row_px, color="#EEF4FC", lw=0))
        for ci, s in enumerate(r):
            put(ci, yc, s, "bold" if (ri, ci) in bold or ci == 0 and ri in tint else "normal")
        if ri < len(rows) - 1:
            ax.plot([pad, W - pad], [y0 + row_px * (ri + 2)] * 2, color=ps.GRID, lw=0.8)
    ax.plot([pad, W - pad], [y0 + row_px * (len(rows) + 1)] * 2, color=ps.AXIS, lw=1.0)
    fig.savefig(OUT / f"{name}.png", dpi=ps.DPI, facecolor=ps.SURFACE)
    plt.close(fig)
    print(name, int(W), "x", int(H), "slide px")

# Table 1 (RESULTS 1.1)
table("table1_ladder",
      ["Model", "S→S", "R→R", "S→R", "R→S"],
      [["TabM", "0.9585 (6)", "0.9954 (1)", "0.9941 (1)", "0.9517 (9)"],
       ["LightGBM", "0.9589 (4)", "0.9951 (2)", "0.9932 (4)", "0.9549 (2)"],
       ["CatBoost, all categorical", "0.9603 (1)", "0.9950 (3)", "0.9934 (3)", "0.9537 (5)"],
       ["RealMLP", "0.9580 (7)", "0.9950 (4)", "0.9937 (2)", "0.9534 (8)"],
       ["XGBoost, 1,024 bins", "0.9592 (3)", "0.9949 (5)", "0.9928 (6)", "0.9549 (3)"],
       ["XGBoost", "0.9587 (5)", "0.9948 (6)", "0.9928 (5)", "0.9549 (4)"],
       ["XGBoost + exact-value encoding", "0.9600 (2)", "0.9946 (7)", "0.9922 (7)", "0.9550 (1)"],
       ["MLP (scikit-learn)", "0.9566 (8)", "0.9934 (8)", "0.9921 (8)", "0.9534 (7)"],
       ["TabPFN (10k context)", "0.9544 (9)", "0.9931 (9)", "0.9892 (9)", "0.9534 (6)"],
       ["Logistic regression", "0.9486 (10)", "0.9784 (10)", "0.9714 (10)", "0.9472 (10)"]],
      [470, 190, 190, 190, 190], "lrrrr",
      bold={(0, 2), (0, 3), (2, 1), (6, 4)}, tint=(), row_px=58)

# Table 2 (RESULTS 1.2), delta AUC x 1e-4 with 95% paired bootstrap interval
table("table2_deltas",
      ["Technique (baseline → variant)", "Δ S→S (×10⁻⁴)", "Δ S→R (×10⁻⁴)"],
      [["Boosted trees (logreg → LightGBM)", "+102.8 [+100.3, +105.9]", "+218.3 [+211.0, +225.0]"],
       ["Hidden layers (logreg → MLP)", "+80.4 [+77.5, +83.3]", "+207.6 [+200.4, +213.9]"],
       ["XGBoost instead of LightGBM", "−1.6 [−2.6, −0.5]", "−4.5 [−5.4, −3.7]"],
       ["1,024 histogram bins", "+4.7 [+3.9, +5.5]", "−0.1 [−0.5, +0.3]"],
       ["Exact-value encoding", "+8.2 [+6.9, +9.3]", "−5.4 [−6.7, −4.3]"],
       ["CatBoost, all categorical", "+14.7 [+13.2, +16.3]", "+2.1 [+0.8, +3.8]"],
       ["RealMLP (vs LightGBM)", "−8.9 [−10.6, −7.1]", "+4.9 [+3.6, +6.2]"],
       ["TabM (vs LightGBM)", "−3.9 [−5.6, −2.2]", "+9.1 [+7.8, +10.2]"],
       ["TabPFN, 10k context (vs LightGBM)", "−44.6 [−46.9, −42.4]", "−40.2 [−42.6, −37.5]"]],
      [620, 420, 420], "lrr", bold={(3, 1), (4, 1), (4, 2), (5, 1), (6, 2), (7, 2)}, row_px=58)

# Table 4 (RESULTS 1.4)
table("table4_agreement",
      ["Settings compared", "Kendall τ", "p", "Same top model?"],
      [["S→S vs R→R", "0.42", "0.11", "no (CatBoost vs TabM)"],
       ["S→S vs S→R", "0.33", "0.22", "no (CatBoost vs TabM)"],
       ["R→R vs S→R", "0.82", "< 0.001", "yes (TabM)"],
       ["R→R vs R→S", "0.11", "0.73", "no"]],
      [330, 190, 190, 430], "lrrl", bold={(2, 1), (2, 2), (2, 3), (0, 1)}, tint=(2,), row_px=64)

# Table 5 (RESULTS 2.3)
table("table5_labelnoise",
      ["Confidence threshold", "Synthetic rows", "Real rows (out-of-fold)", "Ratio"],
      [["p > 0.90 or < 0.10", "4.33 %", "0.40 %", "11×"],
       ["p > 0.95 or < 0.05", "3.93 %", "0.16 %", "25×"],
       ["p > 0.99 or < 0.01", "3.34 %", "0.02 %", "207×"]],
      [480, 260, 400, 150], "lrrr", bold={(1, 1), (1, 3)}, tint=(1,), row_px=66)

# Table 6 (RESULTS 3.1)
table("table6_groups",
      ["Service group (AUC drop when permuted)", "Synthetic model", "Real model"],
      [["Digital (Wi-Fi, online booking, online boarding)", "0.116 (2nd)", "0.183 (1st)"],
       ["Profile (age, gender, loyalty, purpose, class, distance)", "0.152 (1st)", "0.170 (2nd)"],
       ["Ground (check-in, baggage, gate)", "0.011", "0.018"],
       ["Comfort (seat, leg room, cleanliness, entertainment, food)", "0.009", "0.011"],
       ["Crew (on-board service)", "0.002", "0.002"],
       ["Delays", "0.0003", "0.0008"],
       ["Schedule (time convenience)", "0.0006", "0.0003"]],
      [820, 260, 260], "lrr", bold={(0, 2), (1, 1)}, row_px=58)

# Section 4: trust table
table("table_trust",
      ["Decision", "Trust the synthetic copy?", "Evidence"],
      [["Train a satisfaction model", "Mostly. Costs 0.0013 to 0.0019 AUC", "Table 1, utility"],
       ["Choose a model or technique", "No. Rewards generator artefacts", "Tables 2 and 4"],
       ["Estimate real-world accuracy", "No. S→S 0.96 vs R→R 0.995", "Table 5"],
       ["Pick the service to fix first", "Only with what-if, not SHAP", "RQ3"],
       ["Plan around delays", "No. The generator collapsed them", "Distortion"]],
      [520, 640, 330], "lll", bold={(0, 1), (3, 1)}, row_px=72, fs=12)

# Datasets table
table("table_datasets",
      ["", "Synthetic (S)", "Real (R)"],
      [["Source", "Kaggle Playground S6E10", "Binary Aviation Satisfaction 129k"],
       ["Rows", "699,635", "129,880"],
       ["Satisfied", "44.4 %", "43.4 %"],
       ["Features", "21 (shared)", "21 (shared)"],
       ["Licence", "CC BY 4.0", "CC0"]],
      [200, 520, 560], "lll", bold=set(), row_px=70)
