"""Re-draw result figures from saved JSON (no retraining).

Run after prelim_experiments.py and prelim_te.py: python src/figures_results.py
"""
import json

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from airsat import plotstyle as ps
from airsat.config import FIG_DIR, RESULTS_DIR
from prelim_experiments import fig_shap


LADDER_SLOT_PX = 912  # width of the figure's slot on the slide


def ladder_points(prelim: dict, te: dict | None) -> pd.Series:
    ladder = prelim["E1_model_ladder"]
    points = {
        "Logistic regression": ladder["logreg"]["cv_auc"],
        "XGBoost, raw": ladder["xgb"]["cv_auc"],
        "LightGBM, raw": ladder["lgbm"]["cv_auc"],
        "LightGBM + real rows": prelim["E2_add_original"]["lgbm_plus_orig"]["cv_auc"],
    }
    if te is not None:
        points["XGBoost, 1,024 bins"] = te["xgb_raw_maxbin1024"]["cv_auc"]
        points["XGBoost + exact-value enc."] = te["xgb_te_count"]["cv_auc"]
    return pd.Series(points).sort_values()


def fig_ladder(points: pd.Series) -> None:
    """Dot plot (not bars: the axis does not start at zero) of 5-fold CV ROC AUC."""
    fig, ax = plt.subplots(figsize=ps.figsize(LADDER_SLOT_PX, 2.3))
    y = range(len(points))
    left = points.min() - 0.002
    ax.hlines(y, left, points.values, color=ps.GRID, linewidth=1.2)
    ax.scatter(points.values, y, s=90, color=ps.SERIES[0], edgecolor=ps.SURFACE, linewidth=2, zorder=3)
    for yi, v in zip(y, points.values):
        ax.text(v + 0.0005, yi, f"{v:.5f}", va="center", fontsize=12, color=ps.INK_2)
    ax.set_yticks(list(y), points.index)
    ax.set_xlim(left, points.max() + 0.0045)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("5-fold CV ROC AUC (n = 699,635)")
    ps.save(fig, FIG_DIR / "fig_ladder.png")


def main() -> None:
    ps.apply()
    prelim = json.loads((RESULTS_DIR / "prelim_results.json").read_text())
    te_path = RESULTS_DIR / "prelim_te.json"
    te = json.loads(te_path.read_text()) if te_path.exists() else None
    fig_ladder(ladder_points(prelim, te))
    fig_shap(pd.Series(prelim["E5_shap_mean_abs"]).sort_values(ascending=False))


if __name__ == "__main__":
    main()
