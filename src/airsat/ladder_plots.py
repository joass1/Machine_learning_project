"""RQ1 figures, sized for their slot on a 1920x1080 slide (see airsat.plotstyle).

Colour = training data, as in the RQ2 figures: synthetic-trained = SERIES[0], real-trained = SERIES[1].
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airsat import plotstyle as ps

SYN_COLOR, REAL_COLOR = ps.SERIES[0], ps.SERIES[1]
LADDER_SLOT_PX, TRICKS_SLOT_PX = 1500, 1300
MODEL_NAMES = {"logreg__onehot": "Logistic regression", "mlp__onehot": "MLP (sklearn)",
               "lgbm__raw": "LightGBM", "xgb__raw": "XGBoost", "xgb_bins__raw": "XGBoost, 1,024 bins",
               "xgb_ev__exact_value": "XGBoost + exact-value enc.",
               "catboost_cat__all_categorical": "CatBoost, all categorical", "realmlp__raw": "RealMLP",
               "tabm__raw": "TabM", "tabpfn__raw": "TabPFN"}


def model_name(key: str) -> str:
    return MODEL_NAMES.get(key, key)


def _dot_panel(ax: plt.Axes, table: pd.DataFrame, own: str, transfer: str, title: str) -> None:
    y = np.arange(len(table))
    ax.hlines(y, table[transfer], table[own], color=ps.GRID, linewidth=2, zorder=1)
    for col in (own, transfer):
        trained_on_real = col.startswith("R")  # settings read "<train>-><test>"
        ax.scatter(table[col], y, s=36, zorder=2, color=REAL_COLOR if trained_on_real else SYN_COLOR,
                   label="Trained on real" if trained_on_real else "Trained on synthetic")
    ax.set_yticks(y, [model_name(k) for k in table.index])
    ax.invert_yaxis()
    ax.set_title(title, loc="left", color=ps.INK)
    ax.set_xlabel("ROC AUC")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def plot_ladder(table: pd.DataFrame, path: Path) -> None:
    """Two panels: AUC on real passengers (R->R vs S->R) and on synthetic ones (S->S vs R->S)."""
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(LADDER_SLOT_PX, 2.4), sharey=True)
    _dot_panel(axes[0], table, "R->R", "S->R", "Scored on real passengers")
    _dot_panel(axes[1], table, "S->S", "R->S", "Scored on synthetic passengers")
    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.0))
    ps.save(fig, path)


def plot_tricks(deltas: pd.DataFrame, path: Path) -> None:
    """Paired AUC change of each technique over its baseline, on synthetic (S->S) vs real (S->R) passengers."""
    labels = list(dict.fromkeys(deltas["label"]))
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=ps.figsize(TRICKS_SLOT_PX, 2.0))
    for k, (setting, color, name) in enumerate((("S->S", SYN_COLOR, "Synthetic test rows (S→S)"),
                                                ("S->R", REAL_COLOR, "Real passengers (S→R)"))):
        sub = deltas[deltas["setting"] == setting].set_index("label").reindex(labels)
        offset = (k - 0.5) * 0.3
        ax.errorbar(sub["delta"], y + offset, xerr=[sub["delta"] - sub["low"], sub["high"] - sub["delta"]],
                    fmt="o", color=color, markersize=5, capsize=0, linewidth=1.2, label=name)
    ax.axvline(0, color=ps.AXIS, linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("AUC change over baseline (95% paired bootstrap)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right")
    ps.save(fig, path)
