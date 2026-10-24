"""RQ3 figures (used by experiments/rq3_drivers.py), sized for their slot on a 1920x1080 slide.

Colour carries the dataset the model was trained on (synthetic = slot 1, real = slot 2) in every
figure; magnitudes use the single-hue sequential ramp.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from airsat import plotstyle as ps
from airsat.config import DISPLAY_NAMES
from airsat.drivers import SEGMENT_SEP

DATASET_COLORS = {"synthetic": ps.SERIES[0], "real": ps.SERIES[1]}
DATASET_TITLES = {"synthetic": "Synthetic-trained model", "real": "Real-trained model"}
CLASSES = ("Business", "Eco Plus", "Eco")
PURPOSES = {"Business travel": "business trip", "Personal Travel": "personal trip"}
SEGMENT_ORDER = [f"{c}{SEGMENT_SEP}{p}" for p in PURPOSES for c in CLASSES]
PALE = 0.35
WIDE_SLOT_PX, WHATIF_SLOT_PX = 1600, 1700
AGREEMENT_MARKERS = {  # comparison -> (marker, filled, vertical offset, legend label)
    "shap_own_data": ("o", True, 0.2, "SHAP: each model on its own passengers"),
    "shap_real_passengers": ("o", False, 0.0, "SHAP: both models on real passengers"),
    "whatif_real_passengers": ("D", True, -0.2, "What-if: both models on real passengers"),
}


def segment_display(label: str, sep: str = "\n") -> str:
    cabin, purpose = label.split(SEGMENT_SEP)
    return f"{cabin}{sep}{PURPOSES.get(purpose, purpose)}"


def _ordered_segments(present) -> list[str]:
    return [s for s in SEGMENT_ORDER if s in set(present)]


def _plain(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig_permutation(overlap: pd.DataFrame, path: Path) -> None:
    """Per dataset: AUC drop when a group is permuted jointly vs the sum of one-at-a-time drops."""
    ps.apply()
    order = overlap.groupby("group")["group_drop"].mean().sort_values().index
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(WIDE_SLOT_PX, 2.4), sharey=True, sharex=True)
    xmax = overlap[["group_drop", "sum_single_drop"]].to_numpy().max()
    for ax, dataset in zip(axes, ("synthetic", "real")):
        frame = overlap[overlap["dataset"] == dataset].set_index("group").loc[order]
        color = DATASET_COLORS[dataset]
        ax.barh(y + 0.2, frame["group_drop"], height=0.38, color=color)
        ax.barh(y - 0.2, frame["sum_single_drop"], height=0.38, color=color, alpha=PALE)
        for yi, value in zip(y, frame["group_drop"]):
            ax.text(value + xmax * 0.02, yi + 0.2, f"{value:.4f}", va="center", fontsize=11, color=ps.INK_2)
        ax.set_title(DATASET_TITLES[dataset], loc="left", fontsize=13, color=ps.INK)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("AUC drop on the holdout fold")
    axes[0].set_yticks(y, list(order))
    axes[0].set_xlim(0, xmax * 1.22)
    handles = [Patch(color=ps.INK_2), Patch(color=ps.INK_2, alpha=PALE)]
    fig.legend(handles, ["Whole group permuted together", "Sum of its features permuted one at a time"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    ps.save(fig, path)


def _segment_axis(ax: plt.Axes, segments: list[str]) -> None:
    """Cabin class as tick labels; trip purpose once under each block, blocks split by a gap."""
    cabins = [s.split(SEGMENT_SEP) for s in segments]
    ax.set_xticks(range(len(segments)), [c for c, _ in cabins], fontsize=11)
    for purpose, label in PURPOSES.items():
        cols = [i for i, (_, p) in enumerate(cabins) if p == purpose]
        if cols:
            ax.text(np.mean(cols), -0.1, label, transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=12, color=ps.INK)
            if cols[0] > 0:
                ax.axvline(cols[0] - 0.5, color=ps.SURFACE, linewidth=4)


def fig_shap_heatmap(shap: pd.DataFrame, path: Path) -> None:
    """Segment x service mean |SHAP| (colour) and within-segment rank (text), synthetic vs real."""
    ps.apply()
    views = {"syn_on_syn": "Synthetic model · synthetic passengers", "real_on_real": "Real model · real passengers"}
    pivot = {v: shap[shap["view"] == v].pivot(index="feature", columns="segment", values=["mean_abs_shap", "rank"])
             for v in views}
    segments = _ordered_segments(pivot["real_on_real"]["mean_abs_shap"].columns)
    rows = pivot["real_on_real"]["rank"].mean(axis=1).sort_values().index
    vmax = max(p["mean_abs_shap"].to_numpy().max() for p in pivot.values())
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(WIDE_SLOT_PX, 1.75), sharey=True)
    for ax, (view, title) in zip(axes, views.items()):
        values = pivot[view]["mean_abs_shap"].loc[rows, segments].to_numpy()
        ranks = pivot[view]["rank"].loc[rows, segments].to_numpy()
        image = ax.imshow(values, cmap=ps.SEQ_BLUE, vmin=0, vmax=vmax, aspect="auto")
        for (i, j), rank in np.ndenumerate(ranks):
            ax.text(j, i, f"{int(rank)}", ha="center", va="center", fontsize=11,
                    color="white" if values[i, j] > 0.55 * vmax else ps.INK,
                    fontweight="bold" if rank == 1 else "normal")
        _segment_axis(ax, segments)
        ax.set_title(title, loc="left", fontsize=13, color=ps.INK, pad=10)
        _plain(ax)
    axes[0].set_yticks(range(len(rows)), [DISPLAY_NAMES[r] for r in rows])
    bar = fig.colorbar(image, ax=list(axes), shrink=0.85, pad=0.02, fraction=0.03)
    bar.set_label("Mean |SHAP| (log-odds); cell text = rank within segment")
    bar.outline.set_visible(False)
    ps.save(fig, path)


def _fix_first_text(row: pd.Series, side: str) -> str:
    name = DISPLAY_NAMES.get(row[f"top1_{side}"], row[f"top1_{side}"])
    share = row.get(f"top1_share_{side}", np.nan)
    return name if pd.isna(share) else f"{name} ({share:.0%})"


def fig_agreement(agreement: pd.DataFrame, path: Path) -> None:
    """Spearman rho of synthetic vs real service rankings per segment + the two fix-first lists."""
    ps.apply()
    segments = _ordered_segments(agreement["segment"])[::-1]
    y = np.arange(len(segments))
    fig, (ax, table) = plt.subplots(1, 2, figsize=ps.figsize(WIDE_SLOT_PX, 2.5),
                                    gridspec_kw={"width_ratios": [1.0, 1.25]}, sharey=True)
    for comparison, (marker, filled, offset, label) in AGREEMENT_MARKERS.items():
        rho = agreement[agreement["comparison"] == comparison].set_index("segment")["spearman"]
        ax.scatter(rho.reindex(segments), y + offset, s=70, marker=marker, label=label, zorder=3,
                   linewidths=1.6, facecolors=ps.INK if filled else ps.SURFACE, edgecolors=ps.INK)
    ax.set_yticks(y, [segment_display(s, " · ") for s in segments])
    ax.set_xlim(min(-0.1, agreement["spearman"].min() - 0.1), 1.05)
    ax.set_xlabel("Spearman ρ, synthetic vs real ranking of 13 services")
    ax.legend(loc="upper center", bbox_to_anchor=(0.62, 1.32), ncol=1, fontsize=11)
    own = agreement[agreement["comparison"] == "shap_own_data"].set_index("segment").reindex(segments)
    for yi, (_, row) in zip(y, own.iterrows()):
        table.text(0.0, yi, _fix_first_text(row, "syn"), va="center", color=ps.INK)
        table.text(0.47, yi, _fix_first_text(row, "real"), va="center", color=ps.INK)
        table.text(0.93, yi, "same" if row["top1_match"] else "differs", va="center",
                   color=ps.INK if row["top1_match"] else ps.SERIES[7], fontweight="bold")
    for x, head in ((0.0, "Fix-first, synthetic"), (0.47, "Fix-first, real"), (0.93, "#1")):
        table.text(x, len(segments) - 0.35, head, color=ps.INK_2, fontsize=11)
    table.set_xlim(0, 1.05)
    _plain(table)
    table.tick_params(labelleft=False, labelbottom=False)
    fig.tight_layout()
    ps.save(fig, path)


def fig_whatif(whatif: pd.DataFrame, path: Path, top_n: int = 5) -> None:
    """Per segment: +1-rating uplift on real passengers for the real model's top services, both models."""
    ps.apply()
    real = whatif[whatif["passengers"] == "real"].assign(pp=lambda d: 100 * d["mean_uplift"])
    segments = _ordered_segments(real["segment"])
    fig, axes = plt.subplots(2, 3, figsize=ps.figsize(WHATIF_SLOT_PX, 1.55), sharex=True)
    for ax, segment in zip(axes.flat, segments):
        seg = real[real["segment"] == segment].pivot(index="service", columns="model", values="pp")
        seg = seg.sort_values("real", ascending=False).head(top_n)[::-1]
        y = np.arange(len(seg))
        for offset, model in ((0.2, "real"), (-0.2, "synthetic")):
            ax.barh(y + offset, seg[model], height=0.38, color=DATASET_COLORS[model])
        ax.set_yticks(y, [DISPLAY_NAMES[s] for s in seg.index], fontsize=11)
        ax.set_title(segment_display(segment, " · "), loc="left", fontsize=12, color=ps.INK)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=ps.AXIS, linewidth=0.8)
    for ax in axes.flat[len(segments):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Change in P(satisfied), pp")
    handles = [Line2D([], [], color=DATASET_COLORS[m], linewidth=8) for m in ("synthetic", "real")]
    fig.legend(handles, [DATASET_TITLES["synthetic"], DATASET_TITLES["real"]], loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    ps.save(fig, path)
