"""RQ2 figures, sized for their slot on a 1920x1080 slide (see airsat.plotstyle).

Colour identity is fixed across every RQ2 figure: synthetic = SERIES[0] (blue), real = SERIES[1] (orange).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airsat import plotstyle as ps
from airsat.config import DISPLAY_NAMES

SYN_COLOR, REAL_COLOR = ps.SERIES[0], ps.SERIES[1]
DISTORTION_SLOT_PX, DELAY_SLOT_PX, UTILITY_SLOT_PX, DEPENDENCE_SLOT_PX = 900, 1500, 960, 900
DELAY_BAND_NAMES = {"0": "On time", "1-15": "1–15", "16-60": "16–60", "61-180": "61–180", ">180": ">180"}


def display(col: str) -> str:
    return DISPLAY_NAMES.get(col, col)


def _hide_y_grid(ax: plt.Axes) -> None:
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def plot_distortion(distortion: pd.DataFrame, path: Path, top: int = 12) -> None:
    """Horizontal bars of the most distorted features (KS for numerics, TVD for discrete columns)."""
    rows = distortion.sort_values("distortion", ascending=False).head(top)[::-1]
    labels = [f"{display(f)} ({m})" for f, m in zip(rows["feature"], rows["metric"])]
    fig, ax = plt.subplots(figsize=ps.figsize(DISTORTION_SLOT_PX, 1.25))
    ax.barh(labels, rows["distortion"], height=0.62, color=SYN_COLOR)
    vmax = float(rows["distortion"].max()) or 1.0
    for yi, v in enumerate(rows["distortion"]):
        ax.text(v + vmax * 0.015, yi, f"{v:.3f}", va="center", color=ps.INK_2)
    ax.set_xlim(0, vmax * 1.2)
    _hide_y_grid(ax)
    ax.set_xlabel("Synthetic vs real distance (0 = identical)")
    ps.save(fig, path)


def plot_delays(bins: pd.DataFrame, path: Path) -> None:
    """Share of passengers per delay band, synthetic vs real, one panel per delay column."""
    features = list(dict.fromkeys(bins["feature"]))
    fig, axes = plt.subplots(1, len(features), figsize=ps.figsize(DELAY_SLOT_PX, 2.6), sharey=True)
    for ax, feature in zip(np.atleast_1d(axes), features):
        sub = bins[bins["feature"] == feature]
        bands = list(dict.fromkeys(sub["band"]))
        x, width = np.arange(len(bands)), 0.38
        for k, (name, color) in enumerate((("synthetic", SYN_COLOR), ("real", REAL_COLOR))):
            shares = sub[sub["dataset"] == name].set_index("band").loc[bands, "share"].to_numpy()
            ax.bar(x + (k - 0.5) * (width + 0.03), shares, width=width, color=color, label=name.capitalize())
            ax.text(x[0] + (k - 0.5) * (width + 0.03), shares[0] + 0.02, f"{shares[0]:.0%}", ha="center",
                    color=ps.INK_2)
        ax.set_xticks(x, [DELAY_BAND_NAMES.get(b, b) for b in bands])
        ax.set_title(display(feature), loc="left", color=ps.INK)
        ax.set_xlabel("Delay (minutes)")
        ax.grid(axis="x", visible=False)
    first = np.atleast_1d(axes)[0]
    first.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    first.set_ylabel("Share of passengers")
    first.set_ylim(0, 1.05)
    first.legend(loc="upper right")
    ps.save(fig, path)


def _size_label(n: float) -> str:
    return f"{n / 1000:.0f}k" if n >= 1000 else f"{n:.0f}"


def plot_utility(summary: pd.DataFrame, tstr: tuple[float, float, float], equivalent_n: float | None,
                 n_synthetic: int, path: Path) -> None:
    """Real-data learning curve (mean ± sd over seeds) against the TSTR AUC line with its 95% CI."""
    estimate, low, high = tstr
    fig, ax = plt.subplots(figsize=ps.figsize(UTILITY_SLOT_PX, 1.5))
    ax.axhspan(low, high, color=SYN_COLOR, alpha=0.12, linewidth=0)
    ax.axhline(estimate, color=SYN_COLOR, linewidth=2, linestyle="--",
               label=f"Trained on all {n_synthetic:,} synthetic rows")
    ax.errorbar(summary["n"], summary["auc_mean"], yerr=summary["auc_std"].fillna(0), color=REAL_COLOR,
                linewidth=2, marker="o", markersize=6, capsize=3, label="Trained on n real rows")
    if equivalent_n is not None:
        ax.axvline(equivalent_n, color=ps.INK_2, linewidth=1, linestyle=":")
        ax.annotate(f"≈ {_size_label(equivalent_n)} real rows", (equivalent_n, estimate),
                    xytext=(8, -28), textcoords="offset points", color=ps.INK)
    ax.set_xscale("log")
    ax.set_xticks(summary["n"], [_size_label(n) for n in summary["n"]])
    ax.minorticks_off()
    ax.set_xlabel("Real training rows (log scale)")
    ax.set_ylabel("AUC on held-out real passengers")
    ax.legend(loc="lower right")
    ps.save(fig, path)


def plot_target_dependence(dependence: pd.DataFrame, path: Path) -> None:
    """Dumbbell: univariate AUC of each feature for satisfaction, synthetic vs real."""
    rows = dependence.sort_values("real_auc")
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=ps.figsize(DEPENDENCE_SLOT_PX, 0.95))
    ax.hlines(y, rows["syn_auc"], rows["real_auc"], color=ps.AXIS, linewidth=2)
    ax.scatter(rows["syn_auc"], y, s=60, color=SYN_COLOR, label="Synthetic", zorder=3,
               edgecolors=ps.SURFACE, linewidths=1.5)
    ax.scatter(rows["real_auc"], y, s=60, color=REAL_COLOR, label="Real", zorder=3,
               edgecolors=ps.SURFACE, linewidths=1.5)
    ax.set_yticks(y, [display(f) for f in rows["feature"]])
    _hide_y_grid(ax)
    ax.set_xlabel("Univariate AUC for satisfaction (0.5 = no signal)")
    ax.legend(loc="lower right")
    ps.save(fig, path)
