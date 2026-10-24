"""Exploratory data analysis: summary stats (JSON) + publication figures (PNG).

Figures are sized for their slot on the slides (see plotstyle.figsize).
Run: python src/eda.py
"""
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airsat import plotstyle as ps
from airsat.config import (CATEGORICAL, DISPLAY_NAMES, FIG_DIR, NUMERIC, RESULTS_DIR, SERVICE_RATINGS,
                           TARGET)
from airsat.data import load_real as load_original
from airsat.data import load_synthetic as load_competition

log = logging.getLogger("eda")
DELAY_THRESHOLDS = (0, 15, 60)
HEATMAP_SLOT_PX, SEGMENTS_SLOT_PX, DELAY_SLOT_PX = 692, 768, 600


def summary_stats(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame) -> dict:
    def numeric_summary(df: pd.DataFrame) -> dict:
        return {c: {"mean": round(df[c].mean(), 2), "median": float(df[c].median()),
                    "p75": float(df[c].quantile(0.75)), "max": float(df[c].max())} for c in NUMERIC}

    return {
        "rows": {"train": len(train), "test": len(test), "original": len(orig)},
        "positive_rate": {"train": round(train[TARGET].mean(), 4), "original": round(orig[TARGET].mean(), 4)},
        "missing_arrival_delay": {"train": int(train["Arrival Delay in Minutes"].isna().sum()),
                                  "test": int(test["Arrival Delay in Minutes"].isna().sum()),
                                  "original": int(orig["Arrival Delay in Minutes"].isna().sum())},
        "categorical_counts": {c: train[c].value_counts().to_dict() for c in CATEGORICAL},
        "numeric": {"train": numeric_summary(train), "original": numeric_summary(orig)},
        "share_delayed_over": {
            name: {f">{t} min": round((df["Departure Delay in Minutes"] > t).mean(), 4) for t in DELAY_THRESHOLDS}
            for name, df in (("synthetic", train), ("real", orig))
        },
    }


def fig_rating_heatmap(train: pd.DataFrame) -> pd.Series:
    """Satisfaction rate for each service rating level; rows ordered by spread (max-min)."""
    rates = pd.DataFrame({c: train.groupby(c)[TARGET].mean() for c in SERVICE_RATINGS}).T
    spread = rates.max(axis=1) - rates.min(axis=1)
    rates = rates.loc[spread.sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=ps.figsize(HEATMAP_SLOT_PX, 1.38))
    ax.imshow(rates.values, cmap=ps.SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(rates.shape[1]), [str(int(c)) for c in rates.columns])
    ax.set_yticks(range(len(rates)), [DISPLAY_NAMES[c] for c in rates.index])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (i, j), v in np.ndenumerate(rates.values):
        label = "–" if np.isnan(v) else f"{v:.0%}"
        color = "white" if (not np.isnan(v) and v > 0.55) else ps.INK
        ax.text(j, i, label, ha="center", va="center", fontsize=11, color=color)
    ax.set_xlabel("Rating (0 = not applicable, 1 = worst, 5 = best)")
    ps.save(fig, FIG_DIR / "fig_rating_heatmap.png")
    return spread.round(3)


def fig_segments(train: pd.DataFrame) -> dict:
    """Satisfaction rate by cabin class, split by travel purpose (two series)."""
    rates = train.groupby(["Class", "Type of Travel"])[TARGET].mean().unstack()
    rates = rates.loc[["Business", "Eco Plus", "Eco"]]
    fig, ax = plt.subplots(figsize=ps.figsize(SEGMENTS_SLOT_PX, 2.6))
    y = np.arange(len(rates))
    height = 0.36
    labels = {"Business travel": "Business trip", "Personal Travel": "Leisure trip"}
    for k, col in enumerate(rates.columns):
        offs = (k - 0.5) * (height + 0.03)
        ax.barh(y + offs, rates[col], height=height, color=ps.SERIES[k], label=labels[col])
        for yi, v in zip(y + offs, rates[col]):
            ax.text(v + 0.015, yi, f"{v:.0%}", va="center", fontsize=12, color=ps.INK_2)
    ax.set_yticks(y, ["Business", "Eco Plus", "Economy"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of passengers satisfied")
    ax.legend(loc="lower right", ncols=2, bbox_to_anchor=(1, 1.0))
    ps.save(fig, FIG_DIR / "fig_segments.png")
    return {f"{i}|{j}": round(v, 4) for (i, j), v in rates.stack().items()}


def fig_delay_shift(stats: dict) -> None:
    """Share of flights delayed beyond thresholds: synthetic training data vs real survey."""
    shares = pd.DataFrame(stats["share_delayed_over"])
    fig, ax = plt.subplots(figsize=ps.figsize(DELAY_SLOT_PX, 2.05))
    x = np.arange(len(shares))
    width = 0.36
    for k, col in enumerate(["real", "synthetic"]):
        label = "Real survey" if col == "real" else "Synthetic data"
        bars = ax.bar(x + (k - 0.5) * (width + 0.03), shares[col], width=width, color=ps.SERIES[k], label=label)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in shares[col]], padding=2, fontsize=12, color=ps.INK_2)
    ax.set_xticks(x, [f"> {t} min" for t in DELAY_THRESHOLDS])
    ax.set_ylim(0, 0.52)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Departure delay")
    ax.set_ylabel("Share of passengers")
    ax.legend(loc="upper right")
    ps.save(fig, FIG_DIR / "fig_delay_shift.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ps.apply()
    train, test = load_competition()
    orig = load_original()
    stats = summary_stats(train, test, orig)
    stats["rating_spread"] = fig_rating_heatmap(train).to_dict()
    stats["segment_rates"] = fig_segments(train)
    fig_delay_shift(stats)
    (RESULTS_DIR / "eda_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    log.info("saved figures to %s", FIG_DIR)


if __name__ == "__main__":
    main()
