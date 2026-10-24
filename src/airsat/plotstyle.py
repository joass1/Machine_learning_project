"""Shared matplotlib style: validated categorical palette, hairline chrome, thin marks.

Legibility rule: every figure is sized from the pixel width it occupies on a 1920x1080 slide,
so that 12 pt text lands at about 24 px on the slide (the deck's minimum text size).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

# Open Sans = the slide deck's body face (assets/fonts, SIL OFL); falls back to Segoe UI if absent.
FONT_FILE = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "OpenSans.ttf"
if FONT_FILE.exists():
    font_manager.fontManager.addfont(str(FONT_FILE))

SURFACE = "#FFFFFF"  # matches the slide deck's white figure cards
INK = "#13202E"
INK_2 = "#3D4A57"
GRID = "#E3E3DF"
AXIS = "#B9B9B3"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DPI = 220
BASE_PT = 12
SLIDE_PX_PER_PT = 2.0  # 12 pt -> 24 px on the slide


def figsize(slide_px_width: int, aspect: float) -> tuple[float, float]:
    """Figure size in inches for a figure shown `slide_px_width` px wide on the slide."""
    width_in = slide_px_width / (72 * SLIDE_PX_PER_PT)
    return width_in, width_in / aspect


def apply() -> None:
    """Apply the house style globally."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Open Sans", "Segoe UI", "DejaVu Sans"], "font.size": BASE_PT,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": BASE_PT, "ytick.labelsize": BASE_PT, "axes.labelsize": BASE_PT,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": BASE_PT,
    })


def save(fig: plt.Figure, path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
