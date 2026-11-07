"""Shared matplotlib style: one validated categorical palette, thin marks, recessive axes.

Palette = dataviz reference instance (light mode). Charts use <= 3 series so the
first three slots (validated all-pairs for CVD) are the only hues in play.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sic import config

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#a3a29c"
SERIES = [BLUE, ORANGE, AQUA]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "legend.labelcolor": TEXT,
    "font.size": 10,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def save(fig, name: str) -> str:
    out = config.path("figures") / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out.relative_to(config.ROOT) if out.is_relative_to(config.ROOT) else out)


def eur(x, _=None) -> str:
    if abs(x) >= 1e6:
        return f"€{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"€{x / 1e3:.0f}k"
    return f"€{x:.0f}"
