"""
Shared matplotlib style constants and helpers for all DMFT figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Colour palette ────────────────────────────────────────────────────────────
# Matches the paper's blue/purple/red scheme used in Figs 1–5.
COLORS = {
    "test":   "#2563b8",   # blue
    "train":  "#7b35b0",   # purple
    "norm":   "#c0392b",   # red  (ℓ₁ norm / a(t))
    "regime": "#333333",   # dotted vertical lines
    "grid":   "#000000",
}

# Colour cycle for multi-m figures (Figs 4, 5):  blue → green
M_COLORS = [
    "#3f007d", "#54278f", "#6a51a3", "#807dba",
    "#9e9ac8", "#bcbddc", "#dadaeb",
]

# Line-width defaults
LW_MAIN   = 2.3
LW_REGIME = 1.4


def apply_base_style():
    """Apply a clean, paper-like rcParams."""
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         11,
        "axes.linewidth":    0.8,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.size":  4,
        "ytick.major.size":  4,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "0.7",
    })


def add_regime_markers(
    ax,
    t_mf: float,
    t_of: float,
    t_end: float,
    y_arrow: float = 0.70,
    y_label_pad: float = 0.012,
    fontsize: float = 8.5,
):
    """
    Draw <-> arrows and dotted lines for the three learning regimes.

    Args:
        ax:          Left (error) axes.
        t_mf:        t_mf boundary (end of MF feature-learning).
        t_of:        t_of boundary (onset of overfitting).
        t_end:       Right edge of the plot (T_max).
        y_arrow:     y-position of the arrows (data coordinates).
        y_label_pad: Vertical offset above arrow for text.
        fontsize:    Regime-label font size.
    """
    T0  = ax.get_xlim()[0]
    ARR = dict(arrowstyle="<->", color="black", lw=1.0, mutation_scale=12)

    ax.axvline(t_mf, color=COLORS["regime"], ls=":", lw=LW_REGIME)
    ax.axvline(t_of, color=COLORS["regime"], ls=":", lw=LW_REGIME)

    right_end = t_end * 0.87
    ax.annotate("", xy=(t_mf,      y_arrow), xytext=(T0,   y_arrow), arrowprops=ARR)
    ax.annotate("", xy=(t_of,      y_arrow), xytext=(t_mf, y_arrow), arrowprops=ARR)
    ax.annotate("", xy=(right_end, y_arrow), xytext=(t_of, y_arrow), arrowprops=ARR)

    kw = dict(ha="center", va="bottom", fontsize=fontsize)
    ax.text(np.sqrt(T0 * t_mf),        y_arrow + y_label_pad, "MF Feat.\nlearning", **kw)
    ax.text(np.sqrt(t_mf * t_of),      y_arrow + y_label_pad, "Feat. learning",     **kw)
    ax.text(np.sqrt(t_of * right_end), y_arrow + y_label_pad, "Overfit/Unlearn",    **kw)

    # Small time-scale labels just below the x-axis tick marks
    ax.text(t_mf, ax.get_ylim()[0] - 0.017,
            r"$t_{\rm mf}(m){\approx}1$",
            ha="center", va="top", fontsize=7.5, color=COLORS["regime"])
    ax.text(t_of, ax.get_ylim()[0] - 0.017,
            r"$t_{\rm of}(m){\approx}m$",
            ha="center", va="top", fontsize=7.5, color=COLORS["regime"])


def make_legend(ax, entries: list[tuple], **legend_kw):
    """
    Build a custom legend from a list of (color, lw, label) tuples.

    Args:
        ax:       Axes to attach the legend to.
        entries:  List of (color, linewidth, label_string).
        **legend_kw: Passed to ax.legend().
    """
    handles = [
        Line2D([0], [0], color=c, lw=lw, label=lbl)
        for c, lw, lbl in entries
    ]
    legend_kw.setdefault("fontsize", 10)
    legend_kw.setdefault("loc", "center right")
    ax.legend(handles=handles, **legend_kw)