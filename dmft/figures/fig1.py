"""
Figure 1 plotting logic: "Three dynamical regimes of learning."

Separates plotting from data generation so the same plot function
can be called with pre-computed observables (e.g. loaded from disk).

Paper Figure 1 parameters:
    h = φ̂ = (9/10)q + q³/6,  τ = 0.3,  ᾱ = 0.3,  m = 64 (or 128).
    Left axis:  train error (purple), test error (blue).
    Right axis: second-layer ℓ₁ norm proxy  a(t)  (red).

References: Figure 1 caption; Eq C.45–C.46; Appendix D.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dmft.figures.style import (
    COLORS, LW_MAIN, apply_base_style,
    add_regime_markers, make_legend,
)


def plot_fig1(
    obs: dict,
    m: int,
    alpha_bar: float = 0.3,
    tau: float = 0.3,
    savepath: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Produce Figure 1 from a pre-computed observables dict.

    Args:
        obs:        Output of ``dmft.observables.get_observables()``.
        m:          Number of neurons (for axis labels and regime boundaries).
        alpha_bar:  ᾱ = n/(md).
        tau:        Noise level.
        savepath:   If given, save the figure to this path.
        show:       If True, call plt.show().

    Returns:
        The matplotlib Figure object.
    """
    apply_base_style()

    t    = obs["times"]
    e_tr = obs["train_error"]
    e_ts = obs["test_error"]
    a    = obs["a"]

    # Trim t = 0 (undefined on log axis)
    mask = t >= 0.3
    t, e_tr, e_ts, a = t[mask], e_tr[mask], e_ts[mask], a[mask]

    T_MAX = float(t[-1])

    # ── Layout ────────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(7.2, 5.2))
    ax2 = ax1.twinx()

    ax1.plot(t, e_ts, color=COLORS["test"],  lw=LW_MAIN, zorder=3)
    ax1.plot(t, e_tr, color=COLORS["train"], lw=LW_MAIN, zorder=3)
    ax2.plot(t, a,    color=COLORS["norm"],  lw=LW_MAIN)

    # ── Axis limits ───────────────────────────────────────────────────────────
    ax1.set_xscale("log")
    ax1.set_xlim(0.45, T_MAX)
    ax1.set_ylim(-0.02, 0.72)
    ax2.set_ylim(0.0, 5.0)
    ax2.set_yticks([0, 1, 2, 3, 4, 5])
    ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    # ── Axis labels ───────────────────────────────────────────────────────────
    ax1.set_xlabel(r"$t$", fontsize=16)
    ax1.set_ylabel("Train/Test error", fontsize=13)
    ax2.set_ylabel(r"2nd layer $\ell_1$-norm", color=COLORS["norm"], fontsize=13)
    ax2.tick_params(axis="y", labelcolor=COLORS["norm"])

    # ── Regime markers ────────────────────────────────────────────────────────
    add_regime_markers(ax1, t_mf=1.0, t_of=float(m), t_end=T_MAX)

    # ── Complexity annotations on right axis ──────────────────────────────────
    ax2.text(0.52, 1.10,
             r"$\|W_{\rm 2nd}\|_1 \approx 1$",
             fontsize=8.5, color=COLORS["norm"], ha="left", va="bottom")

    ax2.annotate(
        r"$\|W_{\rm 2nd}\|_1 \approx \sqrt{m}$",
        xy=(t[-1], a[-1]),
        xytext=(float(m) * 0.9, 4.0),
        fontsize=8.5, color=COLORS["norm"],
        arrowprops=dict(arrowstyle="->", color=COLORS["norm"], lw=0.9),
    )

    # ── "generaliz. error" annotation ─────────────────────────────────────────
    gen     = e_ts - e_tr
    of_mask = t > float(m)
    if of_mask.any():
        gi   = np.where(of_mask)[0][np.argmax(gen[of_mask])]
        ymid = 0.5 * (e_ts[gi] + e_tr[gi])
        ax1.annotate(
            "generaliz.\nerror",
            xy=(t[gi], ymid),
            xytext=(t[gi] * 0.07, 0.13),
            fontsize=8.5, ha="center",
            arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    make_legend(ax1, [
        (COLORS["test"],  LW_MAIN, "Test error"),
        (COLORS["train"], LW_MAIN, "Train error"),
        (COLORS["norm"],  LW_MAIN, r"$\|W_{\rm 2nd}\|_1$"),
    ])

    ax1.grid(True, which="both", ls="-", alpha=0.14)
    ax1.set_title(
        r"Figure 1  —  Three dynamical regimes  "
        r"($m={:d},\;\alpha={:.1f},\;\tau={:.1f}$)".format(m, alpha_bar, tau),
        fontsize=11, pad=8,
    )

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved → {savepath}")

    if show:
        plt.show()

    return fig