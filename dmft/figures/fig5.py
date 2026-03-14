"""
Figure 5 plotting logic: "Third-regime scaling and collapse."

Three panels using a family of m values, all with the same rescaled axes:

  Left:   a(t)/√m  vs  t/m          (log-x, linear-y)
  Center: v(t)     vs  t/m          (log-x, linear-y)
  Right:  e_ts−e_tr  vs  a(t)/√m   (parametric; + "Lazy" reference)

Parameters match paper_config_2:
    h = φ̂ = (9/10)q + q³/6,  τ=0.3,  ᾱ=0.3.

Colour scheme: purple→teal gradient over m values (matching Fig 5 in paper).

References: Section 2.3, Eq 2.11–2.13; Figure 5 caption.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dmft.figures.style import apply_base_style, LW_MAIN


# ── Colour map: match paper's purple-to-teal gradient ────────────────────────

def _m_colors(m_values: list[int]) -> list:
    """Map m values to colours via a perceptually-uniform gradient."""
    cmap = cm.get_cmap("plasma_r")
    n    = len(m_values)
    return [cmap(0.10 + 0.70 * i / max(n - 1, 1)) for i in range(n)]


# ── Axis helpers ──────────────────────────────────────────────────────────────

def _trim_log(x: np.ndarray, y: np.ndarray,
              x_min: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Remove entries where x < x_min (for semilog axes)."""
    mask = x >= x_min
    return x[mask], y[mask]


def _thin(x: np.ndarray, y: np.ndarray,
          max_pts: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Downsample to at most max_pts points for clean rendering."""
    if len(x) <= max_pts:
        return x, y
    idx = np.round(np.linspace(0, len(x) - 1, max_pts)).astype(int)
    return x[idx], y[idx]


# ── Public API ────────────────────────────────────────────────────────────────

def plot_fig5(
    obs_dict:       dict[int, dict],
    lazy_reference: dict | None = None,
    savepath:       str | None = None,
    show:           bool = False,
    gamma_star_gf:  float | None = None,
) -> plt.Figure:
    """
    Produce Figure 5 from pre-computed observables for a family of m values.

    Args:
        obs_dict:       {m: obs}  where obs = output of get_observables().
        lazy_reference: Optional dict with keys "gamma" (array of γ=a/√m
                        values) and "gap" (array of e_ts−e_tr at equilibrium).
                        Plotted as a dashed black curve on the right panel.
        savepath:       If given, save the figure here.
        show:           If True, call plt.show().
        gamma_star_gf:  If given, draw a dashed red reference line + label
                        on the left panel at this γ*_GF value.

    Returns:
        The matplotlib Figure object.
    """
    apply_base_style()

    m_values = sorted(obs_dict.keys())
    colors   = _m_colors(m_values)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    ax_a, ax_v, ax_gap = axes

    for m, color in zip(m_values, colors):
        obs  = obs_dict[m]
        t    = obs["times"]
        a    = obs["a"]
        v    = obs["v"]
        e_tr = obs["train_error"]
        e_ts = obs["test_error"]
        sqm  = np.sqrt(m)
        gap  = e_ts - e_tr
        lbl  = rf"$m=2^{{{int(np.round(np.log2(m)))}}}$"
        lw   = LW_MAIN - 0.4

        tm   = t / m                   # rescaled time axis
        a_sc = a / sqm                 # rescaled second-layer weight

        # ── Left: a(t)/√m vs t/m ─────────────────────────────────────────────
        tx, ax_ = _thin(*_trim_log(tm, a_sc))
        ax_a.semilogx(tx, ax_, color=color, lw=lw, label=lbl)

        # ── Center: v(t) vs t/m ───────────────────────────────────────────────
        tx, vx = _thin(*_trim_log(tm, v))
        ax_v.semilogx(tx, vx, color=color, lw=lw)

        # ── Right: gap vs a(t)/√m (parametric) ───────────────────────────────
        # Skip the very first few steps where numerics are initialising
        mask = t >= 0.1
        ag, gg = _thin(a_sc[mask], gap[mask])
        ax_gap.plot(ag, gg, color=color, lw=lw, label=lbl)

    # ── γ*_GF reference line (left panel) ────────────────────────────────────
    if gamma_star_gf is not None:
        ax_a.axhline(gamma_star_gf, color="red", ls="--", lw=1.3, alpha=0.8)
        xlim_right = ax_a.get_xlim()[1] if ax_a.get_xlim()[1] > 1 else 100.0
        ax_a.annotate(
            r"$\gamma^*_{\rm GF}$",
            xy=(xlim_right * 0.3, gamma_star_gf),
            xytext=(xlim_right * 0.05, gamma_star_gf + 0.06),
            color="red", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="red", lw=1.0),
        )

    # ── Lazy reference (right panel) ─────────────────────────────────────────
    if lazy_reference is not None:
        g_arr   = np.asarray(lazy_reference["gamma"])
        gap_arr = np.asarray(lazy_reference["gap"])
        idx = np.argsort(g_arr)
        ax_gap.plot(g_arr[idx], gap_arr[idx],
                    color="black", ls="--", lw=2.0,
                    label="Lazy", zorder=5)

    # ── Axis formatting ───────────────────────────────────────────────────────
    _fmt_left(ax_a)
    _fmt_center(ax_v)
    _fmt_right(ax_gap)

    _legend_m(ax_a, m_values, colors, loc="upper left")
    _legend_m(ax_v, m_values, colors, loc="upper right")
    _legend_right(ax_gap, m_values, colors,
                  show_lazy=(lazy_reference is not None))

    fig.suptitle(
        r"Figure 5  $-$  Third-regime scaling"
        r"   ($\bar\alpha=0.3,\ \tau=0.3$,"
        r"  $h=\hat\varphi=\frac{9}{10}q+\frac{q^3}{6}$)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved → {savepath}")
    if show:
        plt.show()

    return fig


# ── Axis formatting helpers ───────────────────────────────────────────────────

def _fmt_left(ax):
    ax.set_xlabel(r"$t/m$",           fontsize=13)
    ax.set_ylabel(r"$a(t)/\sqrt{m}$", fontsize=13)
    ax.set_xlim(left=1e-3)
    ax.set_ylim(bottom=0, top=0.85)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.grid(True, which="both", ls="-", alpha=0.12)


def _fmt_center(ax):
    ax.set_xlabel(r"$t/m$", fontsize=13)
    ax.set_ylabel(r"$v(t)$", fontsize=13)
    ax.set_xlim(left=1e-3)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", ls="-", alpha=0.12)


def _fmt_right(ax):
    ax.set_xlabel(r"$a/\sqrt{m}$",              fontsize=13)
    ax.set_ylabel(r"$e_{\rm ts} - e_{\rm tr}$", fontsize=13)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, ls="-", alpha=0.12)


# ── Legend helpers ────────────────────────────────────────────────────────────

def _m_handles(m_values, colors):
    from matplotlib.lines import Line2D
    return [
        Line2D([0], [0], color=c, lw=1.8,
               label=rf"$m=2^{{{int(np.round(np.log2(m)))}}}$")
        for m, c in zip(m_values, colors)
    ]


def _legend_m(ax, m_values, colors, loc="upper left"):
    ax.legend(handles=_m_handles(m_values, colors),
              fontsize=7.5, loc=loc, framealpha=0.85,
              handlelength=1.5, labelspacing=0.3, ncol=1)


def _legend_right(ax, m_values, colors, show_lazy):
    from matplotlib.lines import Line2D
    handles = _m_handles(m_values, colors)
    if show_lazy:
        handles.append(
            Line2D([0], [0], color="black", ls="--", lw=1.8, label="Lazy")
        )
    ax.legend(handles=handles, fontsize=7.5,
              loc="upper left", framealpha=0.85,
              handlelength=1.5, labelspacing=0.3)