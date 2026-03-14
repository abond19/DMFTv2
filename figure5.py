#!/usr/bin/env python3
"""
Reproduce Figure 5 from Montanari & Urbani (2025):
  "Third-regime scaling: collapse of a(t)/sqrt(m), v(t), and gen. gap."

Three panels:
  Left   — a(t)/sqrt(m) vs t/m  (log-x): curves collapse onto a master curve.
  Center — v(t)         vs t/m  (log-x): same collapse.
  Right  — e_ts - e_tr vs a(t)/sqrt(m) (parametric): finite-m curves approach
           the dashed "Lazy" reference (fixed second-layer weights).

Usage
-----
    python figures/figure5.py                       # default m = 2^3..2^7
    python figures/figure5.py --out my.png
    python figures/figure5.py --show
    python figures/figure5.py --m 8 16 32 64
    python figures/figure5.py --no-lazy             # skip lazy reference

Numerics
--------
  T_max = 9m per run (safe against float64 overflow for all m <= 128).
  Lazy reference: 15 runs with fixed a(t) = gamma*sqrt(m), T_max=2000.
  Total wall time: ~20-40 s for default m values on a modern laptop.
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")

from dmft.config import fig5_config, fig5_lazy_config
from dmft.solver import run_dmft
from dmft.observables import get_observables
from dmft.figures.fig5 import plot_fig5


DEFAULT_M_VALUES   = [8, 16, 32, 64]
GAMMA_STAR_GF      = 0.65          # approximate, read off from paper Fig 5
LAZY_GAMMA_VALUES  = np.linspace(0.05, 0.75, 16)


def parse_args():
    p = argparse.ArgumentParser(description="Reproduce Figure 5")
    p.add_argument("--out",     default="figure_5.png")
    p.add_argument("--show",    action="store_true")
    p.add_argument("--m",       nargs="+", type=int, default=DEFAULT_M_VALUES,
                   help="m values to run (e.g. --m 8 16 32 64)")
    p.add_argument("--no-lazy", dest="no_lazy", action="store_true",
                   help="Skip the lazy reference curve")
    return p.parse_args()


def run_all_m(m_values: list[int]) -> dict[int, dict]:
    """Run DMFT for each m and return {m: observables_dict}."""
    obs_dict = {}
    for m in m_values:
        cfg   = fig5_config(m)
        print(f"\n[m={m}]  T_max={cfg.solver.T_max:.0f}  N={cfg.solver.n_steps}")
        state = run_dmft(cfg, print_every=0, verbose=True)
        obs   = get_observables(state, cfg.cov, cfg.m, cfg.data.tau)
        obs_dict[m] = obs
    return obs_dict


def compute_lazy_reference(gamma_values: np.ndarray) -> dict:
    """
    For each gamma, run a lazy-init fixed-a(t) DMFT (m=8, T_max=100) and
    record the equilibrated generalisation gap e_ts - e_tr.
    Uses m=8 (fast, stable); the lazy curve is m-independent for large m.
    """
    print(f"\n[Lazy reference]  {len(gamma_values)} gamma points  (m=8, T_max=100)")
    gaps = []
    for gamma in gamma_values:
        cfg   = fig5_lazy_config(gamma0=gamma)
        state = run_dmft(cfg, print_every=0, verbose=False)
        obs   = get_observables(state, cfg.cov, cfg.m, cfg.data.tau)
        gap   = float(obs["test_error"][-1] - obs["train_error"][-1])
        gaps.append(gap)
        print(f"  gamma={gamma:.3f}  gap={gap:.4f}")

    return {"gamma": np.array(gamma_values), "gap": np.array(gaps)}


def main():
    args = parse_args()
    if args.show:
        matplotlib.use("TkAgg")

    obs_dict = run_all_m(args.m)

    lazy_ref = None
    if not args.no_lazy:
        lazy_ref = compute_lazy_reference(LAZY_GAMMA_VALUES)

    plot_fig5(
        obs_dict,
        lazy_reference=lazy_ref,
        savepath=args.out,
        show=args.show,
        gamma_star_gf=GAMMA_STAR_GF,
    )


if __name__ == "__main__":
    main()