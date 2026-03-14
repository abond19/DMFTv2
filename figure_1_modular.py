#!/usr/bin/env python3
"""
Reproduce Figure 1 from Montanari & Urbani (2025):
  "Three dynamical regimes of learning in a two-layer neural network."

Usage
─────
    python figures/figure1.py               # saves figure_1.png
    python figures/figure1.py --show        # also opens an interactive window
    python figures/figure1.py --out my.png  # custom output path

Numerics note
─────────────
Uses m=64, η=0.5, T_max=580 — verified stable (overflow begins around T≈630).
Wall time ≈ 5 s on a modern laptop.
"""

import sys
import os
import argparse

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")   # headless by default; overridden below if --show

from dmft.config import fig1_config
from dmft.solver import run_dmft
from dmft.observables import get_observables
from dmft.figures.fig1 import plot_fig1


def parse_args():
    p = argparse.ArgumentParser(description="Reproduce Figure 1")
    p.add_argument("--out",  default="figure_1.png",
                   help="Output file path (default: figure_1.png)")
    p.add_argument("--show", action="store_true",
                   help="Open an interactive matplotlib window after saving")
    p.add_argument("--m",    type=int, default=64,
                   help="Number of neurons (default: 64)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.show:
        matplotlib.use("TkAgg")   # switch to interactive backend

    # ── Build config ──────────────────────────────────────────────────────────
    config = fig1_config(m=args.m)
    # fig1_config already sets safe defaults (eta=0.5, T_max=580 for m≤64).
    # For larger m, T_max is capped at 9*m inside fig1_config.
    print(f"Config: m={config.m}, alpha_bar={config.alpha_bar}, "
          f"tau={config.data.tau}, eta={config.solver.eta}, "
          f"T_max={config.solver.T_max}")

    # ── Run DMFT ──────────────────────────────────────────────────────────────
    state = run_dmft(config, print_every=300)

    # ── Compute observables ───────────────────────────────────────────────────
    obs = get_observables(state, config.cov, config.m, config.data.tau)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_fig1(
        obs,
        m=config.m,
        alpha_bar=config.alpha_bar,
        tau=config.data.tau,
        savepath=args.out,
        show=args.show,
    )


if __name__ == "__main__":
    main()