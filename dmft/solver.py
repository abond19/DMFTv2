"""
Integration driver for SymmDMFT.

Runs the Euler loop and handles the final Sigma + Volterra update
needed so that C_A[last, last] is populated for train-error computation.

Overflow safety note
────────────────────
For mean-field initialisation, a(t) grows without bound in the overfitting
regime.  At double precision, overflow occurs roughly when T_max ≳ 10·m for
η = 0.5.  The SolverConfig helper ``fig1_config`` caps T_max at 9·m.
If you set T_max manually, ensure T_max ≤ 9·m (with η ≤ 0.5) to avoid NaNs.

References: Appendix D.1, Section C.4.
"""

import time
import numpy as np

from dmft.config import ExperimentConfig
from dmft.state import SymmDMFTState, create_symm_state
from dmft.euler import euler_step, update_sigma
from dmft.volterra import volterra_add_row


def _finalise(state: SymmDMFTState, cov, m: int, tau: float) -> None:
    """
    Run one final Sigma + Volterra update for the last step so that
    C_A[last, last] is correctly populated for the train-error formula.
    """
    n   = state.current_step
    at  = state.a[n];  a_s = state.a[:n+1]
    v_s = state.v[:n+1]
    Cd_n = state.Cd[n, :n+1];  Co_n = state.Co[n, :n+1]
    Rd_n = state.Rd[n, :n+1];  Ro_n = state.Ro[n, :n+1]

    sc = (tau**2 + cov.phi_norm_sq
          - at * cov.phi_hat(state.v[n]) - a_s * cov.phi_hat(v_s)
          + (1.0/m) * at * a_s * cov.h(Cd_n)
          + ((m-1.0)/m) * at * a_s * cov.h(Co_n))
    state.Sigma_C[n, :n+1] = sc
    state.Sigma_C[:n+1, n] = sc

    sr = ((1.0/m) * at * a_s * cov.h_p(Cd_n) * Rd_n
          + ((m-1.0)/m) * at * a_s * cov.h_p(Co_n) * Ro_n)
    state.Sigma_R[n, :n+1] = sr

    volterra_add_row(state, n)


def run_dmft(
    config: ExperimentConfig,
    print_every: int = 200,
    verbose: bool = True,
) -> SymmDMFTState:
    """
    Run the full SymmDMFT Euler integration.

    Args:
        config:       Full experiment specification.
        print_every:  Print a progress line every this many steps (0 = silent).
        verbose:      Print header and footer timing lines.

    Returns:
        Fully-populated SymmDMFTState after integration.
    """
    m       = config.m
    alpha   = config.alpha          # = alpha_bar * m = n/d
    tau     = config.data.tau
    eta     = config.solver.eta
    n_steps = config.solver.n_steps
    cov     = config.cov

    if cov is None:
        raise ValueError("config.cov must be set before calling run_dmft()")

    state = create_symm_state(n_steps, eta, config.get_a0(), v0=0.0)

    if verbose:
        mem_gb = 12 * n_steps**2 * 8 / 1e9
        print(f"[DMFT] m={m}, alpha_bar={config.alpha_bar}, alpha={alpha:.2f}, "
              f"tau={tau}, eta={eta}, T={config.solver.T_max}, "
              f"N={n_steps}, mem≈{mem_gb:.2f} GB")

    t0 = time.time()

    for n in range(n_steps - 1):
        euler_step(
            n, state, cov, m, alpha, tau,
            evolve_second_layer=config.evolve_second_layer,
            is_pure_noise=config.data.is_pure_noise,
        )

        if print_every > 0 and (n + 1) % print_every == 0:
            e_tr = -0.5 * state.C_A[n+1, n+1]
            print(f"  step {n+1:5d}/{n_steps-1}  "
                  f"t={state.times[n+1]:8.2f}  "
                  f"a={state.a[n+1]:7.4f}  "
                  f"v={state.v[n+1]:7.4f}  "
                  f"e_tr={e_tr:.5f}  "
                  f"wall={time.time()-t0:.1f}s")

    _finalise(state, cov, m, tau)

    if verbose:
        print(f"Done. Wall time: {time.time()-t0:.1f}s")

    return state