"""
State container for the SymmDMFT integration.

All 2-D arrays are (n_steps × n_steps); only entries with both indices
≤ current_step are meaningful at any given point in the integration.

Boundary conditions (Eq C.44):
    v(0) = 0,   C_d(0,0) = 1,   C_o(0,0) = 0.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SymmDMFTState:
    """Mutable state for one SymmDMFT run."""
    n_steps: int
    eta:     float
    times:   np.ndarray   # (n_steps,)  physical times t_n = n·η

    # Order parameters ── 2-D (n_steps, n_steps)
    Cd: np.ndarray   # C_d(t, t')  diagonal block (i=j)
    Co: np.ndarray   # C_o(t, t')  off-diagonal block (i≠j)
    Rd: np.ndarray   # R_d(t, t')
    Ro: np.ndarray   # R_o(t, t')

    # Order parameters ── 1-D (n_steps,)
    a:  np.ndarray   # second-layer weight a(t)
    v:  np.ndarray   # latent-direction projection v(t)
    nu: np.ndarray   # Lagrange multiplier ν(t)

    # Auxiliary ── 2-D
    Sigma_C: np.ndarray
    Sigma_R: np.ndarray
    R_A:     np.ndarray
    C_A:     np.ndarray

    # Memory kernels ── 2-D
    MR_d: np.ndarray   # M_R^{(d)}
    MR_o: np.ndarray   # M_R^{(o)}
    MC_d: np.ndarray   # M_C^{(d)}
    MC_o: np.ndarray   # M_C^{(o)}

    current_step: int = 0


def create_symm_state(n_steps: int, eta: float, a0: float,
                      v0: float = 0.0) -> SymmDMFTState:
    """
    Allocate and initialise a SymmDMFTState.

    Initial conditions (Eq C.44):
        C_d(0,0) = 1,  C_o(0,0) = 0,  v(0) = 0,  a(0) = a₀.
    All response functions start at zero (causal: R(0,0) = 0).
    """
    s2 = (n_steps, n_steps)
    z  = lambda: np.zeros(s2, dtype=np.float64)

    Cd = z();  Cd[0, 0] = 1.0          # C_d(0,0) = 1

    a_arr  = np.zeros(n_steps, dtype=np.float64);  a_arr[0]  = a0
    v_arr  = np.zeros(n_steps, dtype=np.float64);  v_arr[0]  = v0
    nu_arr = np.zeros(n_steps, dtype=np.float64)

    return SymmDMFTState(
        n_steps=n_steps,
        eta=eta,
        times=np.arange(n_steps, dtype=np.float64) * eta,
        Cd=Cd, Co=z(), Rd=z(), Ro=z(),
        a=a_arr, v=v_arr, nu=nu_arr,
        Sigma_C=z(), Sigma_R=z(), R_A=z(), C_A=z(),
        MR_d=z(), MR_o=z(), MC_d=z(), MC_o=z(),
        current_step=0,
    )