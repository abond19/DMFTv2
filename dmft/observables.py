"""
Observable extraction from a completed SymmDMFTState.

Train error  (Eq C.45):
    e_tr(t) = −½ C_A(t, t)

Test error  (Eq C.46):
    e_ts(t) = ½ [ τ² + ‖φ‖² − 2a(t)φ̂(v(t))
                  + (1/m)a(t)² h(1)
                  + ((m−1)/m)a(t)² h(C_o(t,t)) ]

The ℓ₁ norm proxy (DMFT analogue of ‖W₂nd‖₁, Fig 1 right axis):
    Under symmetric initialisation all second-layer weights are equal to
    a(t), so ‖W₂nd‖₁ = m·|a(t)|.  We expose the per-neuron value a(t)
    and let plotting code choose whether to multiply by m or not.

References: Eq C.45, Eq C.46.
"""

from __future__ import annotations
import numpy as np
from dmft.state import SymmDMFTState
from dmft.covariance import CovarianceFunctions


def train_error(state: SymmDMFTState) -> np.ndarray:
    """
    e_tr(t_n) = −½ C_A(t_n, t_n)   (Eq C.45).

    Returns array of shape (current_step + 1,).
    """
    N = state.current_step + 1
    return -0.5 * np.array([state.C_A[s, s] for s in range(N)])


def test_error(
    state: SymmDMFTState,
    cov: CovarianceFunctions,
    m: int,
    tau: float,
) -> np.ndarray:
    """
    e_ts(t_n)  from Eq C.46.

    Returns array of shape (current_step + 1,).
    """
    N     = state.current_step + 1
    a     = state.a[:N]
    v     = state.v[:N]
    Co_d  = np.array([state.Co[s, s] for s in range(N)])

    return 0.5 * (
        tau**2 + cov.phi_norm_sq
        - 2.0 * a * cov.phi_hat(v)
        + (1.0/m) * a**2 * cov.h(1.0)
        + ((m-1.0)/m) * a**2 * cov.h(Co_d)
    )


def get_observables(
    state: SymmDMFTState,
    cov: CovarianceFunctions,
    m: int,
    tau: float,
) -> dict:
    """
    Collect all time-series observables into a dict.

    Keys
    ----
    times    : physical time axis t_n = n·η
    a        : second-layer weight a(t)
    v        : latent direction overlap v(t)
    nu       : Lagrange multiplier ν(t)
    train_error : e_tr(t)
    test_error  : e_ts(t)
    Co_diag  : diagonal C_o(t,t)
    Cd_diag  : diagonal C_d(t,t) — should equal 1 by constraint
    """
    N    = state.current_step + 1
    e_tr = train_error(state)
    e_ts = test_error(state, cov, m, tau)

    return {
        "times":       state.times[:N].copy(),
        "a":           state.a[:N].copy(),
        "v":           state.v[:N].copy(),
        "nu":          state.nu[:N].copy(),
        "train_error": e_tr,
        "test_error":  e_ts,
        "Co_diag":     np.array([state.Co[s, s] for s in range(N)]),
        "Cd_diag":     np.array([state.Cd[s, s] for s in range(N)]),
    }