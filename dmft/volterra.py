"""
Incremental Volterra solver for R_A and C_A.

The Volterra equations (Eq C.19, discretised as Eqs D.3–D.4) read:

    [I + η Σ_R] R_A = (1/η) I
    [I + η Σ_R] C_A + η Σ_C R_A^T = 0

M = I + η Σ_R is unit lower-triangular (Σ_R is causal and has a zero
diagonal because Rd[n,n] = 0 — the δ boundary only sets Rd[n+1,n]=1).

Solving by forward substitution gives O(n) work per new row, for a total
cost of O(N²) over the full integration — vs O(N⁴) if M were re-inverted
from scratch at every step.

Note on the C_A formula (vs paper Eq D.6):
    D.6 writes C_A = −M⁻¹ Σ_C M⁻¹, implying M is symmetric.  The
    correct formula from D.4 is C_A = −M⁻¹ Σ_C (M^T)⁻¹.  Because M is
    lower-triangular (not symmetric) the two expressions differ; the
    forward-substitution here implements the exact form.

References: Section D.1, Eqs D.3–D.6, C.19.
"""

import numpy as np
from dmft.state import SymmDMFTState


def volterra_add_row(state: SymmDMFTState, n: int) -> None:
    """
    Extend R_A and C_A by one row/column at index n.

    After this call, state.R_A[n, :n+1] and state.C_A[n, :n+1] (and the
    symmetric C_A[:n+1, n]) are fully populated for the current step.

    Assumes state.Sigma_R[n, :n] and state.Sigma_C[n, :n+1] have already
    been filled by update_sigma_matrices().
    """
    eta = state.eta
    SR  = state.Sigma_R
    SC  = state.Sigma_C
    RA  = state.R_A
    CA  = state.C_A

    # ── R_A row n ────────────────────────────────────────────────────────────
    # M[n,n]=1  ⟹  R_A[n,n] = 1/η
    # M[n,j] = η·SR[n,j] for j<n  ⟹  R_A[n,j] = −η · Σ_{k≤j} SR[n,k]·R_A[k,j]
    RA[n, n] = 1.0 / eta
    if n > 0:
        RA[n, :n] = -eta * (SR[n, :n] @ RA[:n, :n])

    # ── C_A row n ─────────────────────────────────────────────────────────────
    # From [I + η SR] C_A = −η SC R_A^T, row n:
    #   C_A[n, s] = −η [ Σ_{k<n} SR[n,k]·CA[k,s]  +  Σ_k SC[n,k]·RA[s,k] ]
    if n == 0:
        CA[0, 0] = -SC[0, 0]   # = −η·SC[0,0]·RA[0,0] = −SC[0,0]
    else:
        # Off-diagonal entries  s < n
        t1 = SR[n, :n] @ CA[:n, :n]            # Σ_{k<n} SR[n,k]·CA[k,s]
        t2 = SC[n, :n+1] @ RA[:n, :n+1].T      # Σ_k SC[n,k]·RA[s,k]  for s<n
        CA[n, :n] = -eta * (t1 + t2)
        CA[:n, n] = CA[n, :n]                  # C_A is symmetric

        # Diagonal entry  s = n
        t3 = SR[n, :n] @ CA[:n, n]
        t4 = np.dot(SC[n, :n+1], RA[n, :n+1])
        CA[n, n] = -eta * (t3 + t4)