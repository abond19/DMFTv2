"""
One complete, vectorised Euler step for the SymmDMFT equations.

Implements Eqs C.32–C.43 (Section C.4) discretised per Appendix D.1.
All loops over the 'second time index' t' (= tb) are replaced by
matrix–vector products for O(n²) cost per step.

Two correctness notes (bugs fixed vs earlier drafts):

  BUG 1 — R_d/R_o transpose in C_d / C_o equations (C.34–C.35):
    The term  ∫₀^{t'} M_C^d(t,s) R_d(t',s) ds  vectorised over t' gives
        (Rd_matrix @ MCd_vec)[t'] = Σ_s Rd[t',s]·MCd[s]
    NOT  (Rd_matrix.T @ MCd_vec),  because R_d(t',s) ≠ R_d(s,t')
    (R_d is lower-triangular, NOT symmetric).

  BUG 2 — Rd diagonal must stay 0:
    The δ(t−t') boundary sets only the sub-diagonal:  Rd[n+1, n] = 1.
    Setting the diagonal Rd[n+1, n+1] = 1 as well would make
    Σ_R[n+1, n+1] ≠ 0, breaking the lower-triangular structure required
    by the Volterra solver and corrupting every subsequent C_A value.

References: Section C.4, Eqs C.32–C.43; Appendix D.1.
"""

import numpy as np
from dmft.state import SymmDMFTState
from dmft.covariance import CovarianceFunctions
from dmft.volterra import volterra_add_row


def update_sigma(n: int, state: SymmDMFTState,
                 cov: CovarianceFunctions, m: int, tau: float) -> None:
    """
    Fill Sigma_C[n, :n+1], Sigma_C[:n+1, n] and Sigma_R[n, :n+1]
    for the current step n.  (Eq C.42)

    Note: Sigma_R[n, n] = 0 because Rd[n,n] = 0 by the δ boundary rule.
    """
    at, vt = state.a[n], state.v[n]
    a_s    = state.a[:n+1]
    v_s    = state.v[:n+1]
    Cd_n   = state.Cd[n, :n+1]
    Co_n   = state.Co[n, :n+1]
    Rd_n   = state.Rd[n, :n+1]
    Ro_n   = state.Ro[n, :n+1]

    sc = (tau**2 + cov.phi_norm_sq
          - at * cov.phi_hat(vt) - a_s * cov.phi_hat(v_s)
          + (1.0/m) * at * a_s * cov.h(Cd_n)
          + ((m-1.0)/m) * at * a_s * cov.h(Co_n))
    state.Sigma_C[n, :n+1] = sc
    state.Sigma_C[:n+1, n] = sc

    sr = ((1.0/m) * at * a_s * cov.h_p(Cd_n) * Rd_n
          + ((m-1.0)/m) * at * a_s * cov.h_p(Co_n) * Ro_n)
    state.Sigma_R[n, :n+1] = sr
    # Sigma_R[:n, n] stays 0 (causal)


def update_memory_kernels(n: int, state: SymmDMFTState,
                          cov: CovarianceFunctions,
                          alpha: float, m: int) -> None:
    """
    Fill MR_d, MR_o, MC_d, MC_o at row n.  (Eqs C.38–C.41)

    MC_d and MC_o are symmetric in (t, s), so we mirror them.
    """
    pf  = alpha / m
    at  = state.a[n]
    a_s = state.a[:n+1]
    aa  = pf * at * a_s

    Cd_n  = state.Cd[n, :n+1];  Co_n  = state.Co[n, :n+1]
    Rd_n  = state.Rd[n, :n+1];  Ro_n  = state.Ro[n, :n+1]
    RA_n  = state.R_A[n, :n+1]; CA_n  = state.C_A[n, :n+1]

    state.MR_d[n, :n+1] = aa * (RA_n*cov.h_p(Cd_n) + CA_n*cov.h_pp(Cd_n)*Rd_n)
    state.MR_o[n, :n+1] = aa * (RA_n*cov.h_p(Co_n) + CA_n*cov.h_pp(Co_n)*Ro_n)
    state.MC_d[n, :n+1] = aa * CA_n * cov.h_p(Cd_n)
    state.MC_o[n, :n+1] = aa * CA_n * cov.h_p(Co_n)

    # Mirror MC (symmetric in t,s)
    state.MC_d[:n+1, n] = state.MC_d[n, :n+1]
    state.MC_o[:n+1, n] = state.MC_o[n, :n+1]


def euler_step(n: int, state: SymmDMFTState,
               cov: CovarianceFunctions,
               m: int, alpha: float, tau: float,
               evolve_second_layer: bool = True,
               is_pure_noise: bool = False) -> None:
    """
    Advance all SymmDMFT quantities from step n to step n+1.

    Order of operations (following D.1):
      1. Sigma_C, Sigma_R  (Eq C.42)
      2. Volterra solve    (Eqs D.3–D.6)  → R_A[n,:], C_A[n,:]
      3. Memory kernels    (Eqs C.38–C.41)
      4. nu[n]             (Eq C.43)
      5. a[n+1]            (Eq C.32)
      6. v[n+1]            (Eq C.33)
      7. Cd, Co            (Eqs C.34–C.35)  — vectorised over t'
      8. Rd, Ro            (Eqs C.36–C.37)  — vectorised over t'
    """
    eta = state.eta
    nn  = n + 1     # length of active slices

    at, vt = state.a[n], state.v[n]
    a_s    = state.a[:nn]
    Cd_n   = state.Cd[n, :nn];   Co_n = state.Co[n, :nn]
    Rd_n   = state.Rd[n, :nn];   Ro_n = state.Ro[n, :nn]

    # ── 1. Sigma ──────────────────────────────────────────────────────────────
    update_sigma(n, state, cov, m, tau)

    # ── 2. Volterra ───────────────────────────────────────────────────────────
    volterra_add_row(state, n)
    RA_n = state.R_A[n, :nn]
    CA_n = state.C_A[n, :nn]

    # ── 3. Memory kernels ─────────────────────────────────────────────────────
    update_memory_kernels(n, state, cov, alpha, m)
    MRd = state.MR_d[n, :nn]
    MRo = state.MR_o[n, :nn]
    MCd = state.MC_d[n, :nn]
    MCo = state.MC_o[n, :nn]

    hCd = cov.h(Cd_n);   hCo = cov.h(Co_n)
    hpCd= cov.h_p(Cd_n); hpCo= cov.h_p(Co_n)

    gr_vt  = cov.phi_hat_grad(vt)
    ph_vt  = cov.phi_hat(vt)
    RA_sum = np.sum(RA_n) * eta

    # ── 4. nu(t_n)  (Eq C.43) ────────────────────────────────────────────────
    nu_n = ((alpha/m)*gr_vt*vt*at*RA_sum
            - (1.0/m)*eta*(np.dot(MRd, Cd_n) + (m-1)*np.dot(MRo, Co_n))
            - (1.0/m)*eta*(np.dot(MCd, Rd_n) + (m-1)*np.dot(MCo, Ro_n)))
    state.nu[n] = nu_n

    # ── 5. a(t_{n+1})  (Eq C.32) ─────────────────────────────────────────────
    if evolve_second_layer:
        t1 = (alpha/m)*ph_vt*RA_sum
        t2 = -(alpha/m)*eta*np.dot(RA_n*a_s,
               (1.0/m)*hCd + ((m-1.0)/m)*hCo)
        t3 = -(alpha/m)*eta*np.dot(CA_n*a_s,
               (1.0/m)*hpCd*Rd_n + ((m-1.0)/m)*hpCo*Ro_n)
        state.a[n+1] = at + eta*(t1 + t2 + t3)
    else:
        state.a[n+1] = at

    # ── 6. v(t_{n+1})  (Eq C.33) ─────────────────────────────────────────────
    if not is_pure_noise:
        MRv = eta * np.dot(MRd + (m-1)*MRo, state.v[:nn])
        state.v[n+1] = vt + eta*(-nu_n*vt
                                 + (alpha/m)*gr_vt*at*RA_sum
                                 - (1.0/m)*MRv)
    else:
        state.v[n+1] = 0.0

    # ── 7. C_d, C_o  (Eqs C.34–C.35) — vectorised over t' ────────────────────
    Cd_m = state.Cd[:nn, :nn]
    Co_m = state.Co[:nn, :nn]
    Rd_m = state.Rd[:nn, :nn]
    Ro_m = state.Ro[:nn, :nn]

    # Common signal term: (ᾱ/m) ⟨∇φ̂(v(t)), v(t')⟩ a(t) ∫R_A ds
    t2_2d = (alpha/m) * gr_vt * state.v[:nn] * at * RA_sum

    # C_d  (Eq C.34)
    # t3: -(1/m)η [M_R^d C_d(t',s) + (m-1) M_R^o C_o(t',s)]  summed over s
    #      vectorised: (Cd_m @ MRd)[t']
    # t4: -(1/m)η [M_C^d R_d(t',s) + (m-1) M_C^o R_o(t',s)]  summed over s≤t'
    #      FIX: (Rd_m @ MCd)[t'] — NOT Rd_m.T @ MCd (R_d NOT symmetric)
    dCd = (-nu_n*Cd_n + t2_2d
           - (1.0/m)*eta*(Cd_m@MRd + (m-1)*(Co_m@MRo))
           - (1.0/m)*eta*(Rd_m@MCd + (m-1)*(Ro_m@MCo)))
    state.Cd[n+1, :nn] = Cd_n + eta*dCd
    state.Cd[:nn, n+1] = state.Cd[n+1, :nn]    # symmetry C_d(t,t') = C_d(t',t)
    state.Cd[n+1, n+1] = 1.0                   # spherical constraint

    # C_o  (Eq C.35)
    # FIX: Ro_m @ MCd and Rd_m @ MCo (no transpose)
    dCo = (-nu_n*Co_n + t2_2d
           - (1.0/m)*eta*(Co_m@MRd + Cd_m@MRo + (m-2)*(Co_m@MRo))
           - (1.0/m)*eta*(Ro_m@MCd + Rd_m@MCo + (m-2)*(Ro_m@MCo)))
    state.Co[n+1, :nn] = state.Co[n, :nn] + eta*dCo
    state.Co[:nn, n+1] = state.Co[n+1, :nn]
    state.Co[n+1, n+1] = 2.0*state.Co[n+1, n] - state.Co[n, n]  # linear extrap

    # ── 8. R_d, R_o  (Eqs C.36–C.37) ────────────────────────────────────────
    # R_d  (Eq C.36)
    # FIX: do NOT set Rd[n+1,n+1]=1 — δ only sets sub-diagonal Rd[n+1,n]
    dRd = (-nu_n*state.Rd[n, :n]
           - (1.0/m)*eta*(MRd@Rd_m + (m-1)*(MRo@Ro_m))[:n])
    state.Rd[n+1, :n] = state.Rd[n, :n] + eta*dRd
    state.Rd[n+1, n]  = 1.0    # δ boundary: sub-diagonal only

    # R_o  (Eq C.37)
    dRo = (-nu_n*state.Ro[n, :n]
           - (1.0/m)*eta*(MRd@Ro_m + MRo@Rd_m + (m-2)*(MRo@Ro_m))[:n])
    state.Ro[n+1, :n] = state.Ro[n, :n] + eta*dRo
    state.Ro[n+1, n]  = 0.0

    state.current_step = n + 1