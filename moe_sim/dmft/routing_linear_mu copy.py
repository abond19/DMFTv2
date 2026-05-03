"""
Single-time kernels for E=2 MoE with cluster means µ_c = kappa * U_c.

SIGNAL CONVENTIONS (all boundary-corrected, no extra eta inside sums):
  For Rv_self  = V_0·U_0:  TOTAL = +inv_E * tPhiv_self   [both clusters via Stein]
  For Rv_cross = V_0·U_1:  TOTAL = +inv_E * tPhiv_cross  [always negative, both clusters]
  For P_self   = W_0·U_0:  TOTAL = inv_m * inv_E * tPhi_self

KEY λ-WEIGHTED ROUTER KERNELS (Stein-derived, both clusters):
  tPhiv_self  = gp*H_mu_lam(P_self, mw_s, ml) + gp*P_self*dH_mu_dm1(P_cross, mw_c, ml)
              c=0: Cov(z^w_0, U_0·x|c=0)=P_self → H_mu_lam term
              c=1: Cov(z^w_0, U_0·x|c=1)=P_self → P_self*dH_mu_dm1 term  [was missing]
  tPhiv_cross = -gp*P_self*dH_mu_dm1(P_cross,mw_c,ml) - gp*H_mu_lam(P_self,mw_s,ml)
              c=0: U_1·x|c=0~N(0,1), Stein via Cov(z^w_1,U_1·x)=P_self → -gp*P_self*dH_mu_dm1
              c=1: U_1·x|c=1=kappa+G=lambda_1 (teacher dir), diff_F sign flips
                   → -gp*H_mu_lam(P_self,mw_s,ml)  [symmetry with tPhiv_self c=0]
              [old code used -P_self*dH_mu_dm1 for c=1: missing kappa mean+lambda → 2× error]

KEY λ-WEIGHTED EXPERT KERNEL:
  tPhi_self  = (0.5+a1*kappa*r_D)*dH_mu_dm1_lam(P_self, kappa*P_self, kappa) + ...
             = E[g_0*sigma'(z^w_0)*phi(lambda_0)*lambda_0]

SigmaC kernel Psi (for a_bar ODE):
  psi_abar = hat_Phi_self + hat_Phi_cross   (no 1/E; see solver psi_abar comment)
  hat_Phi_self = E_c0[g_0*sigma(z^w_0)*phi(lambda_0)]  (same cluster, same expert)
  hat_Phi_cross = E_c1[g_0*sigma(z^w_0)*phi(lambda_1)] (cross-cluster, SAME expert 0 weight)
  hat_Phi_cs    = E_c1[g_0*sigma(z^w_1)*phi(lambda_1)] (cross-cluster, cross-expert weight)
    NOTE: hat_Phi_cs uses expert-1 weights (z^w_1), not expert-0 weights!
          hat_Phi_cross uses expert-0 weights on cluster-1 data.

PV COUPLING CORRECTIONS (Bug 5 fix):
  When Pv_self(t,t) = E[w_{l,e}·v_e] != 0, the noise part of g_0 = a1*(v0-v1)·x/sqrt(2)
  correlates with z^w_0 = w_{l,0}·x via Cov(g_0^noise, z^w_0) = a1*(Pv_self-Pv_cross)/sqrt(2).
  By Stein's lemma, this adds:
    δhat_Phi_self  = +a1*(Pv_self-Pv_cross)/sqrt(2) * dH_mu_dm1(P_self, mw_s, ml)
    δhat_Phi_cross = +a1*(Pv_self-Pv_cross)/sqrt(2) * dH_mu_dm1(P_cross, mw_c, ml)
    δhat_Phi_cs    = -a1*(Pv_self-Pv_cross)/sqrt(2) * dH_mu_dm1(P_self, mw_s, ml)
      [cross-expert sign flip: Cov(g_0^noise, z^w_1) = -(Pv_self-Pv_cross)/sqrt(2)]
  These grow from ~1% at t=0 to ~3% at t=8 and affect both SigmaC and the a_bar ODE.
"""

import numpy as np
from scipy.special import erf
from dmft.kernels_e1 import (
    H_mu, dH_mu_dm1, dH_mu_dm2, H_mu_prime,
    H_mu_lam, dH_mu_dm1_lam, Phi_target_mu,
    K_phi_phip_lam,
    _WW, _ZZ,          # quadrature nodes for SC cross kernel
)


def compute_kernels_mu(P_self, P_cross, Rv_self, Rv_cross, a1, kappa,
                       pv_self=0.0, pv_cross=0.0):
    """All single-time kernels at a given time step.

    pv_self, pv_cross: diagonal values of Pv_self(t,t) and Pv_cross(t,t).
    These enter via the Pv coupling correction (Bug 5): the noise component of g_0
    correlates with z^w through Cov(g_0^noise, z^w) = a1*(Pv_self-Pv_cross)/sqrt(2),
    which generates additional Stein contributions to the hat_Phi kernels.
    """
    gp  = a1 / np.sqrt(2.0)
    r_D = (Rv_self - Rv_cross) / np.sqrt(2.0)

    mw_s = kappa * P_self    # mean of z^w for cluster-0 data (self-cluster)
    mw_c = kappa * P_cross   # mean of z^w for cluster-1 data (cross-cluster)
    ml   = kappa              # mean of lambda for both clusters

    # ── Pv coupling coefficient (Bug 5 fix) ──────────────────────────────────
    # Cov(g_0^noise, z^w_e0) = a1*(Pv_self-Pv_cross)/sqrt(2)
    # Generates extra Stein terms in all hat_Phi kernels.
    delta_pv = a1 * (pv_self - pv_cross) / np.sqrt(2.0)

    # ── λ-weighted router kernels ─────────────────────────────────────────────
    #
    # Physical signal = gp * E_all[y * (Fe0−Fe1) * U_e·x], split by cluster.
    #
    # tPhiv_self  (signal for Rv_self = V_0·U_0):
    #   c=0 contributes: Cov(z^w_0, U_0·x|c=0)=P_self via Stein
    #                    → gp * H_mu_lam(P_self, mw_s, ml)                  [main]
    #   c=1 contributes: Cov(z^w_0, U_0·x|c=1)=P_self via Stein (U_0·x~N(0,1))
    #                    → gp * P_self * dH_mu_dm1(P_cross, mw_c, ml)       [c=1 fix]
    #   (Fe1 under c=1 has Cov(z^w_1, U_0·x|c=1)=0 → no contribution)
    #
    # tPhiv_cross (signal for Rv_cross = V_0·U_1, always NEGATIVE):
    #   c=0 contributes: U_1·x|c=0 = N(0,1) (no teacher signal).
    #                    Stein on G_1=U_1·xi via Cov(z^w_1, G_1)=P_self gives
    #                    E[phi(lambda_0)*(-a*phi'(z^w_1))|c=0] = -P_self*dH_mu_dm1(P_cross,mw_c,ml)
    #                    (dH_mu_dm1 = E[phi'(z^w_1)*phi(lambda_0)], no extra lambda)
    #                    → −gp * P_self * dH_mu_dm1(P_cross, mw_c, ml)      [c=0: CORRECT]
    #   c=1 contributes: U_1·x|c=1 = kappa+G_1 = lambda_1 (teacher direction!).
    #                    By cluster symmetry with tPhiv_self c=0:
    #                    diff_F_c1 ≈ −a*phi(z^w_1) [sign flip vs c=0], so
    #                    E[phi(lambda_1)*(−a*phi(z^w_1))*lambda_1|c=1] = −H_mu_lam(P_self,mw_s,ml)
    #                    → −gp * H_mu_lam(P_self, mw_s, ml)                 [c=1: FIXED]
    #
    # OLD (WRONG): c=1 used -gp*P_self*dH_mu_dm1(P_self,mw_s,ml) which is
    #   E[phi'(z^w_1)*phi(lambda_1)] — missing the kappa mean term and lambda_1 factor.
    #   This gave |tPhiv_cross| ≈ 0.47*tPhiv_self.  GF measures ≈ 1.00*tPhiv_self. ✓
    tPhiv_self = (gp * H_mu_lam(P_self, mw_s, ml)
                  + gp * P_self * dH_mu_dm1(P_cross, mw_c, ml))

    tPhiv_cross = (-gp * P_self * dH_mu_dm1(P_cross, mw_c, ml)   # c=0: correct
                   - gp * H_mu_lam(P_self, mw_s, ml))             # c=1: FIXED

    # ── λ-weighted expert kernels ─────────────────────────────────────────────
    # The full cluster-averaged P_self and P_cross signals require contributions
    # from BOTH clusters.  The cross-cluster terms come from the covariance of
    # the router g_0 with the "wrong" teacher direction U_c·x:
    #
    #   Cov(g_0, U_1·x | c=0) = a1 * Cov(disc, U_1·x | c=0)
    #                          = a1 * (v_0-v_1)·U_1/√2 = -a1*r_D          [NEGATIVE]
    #   Cov(g_0, U_0·x | c=1) = a1 * Cov(disc, U_0·x | c=1) = +a1*r_D    [POSITIVE]
    #
    # Applying Stein's lemma to the U_c·x factor (which has mean 0 in the
    # off-cluster data):
    #
    #   E_c0[g_0*phi(λ_0)*σ'(z^w_0)*(U_1·x)]
    #     = Cov(g_0, U_1·x|c=0) · E_c0[φ(λ_0)·σ'(z^w_0)]
    #     = -a1*r_D · dH_mu_dm1(P_self, mw_s, ml)
    #   (The E[g_0] factor does NOT appear here; Stein acts on U_1·x directly.)
    #
    #   E_c1[g_0*phi(λ_1)*σ'(z^w_0)*(U_0·x)]
    #     = +a1*r_D · dH_mu_dm1(P_cross, mw_c, ml)
    #
    # These have been verified by 10M-sample Monte Carlo.

    # tPhi_self = E_c0[g_0·σ'(z^w_0)·φ(λ_0)·λ_0]   (cluster-0, λ-weighted, main term)
    #           + a1*r_D · dH_mu_dm1(P_cross, mw_c, ml)  (cluster-1 correction, positive)
    tPhi_self = ((0.5 + a1*kappa*r_D) * dH_mu_dm1_lam(P_self, mw_s, ml)
                 + a1*r_D * H_mu_prime(P_self, mw_s, ml) * kappa
                 + a1*r_D * dH_mu_dm1(P_cross, mw_c, ml))

    # tPhi_cross = E_c1[g_0·σ'(z^w_0)·φ(λ_1)·λ_1]   (cluster-1, λ-weighted, main term)
    #            - a1*r_D · dH_mu_dm1(P_self, mw_s, ml)  (cluster-0 Stein correction, NEGATIVE)
    #
    # T2 fix: the Cov(G_0, y·σ'·λ_1 | c=1) correction has TWO parts from Stein on G_λ1:
    #   ∂/∂λ_1 [φ(λ_1)·σ'(z^w_0)·λ_1] = [φ(λ_1) + λ_1·φ'(λ_1)] · σ'
    #
    # The original T2 = -a1*r_D·H_mu_prime·κ captures the λ_1·φ'(λ_1) part only.
    # The missing φ(λ_1) part contributes:
    #   -a1*r_D · E_c1[φ(λ_1)] · E_c1[σ'(z^w_0)] = -a1*r_D · dH_mu_dm1(P_cross, mw_c, ml)
    # (independent because z^w_0 ⊥ λ_1 at P_cross≈0).
    # This term is O(r_D), not O(r_D²), so it is structurally missing, not a
    # higher-order correction.  It closes ~33% of the observed P_cross gap at t=9.
    tPhi_cross = ((0.5 - a1*kappa*r_D) * dH_mu_dm1_lam(P_cross, mw_c, ml)
                  - a1*r_D * H_mu_prime(P_cross, mw_c, ml) * kappa
                  - a1*r_D * dH_mu_dm1(P_cross, mw_c, ml)   # NEW: missing φ(λ_1)·σ' term
                  - a1*r_D * dH_mu_dm1(P_self,  mw_s, ml))  # c=0 Stein correction

    # ── SigmaC / a_bar kernels (NO lambda factor) ─────────────────────────────
    # hat_Phi_self = E_c0[g_0*sigma(z^w_0)*phi(lambda_0)]
    # Pv correction: +delta_pv * dH_mu_dm1(P_self, mw_s, ml)
    #   from Stein: Cov(g_0^noise, z^w_0) * E[sigma'(z^w_0)*phi(lambda_0)]
    hat_Phi_self = ((0.5 + a1*kappa*r_D) * H_mu(P_self, mw_s, ml)
                    + a1*r_D * dH_mu_dm2(P_self, mw_s, ml)
                    + delta_pv * dH_mu_dm1(P_self, mw_s, ml))   # Bug 5 fix

    # hat_Phi_cross = E_c1[g_0*sigma(z^w_0)*phi(lambda_1)]
    # Uses same expert-0 weight (z^w_0) on cluster-1 data. ≈ 0 at P_cross=0.
    # Pv correction: same Cov(g_0^noise, z^w_0) sign, so same delta_pv sign.
    hat_Phi_cross = ((0.5 - a1*kappa*r_D) * H_mu(P_cross, mw_c, ml)
                     - a1*r_D * dH_mu_dm2(P_cross, mw_c, ml)
                     + delta_pv * dH_mu_dm1(P_cross, mw_c, ml))  # Bug 5 fix

    # hat_Phi_cs = E_c1[g_0*sigma(z^w_1)*phi(lambda_1)] [cross-cluster, cross-expert weight]
    # Uses expert-1 weight z^w_1. Cov(g_0^noise, z^w_1) = -(Pv_self-Pv_cross)/sqrt(2)
    # (opposite sign because (v0-v1)·w_1 = Pv_cross - Pv_self).
    # Pv correction: -delta_pv * dH_mu_dm1(P_self, mw_s, ml)
    hat_Phi_cs = ((0.5 - a1*kappa*r_D) * H_mu(P_self, mw_s, ml)
                  - a1*r_D * dH_mu_dm2(P_self, mw_s, ml)
                  - delta_pv * dH_mu_dm1(P_self, mw_s, ml))      # Bug 5 fix

    # ── Self-consistency correction kernels (t=0 only) ───────────────────────
    # sc_K_phi_phip: base kernel for P_self SC correction (G_s factor applied
    #   in solver).  The full correction is ā₀·G_s·sc_K_phi_phip.
    sc_K_phi_phip = K_phi_phip_lam(P_self, mw_s, ml)

    # sc_K_cross: base kernel for P_cross SC correction.
    # For cluster-1 data at P_cross≈0 the SC term factorises as:
    #   E[phi'(mw_c+G)] · (κ·E[phi(mw_s+G)] + P_self·E[phi'(mw_s+G)])
    # The G_{co} prefactor (= a₀²−a₁²(1+κ²rD²)) is applied in solver.
    _phip_wc = float(np.dot(_WW, np.sqrt(2./np.pi)*np.exp(-0.5*(mw_c + _ZZ)**2)))
    _phi_ws  = float(np.dot(_WW, erf((mw_s + _ZZ)/np.sqrt(2.))))
    _phip_ws = float(np.dot(_WW, np.sqrt(2./np.pi)*np.exp(-0.5*(mw_s + _ZZ)**2)))
    sc_K_cross = _phip_wc * (ml * _phi_ws + P_self * _phip_ws)

    return {
        'tPhiv_self':    tPhiv_self,
        'tPhiv_cross':   tPhiv_cross,
        'tPhi_self':     tPhi_self,
        'tPhi_cross':    tPhi_cross,
        'hat_Phi_self':  hat_Phi_self,
        'hat_Phi_cross': hat_Phi_cross,
        'hat_Phi_cs':    hat_Phi_cs,
        'Phi_target':    Phi_target_mu(kappa),
        'sc_K_phi_phip': sc_K_phi_phip,   # E[phi·phi'(z^w)·λ] — P_self SC
        'sc_K_cross':    sc_K_cross,       # E[phi'(z^w_0)]·E[phi(z^w_1)·λ_1] — P_cross SC
    }