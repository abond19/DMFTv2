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

KEY λ-WEIGHTED EXPERT KERNELS:
  tPhi_cross = (0.5-a1*kappa*r_D)*dH_mu_dm1_lam(P_cross, mw_c, ml)  [cluster-1 main]
             - a1*r_D * H_mu_prime(P_cross, mw_c, ml) * kappa/2     [c1 Stein: E[σ'·σ'(λ)·λ]]
             - a1*r_D * dH_mu_dm1(P_cross, mw_c, ml)                [c1 Stein: E[σ'·σ(λ)]]
             - a1*r_D * dH_mu_dm1(P_self,  mw_s, ml)                [c0 Stein correction]
  tPhi_self  = (0.5+a1*kappa*r_D)*dH_mu_dm1_lam(P_self, mw_s, ml)   [cluster-0 main]
             + a1*r_D * H_mu_prime(P_self, mw_s, ml) * kappa/2      [c0 Stein: E[σ'·σ'(λ)·λ]]
             + a1*r_D * dH_mu_dm1(P_self,  mw_s, ml)                [c0 Stein: E[σ'·σ(λ)]]
             + a1*r_D * dH_mu_dm1(P_cross, mw_c, ml)                [c1 Stein correction]

  KEY: Stein corrections use r_D = (Rv_self-Rv_cross)/sqrt(2) [the DIFFERENCE].
  Cov(disc, lam_1) = (V_0-V_1)·U_1/sqrt(2) = (Rv_cross-Rv_self)/sqrt(2) = -r_D.
  Verified by correct joint-distribution MC (with r_D in covariance matrix) at all Δv.

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
                       pv_self=0.0, pv_cross=0.0, disc_ons=0.0):
    """All single-time kernels at a given time step.

    pv_self, pv_cross: diagonal values of Pv_self(t,t) and Pv_cross(t,t).
    These enter via the Pv coupling correction (Bug 5): the noise component of g_0
    correlates with z^w through Cov(g_0^noise, z^w) = a1*(Pv_self-Pv_cross)/sqrt(2),
    which generates additional Stein contributions to the hat_Phi kernels.
    """
    gp  = a1 / np.sqrt(2.0)
    r_D = (Rv_self - Rv_cross) / np.sqrt(2.0)

    # ── Router Onsager correction (disc_ons) ──────────────────────────────
    # disc_ons = delta(E_train[disc|c=1]) <= 0: training-data routing correction.
    # Applies ONLY to tPhi kernels (expert weight signals P_self, P_cross).
    # Does NOT apply to hat_Phi kernels (a_bar Volterra source):
    #   hat_Phi is a POPULATION expectation (d→∞ limit); disc_ons is a
    #   finite-n training-data correction that vanishes as d→∞. Applying
    #   it to hat_Phi inflates psi_abar, making a_bar grow too fast.
    kappa_safe = kappa if abs(kappa) > 1e-10 else 1e-10
    r_D_c = r_D - disc_ons / kappa_safe   # corrected for tPhi_cross (c=1 signal)
    r_D_s = r_D - disc_ons / kappa_safe   # corrected for tPhi_self  (c=0 signal)


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
    # c=1 contribution only — used for disc_ons Onsager source.
    # The c=0 Stein term is not part of E_c1[res·DeltaFe].
    tPhiv_cross_c1 = -gp * H_mu_lam(P_self, mw_s, ml)
    # Stein correction term from H_mu_lam = kappa*H_mu + P_self*H_mu_prime.
    # The P_self*H_mu_prime part is NOT part of E_c1[res*DeltaFe]; subtract it.
    # Source_correct = (sqrt2/kappa)*tPhiv_c_c1 + (a1*P_self*H_mu_prime)/(E*kappa)
    H_mu_prime_c1 = H_mu_prime(P_self, mw_s, ml)   # E[sigma'(z^w_1)*sigma'(lambda_1)]

    # ── λ-weighted expert kernels ─────────────────────────────────────────────
    # The full cluster-averaged P_self and P_cross signals require contributions
    # from BOTH clusters.  The cross-cluster terms come from the covariance of
    # the router g_0 with the "wrong" teacher direction U_c·x:
    #
    #   Cov(disc, lam_0) = (v0-v1)·U_0/sqrt(2) = (Rv_self-Rv_cross)/sqrt(2) = +r_D
    #   Cov(disc, lam_1) = (v0-v1)·U_1/sqrt(2) = (Rv_cross-Rv_self)/sqrt(2) = -r_D
    #
    # NOTE: the covariance uses r_D = (Rv_self-Rv_cross)/sqrt(2) [the DIFFERENCE],
    # because V_0·U_1 = Rv_cross and V_1·U_1 = Rv_self, so (V_0-V_1)·U_1 = Rv_cross-Rv_self.
    # Verified by correct joint-distribution MC at all Δv values.
    #
    # tPhi_self = E[G_0·σ'(z^w_0)·φ(λ_0)·λ_0]  (cluster-0 main + cross-cluster Stein):
    #   main term: (0.5+a1κr_D) * dH_mu_dm1_lam(P_self, mw_s, ml)
    #   c=0 Stein (Cov(disc,lam_0)=+r_D): +a1*r_D * {H_mu_prime*κ/2 + dH_mu_dm1(P_self)}
    #   c=1 Stein (Cov(disc,lam_0_c1)=+r_D, lam_0 has mean 0 for c=1 data):
    #              +a1*r_D * dH_mu_dm1(P_cross, mw_c, ml)
    tPhi_self = ((0.5 + a1*kappa*r_D_s) * dH_mu_dm1_lam(P_self, mw_s, ml)  # Onsager
                 + a1*r_D * H_mu_prime(P_self, mw_s, ml) * kappa / 2.0
                 + a1*r_D * dH_mu_dm1(P_self, mw_s, ml)
                 + a1*r_D * dH_mu_dm1(P_cross, mw_c, ml))

    # tPhi_cross = E[G_0·σ'(z^w_0)·φ(λ_1)·λ_1]  (cluster-1 main + cross-cluster Stein):
    #   main term: (0.5-a1κr_D) * dH_mu_dm1_lam(P_cross, mw_c, ml)
    #   c=1 Stein (Cov(disc,lam_1)=-r_D): -a1*r_D * {H_mu_prime*κ/2 + dH_mu_dm1(P_cross)}
    #             [H_mu_prime_lam = κ/2 * H_mu_prime, by Stein on lam_1 in E[σ'·σ'(lam1)·lam1]]
    #   c=0 Stein (Cov(disc,lam_1_c0)=-r_D, lam_1 has mean 0 for c=0 data):
    #             -a1*r_D * dH_mu_dm1(P_self, mw_s, ml)
    # Verified by correct joint-distribution MC (r_D covariance) at all Δv values.
    tPhi_cross = ((0.5 - a1*kappa*r_D_c) * dH_mu_dm1_lam(P_cross, mw_c, ml)  # Onsager
                  - a1*r_D * H_mu_prime(P_cross, mw_c, ml) * kappa / 2.0
                  - a1*r_D * dH_mu_dm1(P_cross, mw_c, ml)
                  - a1*r_D * dH_mu_dm1(P_self,  mw_s, ml))

    # ── SigmaC / a_bar kernels (NO lambda factor) ─────────────────────────────
    # hat_Phi_self = E_c0[g_0*sigma(z^w_0)*phi(lambda_0)]
    # Stein on disc via Cov(disc,lam_0)=+r_D:
    #   a1*r_D * E[sigma(z^w_0)*phi'(lam_0)] = a1*r_D * dH_mu_dm2(P_self, mw_s, ml)
    # Pv correction: +delta_pv * dH_mu_dm1(P_self, mw_s, ml)
    hat_Phi_self = ((0.5 + a1*kappa*r_D) * H_mu(P_self, mw_s, ml)   # population OP: no disc_ons
                    + a1*r_D * dH_mu_dm2(P_self, mw_s, ml)
                    + delta_pv * dH_mu_dm1(P_self, mw_s, ml))   # Bug 5 fix

    # hat_Phi_cross = E_c1[g_0*sigma(z^w_0)*phi(lambda_1)]
    # Uses same expert-0 weight (z^w_0) on cluster-1 data. ≈ 0 at P_cross=0.
    # Stein on disc via Cov(disc,lam_1)=-r_D:
    #   -a1*r_D * E[sigma(z^w_0)*phi'(lam_1)] = -a1*r_D * dH_mu_dm2(P_cross, mw_c, ml)
    # Pv correction: same Cov(g_0^noise, z^w_0) sign, so same delta_pv sign.
    hat_Phi_cross = ((0.5 - a1*kappa*r_D) * H_mu(P_cross, mw_c, ml)   # population OP: no disc_ons
                     - a1*r_D * dH_mu_dm2(P_cross, mw_c, ml)
                     + delta_pv * dH_mu_dm1(P_cross, mw_c, ml))  # Bug 5 fix

    # hat_Phi_cs = E_c1[g_0*sigma(z^w_1)*phi(lambda_1)] [cross-cluster, cross-expert weight]
    # Uses expert-1 weight z^w_1 (self-aligned for cluster-1, so mw_s mean).
    # Stein on disc via Cov(disc,lam_1)=-r_D:
    #   -a1*r_D * E[sigma(z^w_1)*phi'(lam_1)] = -a1*r_D * dH_mu_dm2(P_self, mw_s, ml)
    # Pv correction: Cov(g_0^noise, z^w_1) = -(Pv_self-Pv_cross)/sqrt(2) (opposite sign).
    hat_Phi_cs = ((0.5 - a1*kappa*r_D) * H_mu(P_self, mw_s, ml)   # population OP: no disc_ons
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
        'tPhiv_cross_c1': tPhiv_cross_c1,  # c=1 only (for disc_ons source)
        'H_mu_prime_c1':  H_mu_prime_c1,   # Stein correction term for disc_ons
        'tPhi_self':     tPhi_self,
        'tPhi_cross':    tPhi_cross,
        'hat_Phi_self':  hat_Phi_self,
        'hat_Phi_cross': hat_Phi_cross,
        'hat_Phi_cs':    hat_Phi_cs,
        'Phi_target':    Phi_target_mu(kappa),
        'sc_K_phi_phip': sc_K_phi_phip,   # E[phi·phi'(z^w)·λ] — P_self SC
        'sc_K_cross':    sc_K_cross,       # E[phi'(z^w_0)]·E[phi(z^w_1)·λ_1] — P_cross SC
    }